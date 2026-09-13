"""Mask geometry and compositing.

Ported from the reference masked editor. All of it is plain array work, so it is the
part of the masked pipeline that can be proved correct without a model, and it is
also the part most able to be subtly wrong.

The shape of a masked edit is: crop around the mask with padding, erase the masked
pixels so the model cannot copy what was there, denoise the crop at a sane
resolution, then paste back through the mask. The source stays identical wherever the
mask is zero, and `difference_metrics` exists to prove that rather than assume it.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Box = tuple[int, int, int, int]
Size = tuple[int, int]

MULTIPLE = 16
MIN_TARGET = 512
MAX_TARGET = 1024
# A pixel is inside the mask when it is clearly painted, and outside when it is
# exactly zero. The soft band between the two is blended on purpose and belongs to
# neither bucket, which is what makes the outside guarantee absolute.
INSIDE_FLOOR = 32


class EmptyMaskError(ValueError):
    """The mask selects nothing, so there is nothing to repaint."""


def load_rgb(path: Path | str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def prepare_mask(source: Image.Image | Path | str, size: Size, feather: int = 0) -> Image.Image:
    """A greyscale mask at the image's size. White is repainted."""
    mask = source if isinstance(source, Image.Image) else Image.open(source)
    mask = mask.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.LANCZOS)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    if not mask.getbbox():
        raise EmptyMaskError("the mask is empty, there is nothing to repaint")
    return mask


def crop_region(mask: Image.Image, padding: int, image_size: Size) -> Box:
    """The mask's bounding box, grown by the padding and clamped to the image."""
    bounds = mask.getbbox()
    if bounds is None:
        raise EmptyMaskError("the mask is empty, so it has no bounding box")
    left, top, right, bottom = bounds
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def work_size(crop_size: Size, target: int) -> Size:
    """Denoising dimensions: longest side at the target, each side a multiple of 16."""
    scale = target / max(crop_size)
    width, height = (max(MULTIPLE, round(value * scale / MULTIPLE) * MULTIPLE) for value in crop_size)
    return width, height


def automatic_target_size(crop_size: Size) -> int:
    """Stay native where possible, never below 512 and never above 1024."""
    native = ((max(crop_size) + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    return max(MIN_TARGET, min(MAX_TARGET, native))


def erased_condition(source: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
    """Replace the masked content with deterministic neutral noise.

    Without this the edit model sees what is already there and tends to reproduce it.
    Erasing first is what forces an actual replacement rather than a touch-up.
    """
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    noise = np.random.default_rng(seed).normal(127.0, 18.0, rgb.shape).clip(72, 182)
    erased = np.rint(rgb * (1.0 - alpha) + noise * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(erased, "RGB")


def crop_blend_mask(box: Box, image_size: Size, feather: int) -> Image.Image:
    """A full-crop alpha mask, softened only along edges interior to the image.

    Used when keeping the whole generated crop rather than only the masked part. Edges
    that sit on the image border are not softened, because there is nothing beyond
    them to blend into.
    """
    width, height = box[2] - box[0], box[3] - box[1]
    if feather <= 0:
        return Image.new("L", (width, height), 255)

    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    distances = []
    if box[0] > 0:
        distances.append(np.broadcast_to(x[None, :], (height, width)))
    if box[2] < image_size[0]:
        distances.append(np.broadcast_to((width - 1 - x)[None, :], (height, width)))
    if box[1] > 0:
        distances.append(np.broadcast_to(y[:, None], (height, width)))
    if box[3] < image_size[1]:
        distances.append(np.broadcast_to((height - 1 - y)[:, None], (height, width)))
    if not distances:
        return Image.new("L", (width, height), 255)

    distance = np.minimum.reduce(distances)
    alpha = np.clip(distance / feather, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
    return Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8), "L")


def full_image_mask(local_mask: Image.Image, box: Box, image_size: Size) -> Image.Image:
    """Lift a crop-sized mask back to image coordinates."""
    effective = Image.new("L", image_size, 0)
    effective.paste(local_mask, box[:2])
    return effective


def composite_local(
    source: Image.Image, generated_crop: Image.Image, blend_mask: Image.Image, box: Box
) -> Image.Image:
    """Paste the generated crop back through the mask."""
    result = source.copy()
    target = (box[2] - box[0], box[3] - box[1])
    patch = generated_crop.resize(target, Image.Resampling.LANCZOS)
    result.paste(patch, box[:2], blend_mask)
    return result


def difference_metrics(source: Image.Image, result: Image.Image, mask: Image.Image) -> dict:
    """Prove what changed and, more importantly, what did not.

    `outside_mask_max_delta` must be zero. It is measured only where the mask is
    exactly zero, which is the region the composite is required to leave untouched.
    The soft edge is excluded because it is blended deliberately.
    """
    before = np.asarray(source.convert("RGB"), dtype=np.int16)
    after = np.asarray(result.convert("RGB"), dtype=np.int16)
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    difference = np.abs(before - after)
    per_pixel = difference.mean(axis=2)

    inside = values >= INSIDE_FLOOR
    outside = values == 0
    return {
        "masked_mean_absolute_error": round(float(per_pixel[inside].mean()), 4) if inside.any() else 0.0,
        "masked_pixels_changed_over_10_percent": (
            round(float((per_pixel[inside] > 10).mean() * 100), 4) if inside.any() else 0.0
        ),
        "outside_mask_max_delta": int(difference[outside].max()) if outside.any() else 0,
        "mask_area_fraction": round(float(inside.mean()), 6),
    }
