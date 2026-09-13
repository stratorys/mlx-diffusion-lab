"""The masked edit, start to finish.

Shared by every runner, because the choreography is the same whatever produces the
pixels: find the region from the phrase, crop around it, erase what was there, let
the model fill the crop, then paste back through the mask.

The runner only has to know how to turn a prompt and an optional conditioning image
into one image. Everything else lives here, which is why this path is testable with
no model at all.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from imagegen import geometry as geo
from imagegen.manifest import Manifest
from imagegen.segment import Segmenter, derive_mask
from imagegen.storage import atomic_image

DIAGNOSTICS = "diagnostics"


def diagnostics_dir(manifest: Manifest) -> Path:
    if manifest.diagnostics is not None:
        return Path(manifest.diagnostics)
    output = Path(manifest.output)
    return output.parent / DIAGNOSTICS / output.stem


def run_masked_edit(runner, manifest: Manifest, segmenter: Segmenter) -> tuple[Image.Image, dict[str, Any]]:
    """Derive the region, repaint it, and prove the rest of the frame did not move."""
    settings = manifest.mask
    if settings is None or manifest.input_image is None:
        raise ValueError("a masked edit needs mask settings and an input image")

    source = geo.load_rgb(manifest.input_image)
    target = manifest.segmentation_target
    diagnostics = diagnostics_dir(manifest)

    heatmap = segmenter.heatmap(source, target)
    atomic_image(
        diagnostics / "heatmap.png",
        Image.fromarray((heatmap.clip(0, 1) * 255).astype("uint8"), "L"),
        format="PNG",
    )

    # The heatmap is already on disk above, so a rejection below is still diagnosable.
    mask, report = derive_mask(heatmap, settings, target, segmenter.name)
    atomic_image(diagnostics / "mask.png", mask, format="PNG")

    box = geo.crop_region(mask, settings.padding, source.size)
    crop = source.crop(box)
    mask_crop = mask.crop(box)
    target_size = settings.size or geo.automatic_target_size(crop.size)
    width, height = geo.work_size(crop.size, target_size)

    erased = geo.erased_condition(crop, mask_crop, manifest.seed)
    condition = diagnostics / "condition.png"
    atomic_image(condition, erased.resize((width, height), Image.Resampling.LANCZOS), format="PNG")
    atomic_image(diagnostics / "source_crop.png", crop, format="PNG")

    generated = runner.generate(
        prompt=manifest.prompt,
        seed=manifest.seed,
        steps=manifest.steps,
        guidance=manifest.guidance,
        width=width,
        height=height,
        image_path=condition,
        image_strength=manifest.image_strength,
    )
    atomic_image(diagnostics / "generated_crop.png", generated, format="PNG")

    result = geo.composite_local(source, generated, mask_crop, box)
    effective = geo.full_image_mask(mask_crop, box, source.size)
    atomic_image(diagnostics / "composite_mask.png", effective, format="PNG")

    metrics = geo.difference_metrics(source, result, effective)
    if metrics["outside_mask_max_delta"] != 0:
        raise RuntimeError(
            "the composite changed pixels outside the mask, which must be impossible; "
            f"maximum delta {metrics['outside_mask_max_delta']}"
        )
    if metrics["masked_pixels_changed_over_10_percent"] < 5.0:
        report.warnings.append(
            "the masked region barely changed; the edit may not have taken effect"
        )

    return result, {
        "mask": report.as_dict(),
        "crop_box": list(box),
        "work_size": [width, height],
        "diagnostics": str(diagnostics),
        **metrics,
    }
