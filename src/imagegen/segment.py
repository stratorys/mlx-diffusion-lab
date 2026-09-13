"""Deriving a mask from a phrase.

The masked endpoint takes an image and a prompt, so the region has to come from the
text. CLIPSeg does that for open vocabulary, which is the point: this module holds no
list of nameable regions, and gains nothing from one.

It runs on the CPU deliberately. The model is small, and keeping it off the Metal
device means it never competes for memory with the generator.

Every rejection is explicit. A mask that is empty, that covers almost the whole
frame, or that the segmenter is not confident about, fails the job with a reason
rather than quietly producing a wrong image.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter

from imagegen.manifest import MaskSettings

SEGMENTER_ENV = "IMAGEGEN_SEGMENTER"
DEFAULT_SEGMENTER = "clipseg"
CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"

# Component analysis runs on a reduced grid. Exact blob shape does not matter here,
# only which blob is which, and this keeps the cost negligible.
ANALYSIS_SIDE = 256
MAX_COMPONENTS = 8


class MaskRejected(ValueError):
    """The derived mask is not safe to edit through."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class MaskReport:
    target: str
    segmenter: str
    threshold_used: float
    thresholds_tried: list[float]
    peak: float
    area_fraction: float
    components: int
    bbox: tuple[int, int, int, int] | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "segmenter": self.segmenter,
            "threshold_used": round(self.threshold_used, 4),
            "thresholds_tried": [round(value, 4) for value in self.thresholds_tried],
            "peak": round(self.peak, 4),
            "area_fraction": round(self.area_fraction, 6),
            "components": self.components,
            "bbox": list(self.bbox) if self.bbox else None,
            "warnings": list(self.warnings),
        }


class Segmenter(Protocol):
    name: str

    def heatmap(self, image: Image.Image, target: str) -> np.ndarray:
        """A float array in [0, 1] at the image's size, high where the phrase matches."""


class BoxSegmenter:
    """A deterministic centre box. Exists so the pipeline is testable without torch."""

    name = "box"

    def __init__(self, fraction: float = 0.35):
        self.fraction = fraction

    def heatmap(self, image: Image.Image, target: str) -> np.ndarray:
        width, height = image.size
        field_ = np.zeros((height, width), dtype=np.float32)
        half_w = int(width * self.fraction / 2)
        half_h = int(height * self.fraction / 2)
        field_[
            max(0, height // 2 - half_h) : height // 2 + half_h,
            max(0, width // 2 - half_w) : width // 2 + half_w,
        ] = 0.95
        return field_


class ClipSegSegmenter:
    """CLIPSeg, loaded lazily and pinned to the CPU."""

    name = "clipseg"

    def __init__(self, model_id: str = CLIPSEG_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self._processor = CLIPSegProcessor.from_pretrained(self.model_id)
        self._model = CLIPSegForImageSegmentation.from_pretrained(self.model_id)
        self._model.to("cpu").eval()

    def heatmap(self, image: Image.Image, target: str) -> np.ndarray:
        self.load()
        torch = self._torch
        inputs = self._processor(text=[target], images=[image.convert("RGB")], return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        if logits.dim() == 2:
            logits = logits[None, None]
        elif logits.dim() == 3:
            logits = logits[:, None]
        probabilities = torch.sigmoid(logits)
        resized = torch.nn.functional.interpolate(
            probabilities, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
        return resized[0, 0].to(torch.float32).numpy()


def build_segmenter(name: str | None = None) -> Segmenter:
    chosen = (name or os.environ.get(SEGMENTER_ENV) or DEFAULT_SEGMENTER).lower()
    if chosen == "box":
        return BoxSegmenter()
    if chosen == "clipseg":
        return ClipSegSegmenter()
    raise ValueError(f"unknown segmenter {chosen}; known: clipseg, box")


# --- turning a heatmap into a mask -------------------------------------------------


def threshold_ladder(threshold: float) -> list[float]:
    """Descending thresholds to try before giving up.

    A phrase the segmenter is only mildly confident about still marks the right
    region; insisting on the first threshold would throw that away.
    """
    return [threshold, round(threshold * 0.75, 4), round(threshold * 0.5, 4)]


def _grow(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Flood one connected region outward from a seed, four-connected."""
    current = seed
    while True:
        expanded = current.copy()
        expanded[1:, :] |= current[:-1, :]
        expanded[:-1, :] |= current[1:, :]
        expanded[:, 1:] |= current[:, :-1]
        expanded[:, :-1] |= current[:, 1:]
        expanded &= allowed
        if int(expanded.sum()) == int(current.sum()):
            return current
        current = expanded


def components(binary: np.ndarray, limit: int = MAX_COMPONENTS) -> list[np.ndarray]:
    """Connected regions, largest first. Counting stops at the limit."""
    remaining = binary.copy()
    found: list[np.ndarray] = []
    while remaining.any() and len(found) < limit:
        flat = int(np.argmax(remaining))
        seed = np.zeros_like(remaining)
        seed.flat[flat] = True
        region = _grow(seed, binary)
        found.append(region)
        remaining &= ~region
    found.sort(key=lambda region: int(region.sum()), reverse=True)
    return found


def _analysis_grid(binary: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = binary.shape
    scale = ANALYSIS_SIDE / max(height, width)
    if scale >= 1.0:
        return binary, (width, height)
    small = Image.fromarray((binary * 255).astype(np.uint8), "L").resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST
    )
    return np.asarray(small) > 127, (width, height)


def derive_mask(
    heatmap: np.ndarray, settings: MaskSettings, target: str, segmenter: str
) -> tuple[Image.Image, MaskReport]:
    """Threshold, clean up and validate a heatmap into a usable mask."""
    peak = float(heatmap.max()) if heatmap.size else 0.0
    warnings: list[str] = []

    if peak < settings.min_confidence:
        raise MaskRejected(
            "mask_low_confidence",
            f"the segmenter is not confident about {target!r}: peak {peak:.3f} is below "
            f"{settings.min_confidence:.2f}. Try a shorter, more concrete target phrase.",
        )

    tried: list[float] = []
    binary = None
    used = settings.threshold
    for candidate in threshold_ladder(settings.threshold):
        tried.append(candidate)
        attempt = heatmap >= candidate
        if attempt.any():
            binary, used = attempt, candidate
            break
    if binary is None:
        raise MaskRejected(
            "mask_empty",
            f"nothing matched {target!r} even at threshold {tried[-1]:.3f}, so there is "
            "nothing to repaint.",
        )
    if used != settings.threshold:
        warnings.append(f"threshold relaxed from {settings.threshold:g} to {used:g}")

    grid, _ = _analysis_grid(binary)
    regions = components(grid)
    count = len(regions)
    if count > 1:
        largest, second = int(regions[0].sum()), int(regions[1].sum())
        if second > 0.4 * largest:
            warnings.append(
                f"{count} separate regions matched {target!r}; keeping only the largest"
            )

    if count > 1:
        keep = Image.fromarray((regions[0] * 255).astype(np.uint8), "L").resize(
            (binary.shape[1], binary.shape[0]), Image.Resampling.NEAREST
        )
        binary = binary & (np.asarray(keep) > 127)

    coverage = float(binary.mean())
    if coverage > settings.max_coverage:
        raise MaskRejected(
            "mask_too_large",
            f"{target!r} covers {coverage:.0%} of the frame, above the "
            f"{settings.max_coverage:.0%} limit. Repainting nearly everything is a "
            "generate, not a mask.",
        )
    if not binary.any():
        raise MaskRejected("mask_empty", f"nothing remained for {target!r} after cleanup")

    mask = Image.fromarray((binary * 255).astype(np.uint8), "L")
    if settings.dilate:
        mask = mask.filter(ImageFilter.MaxFilter(2 * settings.dilate + 1))
    if settings.feather:
        mask = mask.filter(ImageFilter.GaussianBlur(settings.feather))

    report = MaskReport(
        target=target,
        segmenter=segmenter,
        threshold_used=used,
        thresholds_tried=tried,
        peak=peak,
        area_fraction=coverage,
        components=count,
        bbox=mask.getbbox(),
        warnings=warnings,
    )
    return mask, report
