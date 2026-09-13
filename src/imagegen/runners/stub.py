"""A runner that loads no weights.

It honours the manifest contract exactly, so the queue, the API, the sweep tool and
the contact sheet can all be exercised and tested on a machine that must not run a
model. The image is a deterministic function of the manifest: the same job twice
produces byte-identical output, and different seeds produce visibly different images
so a contact sheet still means something.

An edit or mask job blends its input image in, which makes lineage visible at a
glance when reading a sheet of stub output.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imagegen.manifest import Manifest
from imagegen.runners.base import BaseRunner

DELAY_ENV = "IMAGEGEN_STUB_DELAY"
LABEL_HEIGHT = 92


def _stable_seed(material: str) -> int:
    """A stable integer derived from everything that would change a real render."""
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _gradient(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Two random corner colours interpolated across the frame."""
    start = rng.integers(24, 232, size=3).astype(np.float32)
    end = rng.integers(24, 232, size=3).astype(np.float32)
    horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    ramp = (horizontal + vertical) / 2.0
    canvas = start[None, None, :] * (1.0 - ramp) + end[None, None, :] * ramp
    return Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "RGB")


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions without a sized default font
        return ImageFont.load_default()


def _label(image: Image.Image, lines: list[str]) -> None:
    """A legible caption band, so a contact sheet of stubs is readable."""
    draw = ImageDraw.Draw(image, "RGBA")
    band = min(LABEL_HEIGHT, image.height)
    draw.rectangle((0, image.height - band, image.width, image.height), fill=(0, 0, 0, 170))
    font = _font(18)
    offset = image.height - band + 8
    for line in lines[:3]:
        draw.text((10, offset), line[:64], font=font, fill=(255, 255, 255, 255))
        offset += 26


class StubRunner(BaseRunner):
    name: ClassVar[str] = "stub"

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        image_path: Path | None = None,
        image_strength: float | None = None,
    ) -> Image.Image:
        delay = float(os.environ.get(DELAY_ENV, "0") or 0)
        if delay > 0:
            time.sleep(delay)

        material = f"{prompt}|{seed}|{steps}|{guidance}|{width}x{height}"
        rng = np.random.default_rng(_stable_seed(material))
        image = _gradient(width, height, rng)

        if image_path is not None:
            source = Image.open(image_path).convert("RGB")
            source = source.resize((width, height), Image.Resampling.LANCZOS)
            image = Image.blend(source, image, 0.45)

        _label(
            image,
            [
                f"seed {seed}  steps {steps}  {width}x{height}",
                prompt,
                f"profile {self.profile.name}  STUB, no model was run",
            ],
        )
        return image

    def render(self, manifest: Manifest) -> tuple[Image.Image, dict[str, Any]]:
        image, metrics = super().render(manifest)
        return image, {"stub": True, **metrics}
