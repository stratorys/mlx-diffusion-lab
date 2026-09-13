"""What every runner shares.

A runner implements one thing: turn a prompt, and optionally a conditioning image,
into one image. Everything above that is common, which is what keeps the stub and the
real runner honest about each other.

- `run` handles timing, the atomic save and the metadata sidecar.
- `render` dispatches on the job kind.
- the masked path is choreographed in `imagegen.masking` and calls back into
  `generate`, so it works identically under any runner.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from imagegen.config import ResolvedProfile
from imagegen.manifest import Manifest
from imagegen.storage import atomic_image, atomic_json, sha256_text


@dataclass(frozen=True)
class Result:
    output: Path
    metadata: Path
    duration_s: float
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    name: ClassVar[str] = "base"

    def __init__(self, profile: ResolvedProfile, segmenter_name: str | None = None):
        self.profile = profile
        self.segmenter_name = segmenter_name
        self._segmenter = None

    def load(self) -> None:
        """Acquire whatever is expensive. Called once, before the first job."""

    def close(self) -> None:
        """Release whatever was acquired. Called once, at shutdown."""

    @abstractmethod
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
        """One image. The only thing a runner has to know how to do."""

    def segmenter(self):
        """Built on first use, so a server that never masks never loads it."""
        if self._segmenter is None:
            from imagegen.segment import build_segmenter

            self._segmenter = build_segmenter(self.segmenter_name)
        return self._segmenter

    def render(self, manifest: Manifest) -> tuple[Image.Image, dict[str, Any]]:
        if manifest.kind == "mask":
            from imagegen.masking import run_masked_edit

            return run_masked_edit(self, manifest, self.segmenter())

        image = self.generate(
            prompt=manifest.prompt,
            seed=manifest.seed,
            steps=manifest.steps,
            guidance=manifest.guidance,
            width=manifest.width,
            height=manifest.height,
            image_path=Path(manifest.input_image) if manifest.input_image else None,
            image_strength=manifest.image_strength,
        )
        return image, {}

    def run(self, manifest: Manifest) -> Result:
        if manifest.profile != self.profile.name:
            raise ValueError(
                f"this process serves profile {self.profile.name}, "
                f"but the manifest asks for {manifest.profile}"
            )
        started = time.perf_counter()
        image, metrics = self.render(manifest)
        duration = round(time.perf_counter() - started, 3)
        atomic_image(Path(manifest.output), image, format="PNG")
        sidecar = self.sidecar(manifest, metrics, duration)
        atomic_json(Path(manifest.metadata), sidecar)
        return Result(Path(manifest.output), Path(manifest.metadata), duration, metrics)

    def sidecar(self, manifest: Manifest, metrics: dict[str, Any], duration: float) -> dict[str, Any]:
        """Resolved provenance, not just a profile name.

        A catalogue edit six months from now must not silently change what a past
        image claims about itself, so the checkpoint, the encoder and the full ordered
        adapter stack are copied in here rather than referenced.
        """
        return {
            "job_id": manifest.job_id,
            "kind": manifest.kind,
            "runner": self.name,
            "prompt": manifest.prompt,
            "prompt_sha256": sha256_text(manifest.prompt),
            "seed": manifest.seed,
            "steps": manifest.steps,
            "guidance": manifest.guidance,
            "width": manifest.width,
            "height": manifest.height,
            "input_image": str(manifest.input_image) if manifest.input_image else None,
            "mask": manifest.mask.model_dump() if manifest.mask else None,
            "provenance": self.profile.provenance(),
            "metrics": metrics,
            "duration_s": duration,
            "created_at": datetime.now(UTC).isoformat(),
        }
