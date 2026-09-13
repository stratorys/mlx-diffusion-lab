"""The runner contract.

A manifest is the only thing a runner receives. It is written atomically beside the
job, is readable after the fact, and carries everything needed to reproduce the work.
Nothing else crosses the boundary between the queue and a runner.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imagegen.config import Kind
from imagegen.storage import atomic_text


class MaskSettings(BaseModel):
    """How a masked edit derives and applies its region.

    `target` is the phrase handed to the segmenter. It is usually not the generation
    prompt: you segment "the dress" and generate "a blue silk dress". It falls back to
    the prompt when the caller gives only one phrase.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str | None = None
    threshold: float = Field(default=0.4, gt=0.0, lt=1.0)
    dilate: int = Field(default=16, ge=0, le=512)
    feather: int = Field(default=12, ge=0, le=512)
    padding: int = Field(default=112, ge=0, le=2048)
    size: int | None = Field(default=None, ge=256, le=2048)
    max_coverage: float = Field(default=0.85, gt=0.0, le=1.0)
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _size_is_a_model_dimension(self) -> Self:
        if self.size is not None and self.size % 16:
            raise ValueError("size must be a multiple of 16")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Kind
    job_id: int = Field(ge=0)
    profile: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    seed: int = Field(ge=0)
    steps: int = Field(ge=1)
    guidance: float = Field(gt=0)
    width: int = Field(ge=64)
    height: int = Field(ge=64)
    input_image: Path | None = None
    image_strength: float | None = Field(default=None, gt=0.0, le=1.0)
    mask: MaskSettings | None = None
    output: Path
    metadata: Path
    diagnostics: Path | None = None

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> Self:
        if self.kind == "generate":
            if self.input_image is not None:
                raise ValueError("a generate job takes no input image")
            if self.mask is not None:
                raise ValueError("a generate job takes no mask settings")
        else:
            if self.input_image is None:
                raise ValueError(f"a {self.kind} job needs an input image")
        if self.kind == "mask" and self.mask is None:
            raise ValueError("a mask job needs mask settings")
        if self.kind == "edit" and self.mask is not None:
            raise ValueError("an edit job takes no mask settings")
        for side in (self.width, self.height):
            if side % 16:
                raise ValueError("width and height must be multiples of 16")
        return self

    @property
    def segmentation_target(self) -> str:
        """What the segmenter is asked to find. Never empty for a mask job."""
        if self.mask is None:
            raise ValueError("this manifest has no mask settings")
        return self.mask.target or self.prompt

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

    def write(self, path: Path) -> Path:
        return atomic_text(path, self.to_json())

    @classmethod
    def read(cls, path: Path) -> Manifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
