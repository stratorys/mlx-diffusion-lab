"""Request and response models.

Every request forbids unknown fields. A misspelled parameter is a 422 rather than a
silently ignored setting that quietly changes what you get back.

Optional numeric fields default to None and are filled in from the engine, so the
defaults live in the catalogue and not in two places.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from imagegen.jobs import JobKind, Status


def _compact(value: str) -> str:
    return " ".join(value.split())


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    seed: int | None = Field(default=None, ge=0)
    steps: int | None = Field(default=None, ge=1, le=200)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    guidance: float | None = Field(default=None, gt=0)
    force: bool = Field(default=False, description="redo the work even if this exact image exists")

    @field_validator("prompt")
    @classmethod
    def _prompt_has_content(cls, value: str) -> str:
        compacted = _compact(value)
        if not compacted:
            raise ValueError("prompt must not be empty")
        return compacted

    @model_validator(mode="after")
    def _frame_is_model_sized(self) -> Self:
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        return self


class GenerateRequest(BaseRequest):
    pass


class ParentedRequest(BaseRequest):
    parent_id: int = Field(ge=0, description="a previous job, or an uploaded image")


class EditRequest(ParentedRequest):
    image_strength: float | None = Field(default=None, gt=0.0, le=1.0)


class MaskRequest(ParentedRequest):
    target: str | None = Field(
        default=None,
        description="what the segmenter should find. Defaults to the prompt, which is "
        "usually worse: you segment 'the dress' and generate 'a blue silk dress'.",
    )
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    dilate: int | None = Field(default=None, ge=0, le=512)
    feather: int | None = Field(default=None, ge=0, le=512)
    padding: int | None = Field(default=None, ge=0, le=2048)
    size: int | None = Field(default=None, ge=256, le=2048)

    @field_validator("target")
    @classmethod
    def _target_has_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        compacted = _compact(value)
        if not compacted:
            raise ValueError("target must not be empty when given")
        return compacted

    @model_validator(mode="after")
    def _size_is_model_sized(self) -> Self:
        if self.size is not None and self.size % 16:
            raise ValueError("size must be a multiple of 16")
        return self


class JobAccepted(BaseModel):
    """The 202 body, and the 200 body when an identical image already exists."""

    id: int
    kind: JobKind
    status: Status
    reused: bool = Field(description="true when an earlier job already produced this exact image")
    profile: str
    runner: str
    model_state: str
    fingerprint: str
    parent_id: int | None = None
    pending: int = Field(description="jobs queued ahead of this one")
    status_url: str
    image_url: str
    warnings: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    id: int
    kind: JobKind
    status: Status
    profile: str
    runner: str
    fingerprint: str
    parent_id: int | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    queue_position: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    status_url: str
    image_url: str


class Health(BaseModel):
    profile: str
    runner: str
    model_state: Literal["loading", "ready", "error", "stopped"]
    model_error: str | None = None
    queue_depth: int
    current_job: int | None = None
    next_id: int
    accepting: bool
