"""Sweep planning: the product of prompts and seeds, minus what already exists.

A sweep is the unit of work that makes iteration fast. The model loads once for the
whole run, and every output is named by its fingerprint, so rerunning a sweep does
only what is left. Killing a run halfway and starting it again continues it.

Planning is deliberately separate from execution. The same plan drives a dry run and
a real run, so what you preview is exactly what you get.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imagegen import fingerprint
from imagegen.config import Catalogue, ConfigError, Kind, ResolvedProfile
from imagegen.manifest import Manifest
from imagegen.storage import output_name, sidecar_for

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def compact(text: str) -> str:
    """Flatten whitespace. A prompt pasted across lines is still one prompt."""
    return " ".join(text.split())


class SweepEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    prompt: str
    seeds: list[int] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not IDENTIFIER.fullmatch(self.id):
            raise ValueError(f"entry id {self.id!r} must be alphanumeric with dashes or underscores")
        if not compact(self.prompt):
            raise ValueError(f"entry {self.id} has an empty prompt")
        if self.seeds is not None and not self.seeds:
            raise ValueError(f"entry {self.id} has an empty seed list")
        if self.seeds and any(seed < 0 for seed in self.seeds):
            raise ValueError(f"entry {self.id} has a negative seed")
        return self


class Sweep(BaseModel):
    """A sweep file: one profile, one frame size, prompts crossed with seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    seeds: list[int] = Field(default_factory=list)
    prompts: list[SweepEntry] = Field(min_length=1)
    common_prompt: str = ""
    width: int | None = Field(default=None, ge=64)
    height: int | None = Field(default=None, ge=64)
    steps: int | None = Field(default=None, ge=1)
    guidance: float | None = Field(default=None, gt=0)
    kind: Kind = "generate"

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [entry.id for entry in self.prompts]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate entry id(s): {', '.join(duplicates)}")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must not be negative")
        without = [entry.id for entry in self.prompts if not entry.seeds and not self.seeds]
        if without:
            raise ValueError(f"no seeds for entry {without[0]}; set seeds at the top level or per entry")
        return self

    @classmethod
    def read(cls, path: Path) -> Sweep:
        if not path.is_file():
            raise ConfigError(f"sweep file not found: {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def full_prompt(self, entry: SweepEntry) -> str:
        parts = [compact(self.common_prompt), compact(entry.prompt)]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class Item:
    """One image in a plan, already named and already decided."""

    entry_id: str
    seed: int
    fingerprint: str
    manifest: Manifest
    manifest_path: Path
    done: bool

    @property
    def output(self) -> Path:
        return Path(self.manifest.output)


def plan(
    sweep: Sweep,
    profile: ResolvedProfile,
    out_dir: Path,
    runner: str,
    force: bool = False,
) -> list[Item]:
    """Expand the sweep and decide, for each image, whether there is work to do."""
    engine = profile.engine
    width = sweep.width or engine.width
    height = sweep.height or engine.height
    steps = sweep.steps or engine.default_steps(sweep.kind)
    guidance = engine.check_guidance(sweep.guidance if sweep.guidance is not None else engine.guidance)

    identity = profile.identity()
    items: list[Item] = []
    for entry in sweep.prompts:
        prompt = sweep.full_prompt(entry)
        for seed in entry.seeds or sweep.seeds:
            draft = Manifest(
                kind=sweep.kind,
                job_id=0,
                profile=profile.name,
                prompt=prompt,
                seed=seed,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                output=out_dir / "pending.png",
                metadata=out_dir / "pending.json",
            )
            stamp = fingerprint.of(draft, identity, runner)
            image = out_dir / output_name(entry.id, stamp, seed)
            manifest = draft.model_copy(update={"output": image, "metadata": sidecar_for(image)})
            manifest_path = out_dir / "manifests" / f"{image.stem}.json"
            items.append(
                Item(
                    entry_id=entry.id,
                    seed=seed,
                    fingerprint=stamp,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    done=image.exists() and not force,
                )
            )
    return items


def pending(items: list[Item]) -> list[Item]:
    return [item for item in items if not item.done]


def resolve_profile(sweep: Sweep, catalogue: Catalogue, override: str | None = None) -> ResolvedProfile:
    return catalogue.resolve(override or sweep.profile)
