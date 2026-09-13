"""Engine and profile catalogue.

An engine describes a checkpoint, its text encoder and its limits. It knows nothing
about any subject. A profile binds an engine to an ordered stack of LoRA adapters.

Local adapter files resolve against the weights directory, named by the
IMAGEGEN_WEIGHTS environment variable. Nothing in this package holds an absolute
path or a subject name; both live in the TOML catalogue.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Kind = Literal["generate", "edit", "mask"]
KINDS: tuple[Kind, ...] = ("generate", "edit", "mask")

WEIGHTS_ENV = "IMAGEGEN_WEIGHTS"
CATALOGUE_ENV = "IMAGEGEN_PROFILES"
DEFAULT_CATALOGUE = Path("config/profiles.toml")


class ConfigError(ValueError):
    """The catalogue is malformed, or names something that does not exist."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EncoderSpec(Strict):
    """A replacement text encoder, requantised at load.

    `subdir` is empty when the weights sit at the repository root. It drives the
    weight-loader file pattern, so an empty value is meaningful rather than absent.
    """

    repo: str = Field(min_length=1)
    subdir: str = ""
    quantize: int = Field(default=4, ge=1, le=16)

    @property
    def file_pattern(self) -> str:
        return f"{self.subdir}/*.safetensors" if self.subdir else "*.safetensors"


class Engine(Strict):
    checkpoint: str = Field(min_length=1)
    config: str = Field(min_length=1)
    guidance: float = Field(gt=0)
    guidance_locked: bool = True
    steps: dict[str, int]
    width: int = Field(default=1024, ge=64)
    height: int = Field(default=1024, ge=64)
    encoder: EncoderSpec | None = None

    @model_validator(mode="after")
    def _check_steps(self) -> Self:
        missing = [kind for kind in KINDS if kind not in self.steps]
        if missing:
            raise ValueError(f"steps is missing {', '.join(missing)}")
        bad = [kind for kind, value in self.steps.items() if value < 1]
        if bad:
            raise ValueError(f"steps must be at least 1 for {', '.join(sorted(bad))}")
        return self

    def default_steps(self, kind: Kind) -> int:
        return self.steps[kind]

    def check_guidance(self, value: float) -> float:
        """Return the guidance to use, or raise when the engine forbids it."""
        if self.guidance_locked and value != self.guidance:
            raise ConfigError(
                f"this engine is distilled and only accepts guidance {self.guidance:g}, got {value:g}"
            )
        if value <= 0:
            raise ConfigError("guidance must be positive")
        return float(value)


class Adapter(Strict):
    """One LoRA, either a local file or a hub reference."""

    path: str | None = None
    repo: str | None = None
    file: str | None = None
    scale: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Self:
        local = self.path is not None
        hosted = self.repo is not None
        if local == hosted:
            raise ValueError("an adapter needs either path, or repo with file, but not both")
        if hosted and not self.file:
            raise ValueError("a hub adapter needs file alongside repo")
        if local and not self.path:
            raise ValueError("path must not be empty")
        return self

    @property
    def is_local(self) -> bool:
        return self.path is not None

    def reference(self, weights_dir: Path) -> str:
        """The string mflux expects.

        An absolute path for a local file, or a `repo:file` pair, which mflux resolves
        natively in LoraResolution. A hosted adapter therefore needs no local copy.
        """
        if self.path is not None:
            return str(self.resolved_path(weights_dir))
        return f"{self.repo}:{self.file}"

    def resolved_path(self, weights_dir: Path) -> Path:
        """A local adapter path, refused if it escapes the weights directory."""
        if self.path is None:
            raise ConfigError("this adapter is a hub reference, not a local file")
        root = weights_dir.resolve()
        candidate = (root / self.path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ConfigError(f"adapter path escapes the weights directory: {self.path}")
        return candidate

    def identity(self) -> str:
        """A stable key for fingerprinting.

        Deliberately the file name rather than the absolute path. Moving the weights
        directory must not invalidate every output ever produced from it.
        """
        if self.path is not None:
            return Path(self.path).name
        return f"{self.repo}:{self.file}"


class Profile(Strict):
    engine: str = Field(min_length=1)
    adapters: list[Adapter] = Field(default_factory=list)
    # Baking folds the adapters into the weights once, instead of applying them on
    # every forward pass. On a sub-8-bit checkpoint the fold cannot stay at the
    # original precision, so the touched layers are re-promoted; whether that costs or
    # saves time depends on how many layers the adapter touches, so it is a value to
    # measure rather than assume (bench/RESULTS.md).
    # Defaults to False against mflux's own default of True, because the applied path
    # has a cost independent of the adapter's reach and leaves scales adjustable
    # without a reload — not on a speed claim.
    bake: bool = False


class ResolvedProfile:
    """A profile with its engine attached and its adapter references materialised."""

    def __init__(self, name: str, profile: Profile, engine: Engine, weights_dir: Path):
        self.name = name
        self.engine_name = profile.engine
        self.engine = engine
        self.weights_dir = weights_dir
        self._adapters = profile.adapters
        self.bake = profile.bake

    @property
    def adapters(self) -> list[tuple[str, float]]:
        """Ordered adapter references with their scales. Order is significant."""
        return [(a.reference(self.weights_dir), a.scale) for a in self._adapters]

    def missing_weights(self) -> list[Path]:
        """Local adapter files the catalogue names but the disk does not have."""
        absent = []
        for adapter in self._adapters:
            if adapter.is_local:
                candidate = adapter.resolved_path(self.weights_dir)
                if not candidate.is_file():
                    absent.append(candidate)
        return absent

    def identity(self) -> dict:
        """What the fingerprint hashes: the pixel-affecting identity of this stack.

        Adapter file names rather than absolute paths, so relocating the weights
        directory does not fork every output name.
        """
        return {
            "checkpoint": self.engine.checkpoint,
            "model_config": self.engine.config,
            "encoder": None
            if self.engine.encoder is None
            else [self.engine.encoder.repo, self.engine.encoder.subdir, self.engine.encoder.quantize],
            "adapters": [[a.identity(), round(a.scale, 6)] for a in self._adapters],
            # A baked stack and an applied stack are not the same run. Sharing a
            # fingerprint would let a benchmark silently overwrite production output.
            "bake": self.bake,
        }

    def verify_weights(self) -> None:
        absent = self.missing_weights()
        if absent:
            listed = ", ".join(str(path) for path in absent)
            raise ConfigError(f"adapter file not found: {listed}. Is {WEIGHTS_ENV} set correctly?")

    def provenance(self) -> dict:
        """What actually produced an image, for the metadata sidecar.

        The profile name is not enough. A catalogue edit six months from now must not
        silently change what a past image claims to be.
        """
        encoder = self.engine.encoder
        return {
            "profile": self.name,
            "engine": self.engine_name,
            "checkpoint": self.engine.checkpoint,
            "model_config": self.engine.config,
            "encoder": None
            if encoder is None
            else {"repo": encoder.repo, "subdir": encoder.subdir, "quantize": encoder.quantize},
            "adapters": [{"reference": ref, "scale": scale} for ref, scale in self.adapters],
            "bake": self.bake,
        }


class Catalogue(Strict):
    engine: dict[str, Engine]
    profile: dict[str, Profile]

    @model_validator(mode="after")
    def _engines_exist(self) -> Self:
        for name, profile in self.profile.items():
            if profile.engine not in self.engine:
                known = ", ".join(sorted(self.engine)) or "none"
                raise ValueError(f"profile {name} names unknown engine {profile.engine}; known: {known}")
        return self

    @property
    def profile_names(self) -> list[str]:
        return sorted(self.profile)

    def resolve(self, name: str, weights_dir: Path | None = None) -> ResolvedProfile:
        if name not in self.profile:
            known = ", ".join(self.profile_names) or "none"
            raise ConfigError(f"unknown profile {name}; known: {known}")
        profile = self.profile[name]
        return ResolvedProfile(name, profile, self.engine[profile.engine], weights_dir or weights_directory())


def weights_directory() -> Path:
    """Where local adapter files live. Defaults to a weights folder beside the project."""
    return Path(os.environ.get(WEIGHTS_ENV, "weights")).expanduser()


def catalogue_path() -> Path:
    return Path(os.environ.get(CATALOGUE_ENV, DEFAULT_CATALOGUE)).expanduser()


def load_catalogue(path: Path | None = None) -> Catalogue:
    source = path or catalogue_path()
    if not source.is_file():
        raise ConfigError(f"profile catalogue not found: {source}")
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{source} is not valid TOML: {error}") from error
    try:
        return Catalogue.model_validate(raw)
    except ValueError as error:
        raise ConfigError(f"{source} is not a valid catalogue: {error}") from error
