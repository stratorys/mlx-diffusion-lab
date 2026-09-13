"""The real runner: FLUX.2 Klein through mflux on MLX.

Text generation and image conditioning are the same weights behind two wrapper
classes, so a profile serves all three endpoints. The adapters are baked in at
construction, which is exactly why a profile is a boot flag and not a request field.

Nothing in this module is imported unless IMAGEGEN_RUNNER selects it, so neither MLX
nor mflux is touched by the API, the sweep planner or the test suite.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import dataclasses
import gc
import os
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from imagegen.config import EncoderSpec, ResolvedProfile
from imagegen.runners.base import BaseRunner

RESIDENT_ENV = "IMAGEGEN_RESIDENT"
TEXT, EDIT = "text", "edit"


def release_mlx() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001, S110 - cleanup must never be the thing that fails a job
        pass


def swap_text_encoder(model, model_config, encoder: EncoderSpec) -> None:
    """Replace the stock text encoder, requantised in place.

    The subdirectory is the whole point. One engine keeps its encoder in a folder and
    the other at the repository root, and that decides which files the loader sees.
    Getting it wrong loads nothing and raises nothing, so it is passed explicitly.
    """
    import mlx.core as mx
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    component = next(
        item for item in Flux2KleinWeightDefinition.get_components() if item.name == "text_encoder"
    )
    component = dataclasses.replace(component, hf_subdir=encoder.subdir or "")

    model.text_encoder = None
    release_mlx()

    weights = WeightLoader.load_single(
        component=component,
        repo_id=encoder.repo,
        file_pattern=encoder.file_pattern,
    )
    replacement = Qwen3TextEncoder(**model_config.text_encoder_overrides)
    WeightApplier.apply_and_quantize_single(
        weights=weights,
        model=replacement,
        component=component,
        quantize_arg=encoder.quantize,
        quantization_predicate=Flux2KleinWeightDefinition.quantization_predicate,
    )
    mx.eval(replacement.parameters())
    model.text_encoder = replacement
    release_mlx()


def build_variant(profile: ResolvedProfile, variant: str):
    """Construct one wrapper around the profile's checkpoint and adapter stack."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

    engine = profile.engine
    model_config = getattr(ModelConfig, engine.config)()
    adapters = profile.adapters
    cls = Flux2KleinEdit if variant == EDIT else Flux2Klein
    model = cls(
        model_config=model_config,
        model_path=engine.checkpoint,
        quantize=None,
        lora_paths=[reference for reference, _ in adapters],
        lora_scales=[scale for _, scale in adapters],
        # mflux defaults this to True. See Profile.bake for why we do not.
        bake_lora=profile.bake,
    )
    if engine.encoder is not None:
        swap_text_encoder(model, model_config, engine.encoder)
    return model


class MfluxRunner(BaseRunner):
    name: ClassVar[str] = "mflux"

    def __init__(self, profile: ResolvedProfile, segmenter_name: str | None = None):
        super().__init__(profile, segmenter_name)
        self._models: dict[str, Any] = {}
        # Both wrappers cover the same weights, so holding both avoids a full reload
        # whenever a generate is followed by an edit. Set IMAGEGEN_RESIDENT=one on a
        # machine where the pair does not fit.
        self.resident = os.environ.get(RESIDENT_ENV, "both").strip().lower()

    def load(self) -> None:
        """Pay for the text variant up front. The edit variant loads on first use."""
        self.profile.verify_weights()
        self._variant(TEXT)

    def close(self) -> None:
        self._models.clear()
        release_mlx()

    def _variant(self, variant: str):
        if variant not in self._models:
            if self.resident == "one" and self._models:
                self._models.clear()
                release_mlx()
            self._models[variant] = build_variant(self.profile, variant)
        return self._models[variant]

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
        extra: dict[str, Any] = {}
        if image_path is not None:
            model = self._variant(EDIT)
            extra["image_paths"] = [str(image_path)]
            if image_strength is not None:
                extra["image_strength"] = image_strength
        else:
            model = self._variant(TEXT)

        produced = model.generate_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            **extra,
        )
        # mflux hands back a wrapper; the base runner wants the PIL image so that the
        # atomic save and the sidecar behave identically for every runner.
        return produced.image
