"""The real runner's boundary, tested without mflux, MLX or any weights.

What can be checked here is that nothing heavy is imported until it is needed, that a
missing adapter fails before any model work begins, and that the encoder settings the
runner would hand to the loader are the ones the catalogue declares.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import sys

import pytest

from imagegen.config import ConfigError
from imagegen.runners import build_runner
from imagegen.runners.mflux import EDIT, TEXT, MfluxRunner


def test_building_the_runner_imports_no_model_library(catalogue, weights, monkeypatch):
    monkeypatch.delitem(sys.modules, "mflux", raising=False)
    monkeypatch.delitem(sys.modules, "mlx", raising=False)
    runner = build_runner(catalogue.resolve("subject-4b", weights), "mflux")
    assert isinstance(runner, MfluxRunner) and runner.name == "mflux"
    assert "mflux" not in sys.modules and "mlx" not in sys.modules


def test_missing_adapters_fail_before_any_model_work(catalogue, tmp_path):
    runner = MfluxRunner(catalogue.resolve("subject-4b", tmp_path / "nothing-here"))
    with pytest.raises(ConfigError, match="adapter file not found"):
        runner.load()


def test_both_wrappers_are_held_by_default(catalogue, weights, monkeypatch):
    monkeypatch.delenv("IMAGEGEN_RESIDENT", raising=False)
    assert MfluxRunner(catalogue.resolve("subject-4b", weights)).resident == "both"


def test_the_resident_policy_is_configurable(catalogue, weights, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_RESIDENT", "one")
    assert MfluxRunner(catalogue.resolve("subject-4b", weights)).resident == "one"


def test_holding_one_wrapper_evicts_the_other(catalogue, weights, monkeypatch):
    """Switching wrappers under the one policy must not leave both in memory."""
    monkeypatch.setenv("IMAGEGEN_RESIDENT", "one")
    runner = MfluxRunner(catalogue.resolve("subject-4b", weights))
    monkeypatch.setattr(
        "imagegen.runners.mflux.build_variant", lambda profile, variant: f"model-{variant}"
    )
    runner._variant(TEXT)
    runner._variant(EDIT)
    assert set(runner._models) == {EDIT}


def test_holding_both_keeps_them(catalogue, weights, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_RESIDENT", "both")
    runner = MfluxRunner(catalogue.resolve("subject-4b", weights))
    monkeypatch.setattr(
        "imagegen.runners.mflux.build_variant", lambda profile, variant: f"model-{variant}"
    )
    runner._variant(TEXT)
    runner._variant(EDIT)
    assert set(runner._models) == {TEXT, EDIT}


def test_a_conditioning_image_selects_the_edit_wrapper(catalogue, weights, monkeypatch, tmp_path):
    """Text and edit are the same weights behind two wrappers; the input decides which."""
    calls = []

    class FakeModel:
        def __init__(self, variant):
            self.variant = variant

        def generate_image(self, **kwargs):
            calls.append((self.variant, kwargs))

            class Produced:
                image = "an image"

            return Produced()

    runner = MfluxRunner(catalogue.resolve("subject-4b", weights))
    monkeypatch.setattr("imagegen.runners.mflux.build_variant", lambda profile, variant: FakeModel(variant))

    common = {"prompt": "x", "seed": 1, "steps": 4, "guidance": 1.0, "width": 256, "height": 256}
    runner.generate(**common)
    runner.generate(**common, image_path=tmp_path / "parent.png", image_strength=0.6)

    assert calls[0][0] == TEXT and "image_paths" not in calls[0][1]
    assert calls[1][0] == EDIT
    assert calls[1][1]["image_paths"] == [str(tmp_path / "parent.png")]
    assert calls[1][1]["image_strength"] == 0.6


def test_the_encoder_settings_reach_the_loader_unchanged(catalogue):
    """The 4b subdirectory and the 9b root are the one config path with no offline proof."""
    four = catalogue.engine["klein-4b"].encoder
    nine = catalogue.engine["klein-9b"].encoder
    assert four.file_pattern.startswith("klein-4b-alt-text-encoder/")
    assert nine.file_pattern == "*.safetensors"
    assert four.quantize == nine.quantize == 4
