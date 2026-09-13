# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import pytest
from tests.conftest import FOUR_B, NINE_B

from imagegen.config import Adapter, ConfigError, load_catalogue


def test_shipped_catalogue_defines_its_profiles(catalogue):
    assert catalogue.profile_names == ["plain-4b", FOUR_B, "subject-4b-baked", NINE_B]


def test_nine_b_keeps_adapter_order(catalogue, weights):
    """Order is load-bearing: the general adapter precedes the subject adapter."""
    references = [reference for reference, _ in catalogue.resolve(NINE_B, weights).adapters]
    assert references[0].startswith("example-org/")
    assert references[1].endswith("subj-a-9b-rank32-step1200.safetensors")


def test_nine_b_scales_are_preserved(catalogue, weights):
    assert [scale for _, scale in catalogue.resolve(NINE_B, weights).adapters] == [1.0, 1.15]


def test_encoder_pattern_differs_between_engines(catalogue):
    """The 4B encoder sits in a subdirectory, the 9B one at the repository root.

    This single difference decides which files the weight loader sees, so it is the
    one config value most able to fail silently.
    """
    four = catalogue.engine[catalogue.profile[FOUR_B].engine].encoder
    nine = catalogue.engine[catalogue.profile[NINE_B].engine].encoder
    assert four.file_pattern == "klein-4b-alt-text-encoder/*.safetensors"
    assert nine.file_pattern == "*.safetensors"


def test_locked_guidance_refuses_other_values(catalogue):
    engine = catalogue.engine["klein-4b"]
    assert engine.check_guidance(1.0) == 1.0
    with pytest.raises(ConfigError, match="distilled"):
        engine.check_guidance(3.5)


def test_default_steps_are_per_kind(catalogue):
    engine = catalogue.engine["klein-4b"]
    assert engine.default_steps("generate") == 4
    assert engine.default_steps("mask") == 8


def test_unknown_profile_names_the_known_ones(catalogue):
    with pytest.raises(ConfigError, match="unknown profile"):
        catalogue.resolve("nope")


def test_profile_naming_an_unknown_engine_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        '[engine.a]\ncheckpoint="c"\nconfig="k"\nguidance=1.0\n'
        'steps={generate=1,edit=1,mask=1}\n\n[profile.p]\nengine="missing"\n'
    )
    with pytest.raises(ConfigError, match="unknown engine"):
        load_catalogue(path)


def test_missing_weights_are_reported_not_guessed(catalogue, tmp_path):
    resolved = catalogue.resolve(FOUR_B, tmp_path / "empty")
    assert len(resolved.missing_weights()) == 1
    with pytest.raises(ConfigError, match="IMAGEGEN_WEIGHTS"):
        resolved.verify_weights()


def test_present_weights_verify_cleanly(profile):
    assert profile.missing_weights() == []
    profile.verify_weights()


def test_adapter_path_cannot_escape_the_weights_directory(weights):
    adapter = Adapter(path="../../etc/passwd", scale=1.0)
    with pytest.raises(ConfigError, match="escapes"):
        adapter.resolved_path(weights)


def test_adapter_needs_exactly_one_form():
    with pytest.raises(ValueError, match="either path, or repo"):
        Adapter(path="a.safetensors", repo="x/y", file="z")
    with pytest.raises(ValueError, match="either path, or repo"):
        Adapter(scale=1.0)
    with pytest.raises(ValueError, match="needs file alongside repo"):
        Adapter(repo="x/y")


def test_hub_adapter_uses_the_reference_form_mflux_resolves(weights):
    adapter = Adapter(repo="org/repo", file="a b.safetensors")
    assert adapter.reference(weights) == "org/repo:a b.safetensors"


def test_bake_defaults_to_off_because_mflux_defaults_it_on(catalogue):
    """Baking folds adapters into Q4 weights and re-promotes them to Q8.

    That doubles the memory bandwidth of every subsequent step, so the catalogue has
    to ask for it rather than inherit it from the library default.
    """
    assert catalogue.profile[FOUR_B].bake is False
    assert catalogue.profile["subject-4b-baked"].bake is True


def test_baking_forks_the_fingerprint(catalogue, weights):
    """Same engine, same adapter, same scale — but not the same run.

    Without this, a bake benchmark would overwrite the un-baked output it is being
    compared against, because every other input to the name is identical.
    """
    plain = catalogue.resolve(FOUR_B, weights)
    baked = catalogue.resolve("subject-4b-baked", weights)
    assert plain.adapters == baked.adapters
    assert plain.identity() != baked.identity()
