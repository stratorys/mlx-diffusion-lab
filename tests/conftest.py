# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from pathlib import Path

import pytest

from imagegen.config import load_catalogue
from imagegen.manifest import Manifest
from imagegen.storage import output_name, sidecar_for

PROJECT = Path(__file__).resolve().parent.parent
CATALOGUE = PROJECT / "config" / "profiles.toml"

# The names the shipped catalogue defines. Tests assert against the real file rather
# than a fixture copy, so a careless catalogue edit fails the suite.
FOUR_B = "subject-4b"
NINE_B = "subject-9b"


@pytest.fixture
def catalogue():
    return load_catalogue(CATALOGUE)


@pytest.fixture
def weights(tmp_path):
    """A weights directory holding every local adapter the catalogue names."""
    directory = tmp_path / "weights"
    directory.mkdir()
    for name in (
        "subj-a-4b-rank64-step1200.safetensors",
        "subj-a-9b-rank32-step1200.safetensors",
    ):
        (directory / name).write_bytes(b"not real weights")
    return directory


@pytest.fixture
def profile(catalogue, weights):
    return catalogue.resolve(FOUR_B, weights)


def make_manifest(tmp_path, **overrides) -> Manifest:
    """A valid generate manifest, with its output named the way the system names it."""
    fields = {
        "kind": "generate",
        "job_id": 1,
        "profile": FOUR_B,
        "prompt": "a lighthouse at dusk",
        "seed": 7,
        "steps": 4,
        "guidance": 1.0,
        "width": 256,
        "height": 256,
        "output": tmp_path / "out.png",
        "metadata": tmp_path / "out.metadata.json",
    }
    fields.update(overrides)
    return Manifest(**fields)


def place_output(tmp_path, fingerprint: str, seed: int, stem: str = "000001"):
    image = tmp_path / output_name(stem, fingerprint, seed)
    return image, sidecar_for(image)
