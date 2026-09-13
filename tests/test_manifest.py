# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import pytest
from PIL import Image
from tests.conftest import make_manifest

from imagegen.manifest import Manifest, MaskSettings


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "parent.png"
    Image.new("RGB", (128, 128), (30, 40, 50)).save(path)
    return path


def test_generate_refuses_an_input_image(tmp_path, source):
    with pytest.raises(ValueError, match="takes no input image"):
        make_manifest(tmp_path, input_image=source)


def test_generate_refuses_mask_settings(tmp_path):
    with pytest.raises(ValueError, match="takes no mask settings"):
        make_manifest(tmp_path, mask=MaskSettings())


def test_edit_needs_an_input_image(tmp_path):
    with pytest.raises(ValueError, match="needs an input image"):
        make_manifest(tmp_path, kind="edit")


def test_edit_refuses_mask_settings(tmp_path, source):
    with pytest.raises(ValueError, match="takes no mask settings"):
        make_manifest(tmp_path, kind="edit", input_image=source, mask=MaskSettings())


def test_mask_needs_mask_settings(tmp_path, source):
    with pytest.raises(ValueError, match="needs mask settings"):
        make_manifest(tmp_path, kind="mask", input_image=source)


def test_dimensions_must_be_model_sized(tmp_path):
    with pytest.raises(ValueError, match="multiples of 16"):
        make_manifest(tmp_path, width=770)


def test_unknown_fields_are_refused(tmp_path):
    with pytest.raises(ValueError):
        make_manifest(tmp_path, sampler="euler")


def test_segmentation_target_falls_back_to_the_prompt(tmp_path, source):
    manifest = make_manifest(tmp_path, kind="mask", input_image=source, mask=MaskSettings())
    assert manifest.segmentation_target == manifest.prompt

    targeted = manifest.model_copy(update={"mask": MaskSettings(target="the dress")})
    assert targeted.segmentation_target == "the dress"


def test_a_manifest_survives_a_round_trip(tmp_path, source):
    original = make_manifest(
        tmp_path, kind="mask", input_image=source, mask=MaskSettings(threshold=0.5, dilate=4)
    )
    path = tmp_path / "job.json"
    original.write(path)
    assert Manifest.read(path) == original


def test_mask_size_must_be_a_model_dimension():
    with pytest.raises(ValueError, match="multiple of 16"):
        MaskSettings(size=700)
