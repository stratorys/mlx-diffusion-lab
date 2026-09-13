# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imagegen.manifest import MaskSettings
from imagegen.segment import (
    BoxSegmenter,
    MaskRejected,
    build_segmenter,
    components,
    derive_mask,
    threshold_ladder,
)


def field_with(*boxes, size=(200, 160), value=0.9) -> np.ndarray:
    heat = np.zeros((size[1], size[0]), dtype=np.float32)
    for left, top, right, bottom in boxes:
        heat[top:bottom, left:right] = value
    return heat


def derive(heat, **overrides):
    return derive_mask(heat, MaskSettings(**overrides), "the dress", "fake")


def test_the_ladder_descends_from_the_requested_threshold():
    assert threshold_ladder(0.4) == [0.4, 0.3, 0.2]


def test_separate_blobs_are_counted_separately():
    binary = field_with((10, 10, 40, 40), (120, 100, 160, 140)) > 0.5
    assert len(components(binary)) == 2


def test_a_touching_blob_is_one_region():
    binary = field_with((10, 10, 40, 40), (40, 10, 70, 40)) > 0.5
    assert len(components(binary)) == 1


def test_a_clear_region_yields_a_mask():
    mask, report = derive(field_with((50, 40, 120, 110)), dilate=0, feather=0)
    assert report.components == 1 and report.warnings == []
    assert report.threshold_used == 0.4
    assert mask.getbbox() == (50, 40, 120, 110)


def test_a_low_confidence_heatmap_is_refused_with_advice():
    with pytest.raises(MaskRejected) as raised:
        derive(field_with((50, 40, 120, 110), value=0.1))
    assert raised.value.code == "mask_low_confidence"
    assert "target phrase" in str(raised.value)


def test_a_region_below_the_first_threshold_relaxes_rather_than_failing():
    mask, report = derive(field_with((50, 40, 120, 110), value=0.32), threshold=0.4)
    assert report.threshold_used == 0.3
    assert any("relaxed" in note for note in report.warnings)
    assert mask.getbbox() is not None


def test_nothing_above_the_whole_ladder_is_an_empty_mask():
    with pytest.raises(MaskRejected) as raised:
        derive(field_with((50, 40, 120, 110), value=0.26), threshold=0.9, min_confidence=0.2)
    assert raised.value.code == "mask_empty"


def test_a_region_covering_the_frame_is_refused():
    with pytest.raises(MaskRejected) as raised:
        derive(field_with((0, 0, 200, 160)), max_coverage=0.85)
    assert raised.value.code == "mask_too_large"
    assert "generate, not a mask" in str(raised.value)


def test_two_comparable_regions_warn_and_keep_the_larger():
    heat = field_with((10, 10, 60, 60), (120, 90, 175, 150))
    mask, report = derive(heat)
    assert report.components == 2
    assert any("separate regions" in note for note in report.warnings)
    # Only the larger blob survives, so the other corner is untouched.
    values = np.asarray(mask)
    assert values[120, 150] > 0
    assert values[30, 30] == 0


def test_dilation_grows_the_region(tmp_path):
    tight, _ = derive(field_with((80, 60, 100, 80)), dilate=0, feather=0)
    grown, _ = derive(field_with((80, 60, 100, 80)), dilate=6, feather=0)
    assert np.asarray(grown).sum() > np.asarray(tight).sum()


def test_feathering_softens_the_edge():
    hard, _ = derive(field_with((80, 60, 120, 100)), dilate=0, feather=0)
    soft, _ = derive(field_with((80, 60, 120, 100)), dilate=0, feather=5)
    assert set(np.unique(np.asarray(hard))) <= {0, 255}
    assert len(np.unique(np.asarray(soft))) > 2


def test_the_report_survives_serialisation():
    _, report = derive(field_with((50, 40, 120, 110)))
    payload = report.as_dict()
    assert payload["target"] == "the dress" and payload["segmenter"] == "fake"
    assert isinstance(payload["bbox"], list)


def test_the_box_segmenter_is_deterministic():
    image = Image.new("RGB", (200, 160))
    first = BoxSegmenter().heatmap(image, "anything")
    assert np.array_equal(first, BoxSegmenter().heatmap(image, "something else"))


def test_an_unknown_segmenter_is_named_in_the_error():
    with pytest.raises(ValueError, match="unknown segmenter"):
        build_segmenter("nonsense")


def test_the_default_segmenter_is_clipseg_and_loads_nothing_up_front(monkeypatch):
    """Building must not import torch, so a server that never masks never pays."""
    monkeypatch.delenv("IMAGEGEN_SEGMENTER", raising=False)
    assert build_segmenter().name == "clipseg"
