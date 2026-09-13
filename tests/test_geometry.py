# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imagegen import geometry as geo


def blob(size=(200, 160), box=(50, 40, 120, 110)) -> Image.Image:
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    return mask


def noise_image(size=(200, 160), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def test_padding_grows_the_box_and_stops_at_every_border():
    mask = blob()
    assert geo.crop_region(mask, 10, (200, 160)) == (40, 30, 130, 120)
    # A padding larger than the image cannot escape it.
    assert geo.crop_region(mask, 10_000, (200, 160)) == (0, 0, 200, 160)


def test_an_empty_mask_is_an_error_not_a_full_frame():
    empty = Image.new("L", (64, 64), 0)
    with pytest.raises(geo.EmptyMaskError):
        geo.crop_region(empty, 8, (64, 64))
    with pytest.raises(geo.EmptyMaskError):
        geo.prepare_mask(empty, (64, 64))


def test_work_size_is_always_model_sized():
    for crop in [(313, 207), (1000, 17), (64, 64), (5, 900)]:
        width, height = geo.work_size(crop, 768)
        assert width % geo.MULTIPLE == 0 and height % geo.MULTIPLE == 0
        assert width >= geo.MULTIPLE and height >= geo.MULTIPLE


def test_work_size_keeps_the_aspect_ratio_close():
    width, height = geo.work_size((300, 200), 768)
    assert abs((width / height) - 1.5) < 0.06


def test_automatic_target_size_stays_within_its_bounds():
    assert geo.automatic_target_size((100, 80)) == geo.MIN_TARGET
    assert geo.automatic_target_size((4000, 10)) == geo.MAX_TARGET
    assert geo.automatic_target_size((800, 600)) == 800


def test_erasing_is_deterministic_for_a_seed():
    source, mask = noise_image(), blob()
    first = np.asarray(geo.erased_condition(source, mask, 42))
    assert np.array_equal(first, np.asarray(geo.erased_condition(source, mask, 42)))
    assert not np.array_equal(first, np.asarray(geo.erased_condition(source, mask, 43)))


def test_erasing_leaves_unmasked_pixels_exactly_alone():
    source, mask = noise_image(), blob()
    erased = geo.erased_condition(source, mask, 7)
    keep = np.asarray(mask) == 0
    before = np.asarray(source)[keep]
    after = np.asarray(erased)[keep]
    assert np.array_equal(before, after)


def test_a_crop_touching_every_border_is_not_feathered():
    solid = geo.crop_blend_mask((0, 0, 50, 50), (50, 50), feather=8)
    assert np.asarray(solid).min() == 255


def test_an_interior_crop_edge_ramps_from_zero():
    blended = geo.crop_blend_mask((10, 10, 60, 60), (200, 200), feather=8)
    values = np.asarray(blended)
    assert values[0, 0] == 0
    assert values[25, 25] == 255


def test_the_composite_leaves_the_outside_bit_identical():
    """The one invariant that matters: zero-mask pixels never move."""
    source = noise_image()
    mask = blob()
    box = geo.crop_region(mask, 12, source.size)
    generated = noise_image(seed=99).crop(box)

    result = geo.composite_local(source, generated, mask.crop(box), box)
    effective = geo.full_image_mask(mask.crop(box), box, source.size)
    metrics = geo.difference_metrics(source, result, effective)
    assert metrics["outside_mask_max_delta"] == 0
    assert metrics["masked_pixels_changed_over_10_percent"] > 50


@pytest.mark.parametrize("seed", range(6))
def test_the_outside_invariant_holds_for_arbitrary_masks_and_crops(seed):
    rng = np.random.default_rng(seed)
    size = (int(rng.integers(80, 240)), int(rng.integers(80, 240)))
    left = int(rng.integers(0, size[0] - 20))
    top = int(rng.integers(0, size[1] - 20))
    right = int(rng.integers(left + 10, size[0]))
    bottom = int(rng.integers(top + 10, size[1]))

    source = noise_image(size, seed)
    mask = Image.new("L", size, 0)
    mask.paste(255, (left, top, right, bottom))
    box = geo.crop_region(mask, int(rng.integers(0, 40)), size)
    generated = noise_image(size, seed + 100).crop(box)

    result = geo.composite_local(source, generated, mask.crop(box), box)
    effective = geo.full_image_mask(mask.crop(box), box, size)
    assert geo.difference_metrics(source, result, effective)["outside_mask_max_delta"] == 0


def test_an_identical_result_reports_no_change():
    source = noise_image()
    mask = blob()
    metrics = geo.difference_metrics(source, source, mask)
    assert metrics["masked_mean_absolute_error"] == 0.0
    assert metrics["outside_mask_max_delta"] == 0


def test_a_mask_is_resized_to_the_image(tmp_path):
    small = Image.new("L", (50, 40), 0)
    small.paste(255, (10, 10, 30, 30))
    assert geo.prepare_mask(small, (200, 160)).size == (200, 160)
