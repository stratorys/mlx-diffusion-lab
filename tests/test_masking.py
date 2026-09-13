# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from tests.conftest import FOUR_B

from imagegen import geometry as geo
from imagegen.manifest import Manifest, MaskSettings
from imagegen.masking import diagnostics_dir, run_masked_edit
from imagegen.runners.stub import StubRunner
from imagegen.segment import BoxSegmenter, MaskRejected

SIZE = 256


class FakeSegmenter:
    """A segmenter whose heatmap the test dictates."""

    name = "fake"

    def __init__(self, heat):
        self.heat = heat

    def heatmap(self, image, target):
        return self.heat


@pytest.fixture
def parent(tmp_path):
    path = tmp_path / "parent.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8), "RGB").save(path)
    return path


def mask_manifest(tmp_path, parent, **settings) -> Manifest:
    return Manifest(
        kind="mask",
        job_id=1,
        profile=FOUR_B,
        prompt="a blue silk dress",
        seed=5,
        steps=8,
        guidance=1.0,
        width=SIZE,
        height=SIZE,
        input_image=parent,
        mask=MaskSettings(target="the dress", **settings),
        output=tmp_path / "result.png",
        metadata=tmp_path / "result.metadata.json",
    )


def test_a_masked_edit_changes_the_region_and_nothing_else(tmp_path, profile, parent):
    manifest = mask_manifest(tmp_path, parent)
    result, metrics = run_masked_edit(StubRunner(profile), manifest, BoxSegmenter())

    assert result.size == (SIZE, SIZE)
    assert metrics["outside_mask_max_delta"] == 0
    assert metrics["masked_pixels_changed_over_10_percent"] > 5.0
    assert metrics["mask"]["target"] == "the dress"


def test_every_diagnostic_is_written(tmp_path, profile, parent):
    manifest = mask_manifest(tmp_path, parent)
    run_masked_edit(StubRunner(profile), manifest, BoxSegmenter())
    produced = {path.name for path in diagnostics_dir(manifest).iterdir()}
    assert produced == {
        "heatmap.png", "mask.png", "condition.png", "source_crop.png",
        "generated_crop.png", "composite_mask.png",
    }


def test_the_crop_is_model_sized(tmp_path, profile, parent):
    manifest = mask_manifest(tmp_path, parent, padding=40)
    _, metrics = run_masked_edit(StubRunner(profile), manifest, BoxSegmenter())
    width, height = metrics["work_size"]
    assert width % geo.MULTIPLE == 0 and height % geo.MULTIPLE == 0


def test_padding_widens_the_crop_without_widening_the_edit(tmp_path, profile, parent):
    tight = mask_manifest(tmp_path, parent, padding=0)
    loose = mask_manifest(tmp_path, parent, padding=60)
    _, narrow = run_masked_edit(StubRunner(profile), tight, BoxSegmenter())
    _, wide = run_masked_edit(StubRunner(profile), loose, BoxSegmenter())

    assert wide["crop_box"][0] < narrow["crop_box"][0]
    # A wider crop gives the model more context, but the region it may alter is the same.
    assert wide["outside_mask_max_delta"] == 0


def test_a_rejected_mask_still_leaves_its_heatmap_to_look_at(tmp_path, profile, parent):
    """A failure you cannot diagnose is worse than no failure message."""
    flat = np.full((SIZE, SIZE), 0.05, dtype=np.float32)
    manifest = mask_manifest(tmp_path, parent)
    with pytest.raises(MaskRejected) as raised:
        run_masked_edit(StubRunner(profile), manifest, FakeSegmenter(flat))
    assert raised.value.code == "mask_low_confidence"
    assert (diagnostics_dir(manifest) / "heatmap.png").is_file()


def test_a_full_frame_match_is_refused(tmp_path, profile, parent):
    everything = np.full((SIZE, SIZE), 0.95, dtype=np.float32)
    manifest = mask_manifest(tmp_path, parent)
    with pytest.raises(MaskRejected) as raised:
        run_masked_edit(StubRunner(profile), manifest, FakeSegmenter(everything))
    assert raised.value.code == "mask_too_large"


def test_an_edit_that_does_nothing_is_flagged(tmp_path, profile):
    """Silence is the worst outcome: the job succeeds and the image is unchanged.

    A smooth source is used on purpose: the crop is scaled up to the model size and
    back, and that round trip would itself register as change on random noise.
    """
    parent = tmp_path / "smooth.png"
    Image.new("RGB", (SIZE, SIZE), (70, 110, 150)).save(parent)

    class Echo(StubRunner):
        """Hands back exactly the crop it was given, as a model that ignored the prompt would."""

        def generate(self, *, width, height, image_path, **rest):
            untouched = image_path.parent / "source_crop.png"
            return Image.open(untouched).convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )

    heat = np.zeros((SIZE, SIZE), dtype=np.float32)
    heat[80:150, 80:150] = 0.9
    manifest = mask_manifest(tmp_path, parent, padding=0, dilate=0, feather=0)
    _, metrics = run_masked_edit(Echo(profile), manifest, FakeSegmenter(heat))
    assert any("barely changed" in note for note in metrics["mask"]["warnings"])


def test_a_runner_that_paints_outside_the_mask_is_caught(tmp_path, profile, parent, monkeypatch):
    """The outside guarantee is asserted, not assumed."""
    manifest = mask_manifest(tmp_path, parent)
    monkeypatch.setattr(
        geo, "difference_metrics", lambda *a, **k: {
            "outside_mask_max_delta": 12,
            "masked_pixels_changed_over_10_percent": 90.0,
            "masked_mean_absolute_error": 40.0,
            "mask_area_fraction": 0.1,
        }
    )
    with pytest.raises(RuntimeError, match="outside the mask"):
        run_masked_edit(StubRunner(profile), manifest, BoxSegmenter())


def test_the_target_falls_back_to_the_prompt(tmp_path, profile, parent):
    manifest = mask_manifest(tmp_path, parent)
    without = manifest.model_copy(update={"mask": MaskSettings()})
    _, metrics = run_masked_edit(StubRunner(profile), without, BoxSegmenter())
    assert metrics["mask"]["target"] == "a blue silk dress"
