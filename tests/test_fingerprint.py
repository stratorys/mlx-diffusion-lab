# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from tests.conftest import make_manifest

from imagegen import fingerprint


def fp(manifest, profile, runner="stub"):
    return fingerprint.of(manifest, profile.identity(), runner)


def test_same_inputs_give_the_same_name(tmp_path, profile):
    manifest = make_manifest(tmp_path)
    assert fp(manifest, profile) == fp(manifest, profile)
    assert len(fp(manifest, profile)) == fingerprint.LENGTH


def test_every_pixel_affecting_field_forks_the_name(tmp_path, profile):
    base = make_manifest(tmp_path)
    original = fp(base, profile)
    for change in ({"seed": 8}, {"steps": 6}, {"guidance": 2.0}, {"width": 512}, {"prompt": "other"}):
        assert fp(base.model_copy(update=change), profile) != original, change


def test_output_path_does_not_affect_the_name(tmp_path, profile):
    """The fingerprint decides the filename, so it cannot depend on the filename."""
    base = make_manifest(tmp_path)
    moved = base.model_copy(update={"output": tmp_path / "elsewhere.png", "job_id": 99})
    assert fp(moved, profile) == fp(base, profile)


def test_runner_is_part_of_the_name(tmp_path, profile):
    """A stub image must never occupy the filename a real render would claim.

    Otherwise a sweep run in stub mode would make every later real run think the work
    was already done.
    """
    manifest = make_manifest(tmp_path)
    assert fp(manifest, profile, "stub") != fp(manifest, profile, "mflux")


def test_adapter_scale_and_order_are_part_of_the_name(tmp_path, catalogue, weights):
    manifest = make_manifest(tmp_path)
    four = catalogue.resolve("subject-4b", weights)
    nine = catalogue.resolve("subject-9b", weights)
    assert fp(manifest, four) != fp(manifest, nine)

    stack = catalogue.profile["subject-9b"].adapters
    rescaled = nine.identity()
    rescaled["adapters"][1][1] = 9.0
    assert fingerprint.digest(rescaled) != fingerprint.digest(nine.identity())

    reordered = nine.identity()
    reordered["adapters"].reverse()
    assert fingerprint.digest(reordered) != fingerprint.digest(nine.identity())
    assert len(stack) == 2


def test_moving_the_weights_directory_does_not_fork_every_name(tmp_path, catalogue, weights):
    """Relocating weights must not invalidate an entire output tree."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    for source in weights.iterdir():
        (elsewhere / source.name).write_bytes(source.read_bytes())
    manifest = make_manifest(tmp_path)
    assert fp(manifest, catalogue.resolve("subject-4b", weights)) == fp(
        manifest, catalogue.resolve("subject-4b", elsewhere)
    )


def test_input_image_is_hashed_by_content_not_by_path(tmp_path, profile):
    from PIL import Image

    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(first)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(second)

    edit = make_manifest(tmp_path, kind="edit", input_image=first)
    same_bytes = edit.model_copy(update={"input_image": second})
    assert fp(same_bytes, profile) == fp(edit, profile)

    Image.new("RGB", (64, 64), (200, 0, 0)).save(second)
    assert fp(same_bytes, profile) != fp(edit, profile)


def test_mask_settings_are_part_of_the_name(tmp_path, profile):
    from PIL import Image

    from imagegen.manifest import MaskSettings

    source = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (5, 5, 5)).save(source)
    base = make_manifest(tmp_path, kind="mask", input_image=source, mask=MaskSettings())
    looser = base.model_copy(update={"mask": MaskSettings(threshold=0.6)})
    assert fp(looser, profile) != fp(base, profile)


def test_segmentation_target_not_the_prompt_drives_the_mask_name(tmp_path, profile):
    from PIL import Image

    from imagegen.manifest import MaskSettings

    source = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (5, 5, 5)).save(source)
    base = make_manifest(tmp_path, kind="mask", input_image=source, mask=MaskSettings())
    targeted = base.model_copy(update={"mask": MaskSettings(target="the dress")})
    assert fp(targeted, profile) != fp(base, profile)
