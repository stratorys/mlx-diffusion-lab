# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import json

import pytest
from PIL import Image
from tests.conftest import make_manifest

from imagegen.runners import build_runner
from imagegen.runners.stub import StubRunner


def run(profile, manifest):
    runner = StubRunner(profile)
    runner.load()
    try:
        return runner.run(manifest)
    finally:
        runner.close()


def test_the_same_job_twice_is_byte_identical(tmp_path, profile):
    """Determinism is what makes skip-existing safe and tests meaningful."""
    first = make_manifest(tmp_path, output=tmp_path / "a.png", metadata=tmp_path / "a.json")
    second = make_manifest(tmp_path, output=tmp_path / "b.png", metadata=tmp_path / "b.json")
    run(profile, first)
    run(profile, second)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_different_seeds_produce_different_images(tmp_path, profile):
    first = make_manifest(tmp_path, output=tmp_path / "a.png", metadata=tmp_path / "a.json")
    second = make_manifest(
        tmp_path, seed=99, output=tmp_path / "b.png", metadata=tmp_path / "b.json"
    )
    run(profile, first)
    run(profile, second)
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()


def test_the_image_has_the_requested_size(tmp_path, profile):
    manifest = make_manifest(tmp_path, width=320, height=448)
    run(profile, manifest)
    assert Image.open(manifest.output).size == (320, 448)


def test_the_sidecar_records_resolved_provenance(tmp_path, profile):
    manifest = make_manifest(tmp_path)
    result = run(profile, manifest)
    sidecar = json.loads(result.metadata.read_text())
    provenance = sidecar["provenance"]
    assert provenance["checkpoint"] == "mlx-community/flux2-klein-4b-4bit"
    assert provenance["encoder"]["subdir"] == "klein-4b-alt-text-encoder"
    assert len(provenance["adapters"]) == 1
    assert sidecar["runner"] == "stub"
    assert sidecar["prompt"] == manifest.prompt


def test_an_edit_blends_its_parent_so_lineage_is_visible(tmp_path, profile):
    parent = tmp_path / "parent.png"
    Image.new("RGB", (256, 256), (255, 0, 0)).save(parent)
    plain = make_manifest(tmp_path, output=tmp_path / "a.png", metadata=tmp_path / "a.json")
    derived = make_manifest(
        tmp_path,
        kind="edit",
        input_image=parent,
        output=tmp_path / "b.png",
        metadata=tmp_path / "b.json",
    )
    run(profile, plain)
    run(profile, derived)
    # Same prompt and seed, but the parent is mixed in, so the pixels differ.
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()


def test_a_manifest_for_another_profile_is_refused(tmp_path, profile):
    manifest = make_manifest(tmp_path, profile="subject-9b")
    with pytest.raises(ValueError, match="serves profile"):
        run(profile, manifest)


def test_nothing_is_written_when_rendering_fails(tmp_path, profile, monkeypatch):
    """A failed job must not leave a partial image behind for skip-existing to find."""
    manifest = make_manifest(tmp_path)
    monkeypatch.setattr(
        StubRunner, "render", lambda self, m: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        run(profile, manifest)
    assert not manifest.output.exists()
    assert not manifest.metadata.exists()


def test_the_runner_registry_honours_an_explicit_name(profile):
    assert build_runner(profile, "stub").name == "stub"
    with pytest.raises(ValueError, match="unknown runner"):
        build_runner(profile, "nonsense")
