# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import json

import pytest
from tests.conftest import CATALOGUE, FOUR_B

from imagegen import sweep as planning
from imagegen.cli import sweep as cli
from imagegen.config import ConfigError


def write_sweep(tmp_path, **overrides):
    body = {
        "profile": FOUR_B,
        "width": 256,
        "height": 256,
        "seeds": [1, 2],
        "prompts": [
            {"id": "alpha", "prompt": "a lighthouse"},
            {"id": "beta", "prompt": "a harbour"},
        ],
    }
    body.update(overrides)
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(body))
    return path


def build_plan(tmp_path, profile, **overrides):
    sweep = planning.Sweep.read(write_sweep(tmp_path, **overrides))
    return sweep, planning.plan(sweep, profile, tmp_path / "out", "stub")


def test_prompts_cross_with_seeds(tmp_path, profile):
    _, items = build_plan(tmp_path, profile)
    assert len(items) == 4
    assert {(item.entry_id, item.seed) for item in items} == {
        ("alpha", 1), ("alpha", 2), ("beta", 1), ("beta", 2)
    }


def test_an_entry_may_override_the_seed_list(tmp_path, profile):
    _, items = build_plan(
        tmp_path,
        profile,
        prompts=[
            {"id": "alpha", "prompt": "a lighthouse"},
            {"id": "beta", "prompt": "a harbour", "seeds": [9]},
        ],
    )
    assert [item.seed for item in items if item.entry_id == "beta"] == [9]


def test_the_common_prompt_is_prepended(tmp_path, profile):
    _, items = build_plan(tmp_path, profile, common_prompt="  cinematic   photograph ")
    assert items[0].manifest.prompt == "cinematic photograph a lighthouse"


def test_defaults_come_from_the_engine(tmp_path, profile):
    _, items = build_plan(tmp_path, profile)
    assert items[0].manifest.steps == profile.engine.default_steps("generate")
    assert items[0].manifest.guidance == profile.engine.guidance


def test_guidance_the_engine_forbids_is_refused(tmp_path, profile):
    with pytest.raises(ConfigError, match="distilled"):
        build_plan(tmp_path, profile, guidance=3.5)


def test_duplicate_entry_ids_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate entry id"):
        planning.Sweep.read(
            write_sweep(
                tmp_path,
                prompts=[{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}],
            )
        )


def test_an_entry_without_any_seed_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no seeds for entry"):
        planning.Sweep.read(write_sweep(tmp_path, seeds=[]))


def test_an_unsafe_entry_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="alphanumeric"):
        planning.Sweep.read(write_sweep(tmp_path, prompts=[{"id": "../escape", "prompt": "x"}]))


def test_existing_output_is_marked_done(tmp_path, profile):
    _, items = build_plan(tmp_path, profile)
    items[0].output.parent.mkdir(parents=True, exist_ok=True)
    items[0].output.write_bytes(b"already here")

    _, again = build_plan(tmp_path, profile)
    assert again[0].done is True
    assert len(planning.pending(again)) == 3


def test_force_ignores_existing_output(tmp_path, profile):
    _, items = build_plan(tmp_path, profile)
    items[0].output.parent.mkdir(parents=True, exist_ok=True)
    items[0].output.write_bytes(b"already here")

    sweep = planning.Sweep.read(write_sweep(tmp_path))
    forced = planning.plan(sweep, profile, tmp_path / "out", "stub", force=True)
    assert planning.pending(forced) == forced


def run_cli(tmp_path, monkeypatch, *extra):
    monkeypatch.setenv("IMAGEGEN_RUNNER", "stub")
    path = write_sweep(tmp_path)
    return cli.main([str(path), "--out", str(tmp_path / "out"), "--catalogue", str(CATALOGUE), *extra])


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    assert run_cli(tmp_path, monkeypatch, "--dry-run") == 0
    capsys.readouterr()
    assert not (tmp_path / "out").exists()


def test_a_sweep_produces_every_image_then_resumes_to_nothing(tmp_path, monkeypatch, capsys):
    assert run_cli(tmp_path, monkeypatch) == 0
    produced = sorted((tmp_path / "out").glob("*.png"))
    assert len(produced) == 4

    index = json.loads((tmp_path / "out" / "index.json").read_text())
    assert index["ran"] == 4 and index["failed"] == 0

    capsys.readouterr()
    assert run_cli(tmp_path, monkeypatch) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert sorted((tmp_path / "out").glob("*.png")) == produced


def test_a_deleted_image_is_the_only_thing_redone(tmp_path, monkeypatch, capsys):
    run_cli(tmp_path, monkeypatch)
    victim = sorted((tmp_path / "out").glob("*.png"))[1]
    victim.unlink()
    capsys.readouterr()

    assert run_cli(tmp_path, monkeypatch) == 0
    assert "[1/1]" in capsys.readouterr().out
    assert victim.exists()


def test_limit_caps_the_run(tmp_path, monkeypatch, capsys):
    assert run_cli(tmp_path, monkeypatch, "--limit", "2") == 0
    capsys.readouterr()
    assert len(list((tmp_path / "out").glob("*.png"))) == 2
