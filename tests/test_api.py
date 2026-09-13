# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.conftest import CATALOGUE, FOUR_B

from imagegen.api.app import create_app
from imagegen.jobs import Job
from imagegen.ledger import LEDGER_NAME, Ledger

SMALL = {"width": 256, "height": 256}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_RUNNER", "stub")
    app = create_app(FOUR_B, "stub", CATALOGUE, tmp_path / "runs")
    with TestClient(app) as opened:
        yield opened


def wait_for(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never settled")


def generate(client, **overrides):
    body = {"prompt": "a lighthouse at dusk", "seed": 7, **SMALL}
    body.update(overrides)
    return client.post("/generate", json=body)


def upload(client, tmp_path, colour=(200, 30, 30)):
    source = tmp_path / "parent.png"
    Image.new("RGB", (256, 256), colour).save(source)
    with source.open("rb") as handle:
        return client.post("/images", files={"file": ("parent.png", handle, "image/png")})


def test_generate_is_accepted_and_completes(client):
    response = generate(client)
    assert response.status_code == 202
    body = response.json()
    # The worker may already have picked it up, so either state is correct here.
    assert body["status"] in ("queued", "running") and body["reused"] is False
    assert body["status_url"] == f"/jobs/{body['id']}"

    settled = wait_for(client, body["id"])
    assert settled["status"] == "done" and settled["error"] is None
    assert client.get(f"/jobs/{body['id']}/image").status_code == 200


def test_health_reports_the_boot_profile(client):
    body = client.get("/healthz").json()
    assert body["profile"] == FOUR_B and body["runner"] == "stub"
    assert body["accepting"] is True


def test_asking_twice_for_the_same_image_costs_nothing(client):
    first = generate(client)
    wait_for(client, first.json()["id"])

    second = generate(client)
    assert second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["id"] == first.json()["id"]


def test_force_redoes_the_work(client):
    first = generate(client)
    wait_for(client, first.json()["id"])

    forced = generate(client, force=True)
    assert forced.status_code == 202
    assert forced.json()["id"] != first.json()["id"]


def test_an_unpinned_seed_is_a_new_image_every_time(client):
    first = generate(client, seed=None)
    wait_for(client, first.json()["id"])
    second = generate(client, seed=None)
    assert second.status_code == 202 and second.json()["reused"] is False


def test_an_unknown_field_is_refused(client):
    assert generate(client, sampler="euler").status_code == 422


def test_guidance_the_engine_forbids_is_refused(client):
    response = generate(client, guidance=3.5)
    assert response.status_code == 422
    assert "distilled" in response.json()["detail"]


def test_a_frame_that_is_not_model_sized_is_refused(client):
    assert generate(client, width=770).status_code == 422


def test_an_empty_prompt_is_refused(client):
    assert generate(client, prompt="   ").status_code == 422


def test_an_unknown_job_is_a_404(client):
    assert client.get("/jobs/4242").status_code == 404
    assert client.get("/jobs/4242/image").status_code == 404


def test_an_upload_becomes_a_job_that_can_be_a_parent(client, tmp_path):
    uploaded = upload(client, tmp_path)
    assert uploaded.status_code == 201
    assert uploaded.json()["kind"] == "upload" and uploaded.json()["status"] == "done"

    parent_id = uploaded.json()["id"]
    edit = client.post(
        "/edit", json={"prompt": "make it blue", "parent_id": parent_id, "seed": 1, **SMALL}
    )
    assert edit.status_code == 202 and edit.json()["parent_id"] == parent_id
    assert wait_for(client, edit.json()["id"])["status"] == "done"


def test_a_generated_image_can_be_edited(client):
    first = generate(client)
    parent_id = first.json()["id"]
    wait_for(client, parent_id)

    edit = client.post("/edit", json={"prompt": "warmer light", "parent_id": parent_id, "seed": 3, **SMALL})
    assert edit.status_code == 202
    assert wait_for(client, edit.json()["id"])["status"] == "done"


def test_editing_an_unknown_parent_is_a_404(client):
    assert client.post("/edit", json={"prompt": "x", "parent_id": 9999}).status_code == 404


def test_a_bad_upload_type_is_refused(client, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("not an image")
    with note.open("rb") as handle:
        response = client.post("/images", files={"file": ("note.txt", handle, "text/plain")})
    assert response.status_code == 415


def test_a_mask_request_warns_when_no_target_is_given(client, tmp_path):
    parent_id = upload(client, tmp_path).json()["id"]
    response = client.post(
        "/mask", json={"prompt": "a blue silk dress", "parent_id": parent_id, "seed": 1, **SMALL}
    )
    assert response.status_code == 202
    assert any("segmenter will use the prompt" in note for note in response.json()["warnings"])


def test_a_mask_request_with_a_target_does_not_warn(client, tmp_path):
    parent_id = upload(client, tmp_path).json()["id"]
    response = client.post(
        "/mask",
        json={"prompt": "a blue silk dress", "target": "the dress", "parent_id": parent_id, "seed": 1, **SMALL},
    )
    assert response.status_code == 202 and response.json()["warnings"] == []


def test_the_queue_serialises_and_reports_depth(client, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_STUB_DELAY", "0.05")
    accepted = [generate(client, seed=index).json() for index in range(4)]
    assert all(body["status"] in ("queued", "running") for body in accepted)
    assert max(body["pending"] for body in accepted) >= 1
    for body in accepted:
        assert wait_for(client, body["id"])["status"] == "done"


def test_a_restart_surfaces_a_job_that_was_running(tmp_path, monkeypatch):
    """A crash must leave evidence, not a silently missing job."""
    monkeypatch.setenv("IMAGEGEN_RUNNER", "stub")
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    ledger = Ledger(runs / LEDGER_NAME)
    job = Job(
        id=5, kind="generate", status="queued", profile=FOUR_B, runner="stub",
        fingerprint="a" * 10, output=str(runs / "000005_aaaaaaaaaa_seed1.png"),
    )
    ledger.record(job, "queued")
    job.status = "running"
    ledger.record(job, "running")

    app = create_app(FOUR_B, "stub", CATALOGUE, runs)
    with TestClient(app) as client:
        body = client.get("/jobs/5").json()
        assert body["status"] == "interrupted"
        # The identifier is never reused, so the next job cannot overwrite job five.
        assert client.get("/healthz").json()["next_id"] > 5
