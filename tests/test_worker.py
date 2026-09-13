# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import time

import pytest
from tests.conftest import FOUR_B, make_manifest

from imagegen.jobs import Job
from imagegen.ledger import Ledger, next_identifier
from imagegen.worker import Worker


def make_worker(tmp_path, profile) -> Worker:
    return Worker(profile, "stub", Ledger(tmp_path / "ledger.jsonl"))


def make_job(worker, manifest, kind="generate") -> Job:
    return Job(
        id=worker.next_id(),
        kind=kind,
        status="queued",
        profile=FOUR_B,
        runner="stub",
        fingerprint="f" * 10,
        output=str(manifest.output),
        seed=manifest.seed,
    )


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_job_runs_and_reaches_done(tmp_path, profile):
    worker = make_worker(tmp_path, profile)
    worker.start()
    try:
        manifest = make_manifest(tmp_path, job_id=1)
        job = make_job(worker, manifest)
        worker.submit(job, manifest)
        assert wait_until(lambda: job.status == "done")
        assert manifest.output.is_file()
        assert job.duration_s is not None
    finally:
        worker.shutdown(grace=5.0)


def test_a_failing_job_is_recorded_not_swallowed(tmp_path, profile, monkeypatch):
    from imagegen.runners.stub import StubRunner

    monkeypatch.setattr(
        StubRunner, "render", lambda self, m: (_ for _ in ()).throw(RuntimeError("no pixels"))
    )
    worker = make_worker(tmp_path, profile)
    worker.start()
    try:
        manifest = make_manifest(tmp_path, job_id=1)
        job = make_job(worker, manifest)
        worker.submit(job, manifest)
        assert wait_until(lambda: job.status == "failed")
        assert "no pixels" in job.error
        assert not manifest.output.exists()
    finally:
        worker.shutdown(grace=5.0)


def test_one_job_runs_at_a_time(tmp_path, profile, monkeypatch):
    """The queue exists to keep exactly one job in flight."""
    monkeypatch.setenv("IMAGEGEN_STUB_DELAY", "0.05")
    worker = make_worker(tmp_path, profile)
    worker.start()
    try:
        jobs = []
        for index in range(4):
            manifest = make_manifest(
                tmp_path,
                seed=index,
                output=tmp_path / f"{index}.png",
                metadata=tmp_path / f"{index}.json",
            )
            job = make_job(worker, manifest)
            worker.submit(job, manifest)
            jobs.append(job)

        seen_running = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(job.status != "done" for job in jobs):
            seen_running.append(sum(1 for job in jobs if job.status == "running"))
            time.sleep(0.01)
        assert all(job.status == "done" for job in jobs)
        assert max(seen_running) <= 1
    finally:
        worker.shutdown(grace=5.0)


def test_queue_position_counts_only_what_is_ahead(tmp_path, profile, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_STUB_DELAY", "0.2")
    worker = make_worker(tmp_path, profile)
    worker.start()
    try:
        last = None
        for index in range(3):
            manifest = make_manifest(
                tmp_path,
                seed=index,
                output=tmp_path / f"{index}.png",
                metadata=tmp_path / f"{index}.json",
            )
            last = make_job(worker, manifest)
            worker.submit(last, manifest)
        assert worker.queue_position(last.id) in (1, 2)
    finally:
        worker.shutdown(grace=5.0, drain=True)


def test_shutdown_cancels_what_never_started(tmp_path, profile, monkeypatch):
    monkeypatch.setenv("IMAGEGEN_STUB_DELAY", "0.3")
    worker = make_worker(tmp_path, profile)
    worker.start()
    jobs = []
    for index in range(4):
        manifest = make_manifest(
            tmp_path, seed=index, output=tmp_path / f"{index}.png", metadata=tmp_path / f"{index}.json"
        )
        job = make_job(worker, manifest)
        worker.submit(job, manifest)
        jobs.append(job)
    cancelled = worker.shutdown(grace=5.0)
    assert cancelled
    assert all(worker.get(job_id).status == "cancelled" for job_id in cancelled)


def test_submitting_after_shutdown_is_refused(tmp_path, profile):
    worker = make_worker(tmp_path, profile)
    worker.start()
    worker.shutdown(grace=5.0)
    manifest = make_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="shutting down"):
        worker.submit(make_job(worker, manifest), manifest)


def test_identifiers_never_reuse_a_number_already_on_disk(tmp_path, profile):
    (tmp_path / "000007_abcdef0123_seed1.png").write_bytes(b"an earlier image")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    jobs, _ = ledger.replay()
    assert next_identifier(jobs, tmp_path) == 8


def test_an_interrupted_job_is_visible_after_a_restart(tmp_path, profile):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    job = Job(
        id=3, kind="generate", status="queued", profile=FOUR_B, runner="stub",
        fingerprint="f" * 10, output=str(tmp_path / "x.png"),
    )
    ledger.record(job, "queued")
    job.status = "running"
    ledger.record(job, "running")

    jobs, interrupted = Ledger(tmp_path / "ledger.jsonl").replay()
    assert interrupted == [3]
    assert jobs[3].status == "interrupted"
    assert "stopped while this job was running" in jobs[3].error
