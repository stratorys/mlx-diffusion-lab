"""One thread, one job at a time.

Serialising work is not a politeness feature here, it is the architecture. A single
GPU gains nothing from concurrent denoising, so the queue exists to keep exactly one
job in flight and to make the wait visible rather than hidden behind contention.

The worker owns the runner. Nothing else touches it. The HTTP layer only ever
appends to the queue and reads the job index under the lock.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from imagegen.config import ResolvedProfile
from imagegen.jobs import Job, now
from imagegen.ledger import Ledger
from imagegen.manifest import Manifest
from imagegen.runners import build_runner

SENTINEL = None
ModelState = str  # "loading" | "ready" | "error" | "stopped"


class Worker:
    def __init__(
        self,
        profile: ResolvedProfile,
        runner_name: str,
        ledger: Ledger,
        jobs: dict[int, Job] | None = None,
        counter: int = 0,
    ):
        self.profile = profile
        self.runner_name = runner_name
        self.ledger = ledger
        self.jobs: dict[int, Job] = jobs or {}
        self.counter = counter
        self.lock = threading.Lock()
        self.queue: queue.Queue = queue.Queue()
        self.current: int | None = None
        self.accepting = True
        self.model_state: ModelState = "loading"
        self.model_error: str | None = None
        self._thread: threading.Thread | None = None
        self._runner = None

    # --- identifiers and registration ------------------------------------------

    def next_id(self) -> int:
        with self.lock:
            self.counter += 1
            return self.counter

    def register(self, job: Job, event: str = "created") -> Job:
        """Record a job that needs no work, such as an upload."""
        with self.lock:
            self.jobs[job.id] = job
        self.ledger.record(job, event)
        return job

    def submit(self, job: Job, manifest: Manifest) -> int:
        """Queue a job. Returns how many jobs are ahead of it."""
        with self.lock:
            if not self.accepting:
                raise RuntimeError("the server is shutting down")
            self.jobs[job.id] = job
            ahead = self.queue.qsize()
            self.queue.put((job.id, manifest))
        self.ledger.record(job, "queued")
        return ahead

    def get(self, job_id: int) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def find_by_fingerprint(self, fingerprint: str) -> Job | None:
        """An earlier job that already produced exactly this image.

        Skip-existing, lifted to the API. Asking twice for the same thing should cost
        nothing the second time.
        """
        with self.lock:
            for job in sorted(self.jobs.values(), key=lambda item: item.id):
                if job.fingerprint == fingerprint and job.succeeded and job.output_path().exists():
                    return job
        return None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "profile": self.profile.name,
                "runner": self.runner_name,
                "model_state": self.model_state,
                "model_error": self.model_error,
                "queue_depth": self.queue.qsize(),
                "current_job": self.current,
                "next_id": self.counter + 1,
                "accepting": self.accepting,
            }

    def queue_position(self, job_id: int) -> int | None:
        """How many queued jobs sit ahead of this one."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                return None
            return sum(1 for other in self.jobs.values() if other.status == "queued" and other.id < job_id)

    # --- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="imagegen-worker", daemon=True)
        self._thread.start()

    def _load_runner(self) -> None:
        try:
            runner = build_runner(self.profile, self.runner_name)
            if runner.name != "stub":
                self.profile.verify_weights()
            started = time.perf_counter()
            runner.load()
            self._runner = runner
            with self.lock:
                self.model_state = "ready"
                self.model_error = None
            print(f"runner {runner.name} ready in {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed load is reported, not fatal
            with self.lock:
                self.model_state = "error"
                self.model_error = str(error)
            print(f"runner unavailable: {error}", flush=True)

    def _loop(self) -> None:
        self._load_runner()
        while True:
            item = self.queue.get()
            if item is SENTINEL:
                with self.lock:
                    self.model_state = "stopped"
                if self._runner is not None:
                    self._runner.close()
                return
            job_id, manifest = item
            try:
                self._run_one(job_id, manifest)
            except Exception as error:  # noqa: BLE001 - bookkeeping must never kill the thread
                print(f"[{job_id}] internal error: {error}", flush=True)

    def _run_one(self, job_id: int, manifest: Manifest) -> None:
        job = self.jobs[job_id]
        with self.lock:
            job.status = "running"
            job.started_at = now()
            self.current = job_id
        self.ledger.record(job, "running")

        started = time.perf_counter()
        try:
            if self._runner is None:
                raise RuntimeError(self.model_error or "no runner is loaded")
            result = self._runner.run(manifest)
        except Exception as error:  # noqa: BLE001 - any runner failure becomes a failed job
            with self.lock:
                job.status = "failed"
                job.error = str(error)
                job.finished_at = now()
                job.duration_s = round(time.perf_counter() - started, 3)
                self.current = None
            self.ledger.record(job, "error")
            print(f"[{job_id}] failed: {error}", flush=True)
            return

        with self.lock:
            job.status = "done"
            job.finished_at = now()
            job.duration_s = result.duration_s
            job.metrics = result.metrics
            job.error = None
            self.current = None
        self.ledger.record(job, "done")
        print(f"[{job_id}] {result.output.name} in {result.duration_s:g}s", flush=True)

    def drain(self) -> list[int]:
        """Empty the queue, marking as cancelled everything never attempted."""
        cancelled = []
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is SENTINEL:
                continue
            job_id, _ = item
            with self.lock:
                job = self.jobs[job_id]
                job.status = "cancelled"
                job.finished_at = now()
            self.ledger.record(job, "cancelled")
            cancelled.append(job_id)
        return cancelled

    def shutdown(self, grace: float = 900.0, drain: bool = True) -> list[int]:
        with self.lock:
            self.accepting = False
        cancelled = self.drain() if drain else []
        self.queue.put(SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=grace)
        return cancelled
