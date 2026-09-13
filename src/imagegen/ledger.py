"""The durable record of every job, as one JSON object per line.

Append only. State is rebuilt by replaying it at boot, which is what makes a crash
survivable: the files that landed are still there, and the jobs that did not are
visible as interrupted rather than silently missing.

Unlike the reference server, appends take a lock. Under an ASGI server the request
handler writes the `queued` line while the worker writes the rest, so more than one
thread reaches this file.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from imagegen.jobs import Job
from imagegen.storage import append_line, highest_numeric_stem

LEDGER_NAME = "ledger.jsonl"


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            append_line(self.path, record)

    def record(self, job: Job, event: str) -> None:
        self.append(job.ledger_record(event))

    def lines(self) -> list[dict[str, Any]]:
        """Every parseable record.

        A process killed mid-write can leave a truncated final line. That line is
        skipped rather than treated as corruption, because the alternative is refusing
        to start over one lost job.
        """
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("id"), int):
                records.append(parsed)
        return records

    def replay(self) -> tuple[dict[int, Job], list[int]]:
        """Rebuild the job index, and name the jobs that were running at the crash."""
        merged: dict[int, dict[str, Any]] = {}
        for record in self.lines():
            job_id = record["id"]
            state = merged.setdefault(job_id, {})
            state.update({key: value for key, value in record.items() if key not in ("event", "ts")})
            event = record.get("event")
            if event == "running":
                # Superseded by a later done or error line. Surviving to the end of the
                # replay means the process died with this job in flight.
                state["status"] = "interrupted"
                state["error"] = "the server stopped while this job was running"
            elif event in ("done", "error", "cancelled"):
                state["status"] = record.get("status", state.get("status"))
                state["error"] = record.get("error")

        jobs = {job_id: Job.from_record(state) for job_id, state in merged.items()}
        interrupted = sorted(job_id for job_id, job in jobs.items() if job.status == "interrupted")
        return jobs, interrupted


def next_identifier(jobs: dict[int, Job], out_dir: Path) -> int:
    """One past the highest identifier ever used, on disk or in the ledger.

    Consulting the disk as well as the ledger means a lost ledger can never cause an
    existing image to be overwritten.
    """
    return max([*jobs, highest_numeric_stem(out_dir), 0]) + 1
