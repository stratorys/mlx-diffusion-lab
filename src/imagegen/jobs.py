"""What the server knows about one unit of work.

A job is the durable record. The manifest is what a runner sees; the job is what the
ledger stores and what GET /jobs/{id} returns. An upload is a job too, which is what
lets lineage be a single graph of integer identifiers rather than a mix of ids and
client-supplied paths.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

JobKind = Literal["generate", "edit", "mask", "upload"]
Status = Literal["queued", "running", "done", "failed", "cancelled", "interrupted"]

TERMINAL: frozenset[str] = frozenset({"done", "failed", "cancelled", "interrupted"})

# Only these fields reach the ledger. Anything else is derived and would only rot.
LEDGER_FIELDS = (
    "id", "kind", "status", "profile", "runner", "fingerprint", "parent_id",
    "prompt_sha256", "seed", "steps", "guidance", "width", "height",
    "output", "metadata", "manifest_path",
    "created_at", "started_at", "finished_at", "duration_s", "error", "metrics",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    id: int
    kind: JobKind
    status: Status
    profile: str
    runner: str
    fingerprint: str
    output: str
    metadata: str | None = None
    manifest_path: str | None = None
    parent_id: int | None = None
    prompt_sha256: str | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    created_at: str = field(default_factory=now)
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def succeeded(self) -> bool:
        return self.status == "done"

    def output_path(self) -> Path:
        return Path(self.output)

    def ledger_record(self, event: str) -> dict[str, Any]:
        record = {key: getattr(self, key) for key in LEDGER_FIELDS}
        record["event"] = event
        record["ts"] = now()
        return record

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Job:
        known = {key: record.get(key) for key in LEDGER_FIELDS if key in record}
        known.setdefault("metrics", {})
        if known.get("metrics") is None:
            known["metrics"] = {}
        return cls(**known)
