"""Runners turn a manifest into an image.

IMAGEGEN_RUNNER picks the implementation. The stub honours the same contract as the
real one and loads no weights, so the queue, the API, the sweep tool and the contact
sheet are all exercisable on a machine that must not run a model.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import os

from imagegen.config import ResolvedProfile
from imagegen.runners.base import BaseRunner, Result

RUNNER_ENV = "IMAGEGEN_RUNNER"
DEFAULT_RUNNER = "stub"

__all__ = ["BaseRunner", "Result", "build_runner", "runner_name"]


def runner_name() -> str:
    return os.environ.get(RUNNER_ENV, DEFAULT_RUNNER).strip().lower()


def build_runner(profile: ResolvedProfile, name: str | None = None) -> BaseRunner:
    chosen = (name or runner_name()).lower()
    if chosen == "stub":
        from imagegen.runners.stub import StubRunner

        return StubRunner(profile)
    if chosen == "mflux":
        # Imported lazily so that neither mflux nor MLX is touched unless asked for.
        from imagegen.runners.mflux import MfluxRunner

        return MfluxRunner(profile)
    raise ValueError(f"unknown runner {chosen}; known: mflux, stub")
