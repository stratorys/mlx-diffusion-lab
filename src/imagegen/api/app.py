"""The FastAPI application.

Nothing here imports a model. The worker loads the runner on its own thread after
startup, so the server answers immediately and reports `model_state` as loading until
it is ready. Requests that arrive meanwhile simply queue.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from imagegen.config import Catalogue, ResolvedProfile, load_catalogue
from imagegen.ledger import LEDGER_NAME, Ledger, next_identifier
from imagegen.runners import runner_name
from imagegen.storage import runs_directory
from imagegen.worker import Worker

PROFILE_ENV = "IMAGEGEN_PROFILE"


@dataclass
class Service:
    """Everything a request handler is allowed to touch."""

    catalogue: Catalogue
    profile: ResolvedProfile
    worker: Worker
    out_dir: Path
    runner: str


def build_service(
    profile_name: str | None = None,
    runner: str | None = None,
    catalogue_path: Path | None = None,
    out_dir: Path | None = None,
) -> Service:
    catalogue = load_catalogue(catalogue_path)
    chosen = profile_name or os.environ.get(PROFILE_ENV) or catalogue.profile_names[0]
    profile = catalogue.resolve(chosen)
    directory = Path(out_dir) if out_dir else runs_directory()
    directory.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(directory / LEDGER_NAME)
    jobs, interrupted = ledger.replay()
    if interrupted:
        listed = ", ".join(str(job_id) for job_id in interrupted)
        print(f"jobs interrupted by the last shutdown: {listed}", flush=True)

    worker = Worker(
        profile=profile,
        runner_name=(runner or runner_name()).lower(),
        ledger=ledger,
        jobs=jobs,
        counter=next_identifier(jobs, directory) - 1,
    )
    return Service(catalogue, profile, worker, directory, worker.runner_name)


def create_app(
    profile_name: str | None = None,
    runner: str | None = None,
    catalogue_path: Path | None = None,
    out_dir: Path | None = None,
) -> FastAPI:
    service = build_service(profile_name, runner, catalogue_path, out_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.worker.start()
        try:
            yield
        finally:
            cancelled = service.worker.shutdown(grace=30.0)
            if cancelled:
                print(f"{len(cancelled)} job(s) cancelled at shutdown", flush=True)

    app = FastAPI(
        title="imagegen",
        version="0.1.0",
        summary="Generate, edit and masked-edit images behind a single-consumer queue.",
        lifespan=lifespan,
    )
    app.state.service = service

    from imagegen.api.routes import router

    app.include_router(router)
    return app
