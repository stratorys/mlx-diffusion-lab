"""The endpoints.

Every image is referenced by job identifier, never by a path the client supplies.
An upload is itself a job, so lineage is one graph of integers and path traversal is
not part of the threat model.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from imagegen import fingerprint as naming
from imagegen.api.schemas import (
    EditRequest,
    GenerateRequest,
    Health,
    JobAccepted,
    JobStatus,
    MaskRequest,
)
from imagegen.config import ConfigError
from imagegen.jobs import Job, JobKind
from imagegen.manifest import Manifest, MaskSettings
from imagegen.storage import atomic_bytes, output_name, sha256_text, sidecar_for

router = APIRouter()

SEED_CEILING = 2**31
UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def service_of(request: Request):
    return request.app.state.service


def urls(job_id: int) -> tuple[str, str]:
    return f"/jobs/{job_id}", f"/jobs/{job_id}/image"


def accepted(service, job: Job, pending: int, reused: bool, warnings: list[str]) -> JobAccepted:
    status_url, image_url = urls(job.id)
    return JobAccepted(
        id=job.id,
        kind=job.kind,
        status=job.status,
        reused=reused,
        profile=service.profile.name,
        runner=service.runner,
        model_state=service.worker.snapshot()["model_state"],
        fingerprint=job.fingerprint,
        parent_id=job.parent_id,
        pending=pending,
        status_url=status_url,
        image_url=image_url,
        warnings=warnings,
    )


def as_status(service, job: Job) -> JobStatus:
    status_url, image_url = urls(job.id)
    return JobStatus(
        **{
            key: getattr(job, key)
            for key in (
                "id", "kind", "status", "profile", "runner", "fingerprint", "parent_id",
                "seed", "steps", "guidance", "width", "height", "created_at",
                "started_at", "finished_at", "duration_s", "error", "metrics",
            )
        },
        queue_position=service.worker.queue_position(job.id),
        status_url=status_url,
        image_url=image_url,
    )


def resolve_parent(service, parent_id: int) -> Job:
    parent = service.worker.get(parent_id)
    if parent is None:
        raise HTTPException(404, f"no job {parent_id}")
    if not parent.succeeded:
        raise HTTPException(409, f"job {parent_id} is {parent.status}, so it has no image to work from")
    if not parent.output_path().is_file():
        raise HTTPException(409, f"the image for job {parent_id} is no longer on disk")
    return parent


def mask_settings(body: MaskRequest) -> MaskSettings:
    given = {
        key: value
        for key, value in {
            "target": body.target,
            "threshold": body.threshold,
            "dilate": body.dilate,
            "feather": body.feather,
            "padding": body.padding,
            "size": body.size,
        }.items()
        if value is not None
    }
    return MaskSettings(**given)


def submit(service, kind: JobKind, body, parent: Job | None) -> JSONResponse:
    engine = service.profile.engine
    warnings: list[str] = []
    try:
        guidance = engine.check_guidance(body.guidance if body.guidance is not None else engine.guidance)
    except ConfigError as error:
        raise HTTPException(422, str(error)) from error

    # An unpinned seed is random, so the fingerprint is new every time and the
    # already-done shortcut cannot apply. That is the intended behaviour: asking for
    # a surprise twice should give two surprises.
    seed = body.seed if body.seed is not None else secrets.randbelow(SEED_CEILING)
    settings = mask_settings(body) if kind == "mask" else None
    if kind == "mask" and settings.target is None:
        warnings.append("no target given, the segmenter will use the prompt, which is usually worse")

    draft = Manifest(
        kind=kind,
        job_id=0,
        profile=service.profile.name,
        prompt=body.prompt,
        seed=seed,
        steps=body.steps or engine.default_steps(kind),
        guidance=guidance,
        width=body.width or engine.width,
        height=body.height or engine.height,
        input_image=parent.output_path() if parent else None,
        image_strength=getattr(body, "image_strength", None),
        mask=settings,
        output=service.out_dir / "pending.png",
        metadata=service.out_dir / "pending.json",
    )
    stamp = naming.of(draft, service.profile.identity(), service.runner)

    if not body.force:
        existing = service.worker.find_by_fingerprint(stamp)
        if existing is not None:
            payload = accepted(service, existing, 0, True, warnings).model_dump()
            return JSONResponse(payload, status_code=200)

    job_id = service.worker.next_id()
    image = service.out_dir / output_name(f"{job_id:06d}", stamp, seed)
    manifest = draft.model_copy(
        update={"job_id": job_id, "output": image, "metadata": sidecar_for(image)}
    )
    manifest_path = service.out_dir / "manifests" / f"{image.stem}.json"
    manifest.write(manifest_path)

    job = Job(
        id=job_id,
        kind=kind,
        status="queued",
        profile=service.profile.name,
        runner=service.runner,
        fingerprint=stamp,
        output=str(image),
        metadata=str(manifest.metadata),
        manifest_path=str(manifest_path),
        parent_id=parent.id if parent else None,
        prompt_sha256=sha256_text(manifest.prompt),
        seed=seed,
        steps=manifest.steps,
        guidance=manifest.guidance,
        width=manifest.width,
        height=manifest.height,
    )
    try:
        pending = service.worker.submit(job, manifest)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return JSONResponse(accepted(service, job, pending, False, warnings).model_dump(), status_code=202)


@router.post("/generate", response_model=JobAccepted, status_code=202)
def generate(body: GenerateRequest, request: Request):
    return submit(service_of(request), "generate", body, None)


@router.post("/edit", response_model=JobAccepted, status_code=202)
def edit(body: EditRequest, request: Request):
    service = service_of(request)
    return submit(service, "edit", body, resolve_parent(service, body.parent_id))


@router.post("/mask", response_model=JobAccepted, status_code=202)
def mask(body: MaskRequest, request: Request):
    service = service_of(request)
    return submit(service, "mask", body, resolve_parent(service, body.parent_id))


@router.post("/images", response_model=JobStatus, status_code=201)
def upload(request: Request, file: UploadFile):
    """Bring an outside image in, so it can be the parent of an edit or a mask."""
    service = service_of(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(415, f"unsupported image type {suffix or 'unknown'}")
    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(422, "the uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "the uploaded file is too large")

    job_id = service.worker.next_id()
    destination = service.out_dir / "inputs" / f"{job_id:06d}{suffix}"
    atomic_bytes(destination, payload)
    job = Job(
        id=job_id,
        kind="upload",
        status="done",
        profile=service.profile.name,
        runner=service.runner,
        fingerprint=f"upload-{job_id:06d}",
        output=str(destination),
        finished_at=None,
        metrics={"bytes": len(payload), "filename": file.filename},
    )
    service.worker.register(job, "uploaded")
    return as_status(service, job)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: int, request: Request):
    service = service_of(request)
    job = service.worker.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return as_status(service, job)


@router.get("/jobs/{job_id}/image")
def job_image(job_id: int, request: Request):
    service = service_of(request)
    job = service.worker.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    path = job.output_path()
    if not path.is_file():
        raise HTTPException(409, f"job {job_id} is {job.status} and has no image yet")
    return FileResponse(path)


@router.get("/healthz", response_model=Health)
def health(request: Request):
    return Health(**service_of(request).worker.snapshot())
