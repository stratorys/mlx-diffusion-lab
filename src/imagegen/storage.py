"""Filesystem helpers. Every write lands atomically.

A reader must never observe a half-written image, manifest or sidecar. Each write
goes to a dotted temporary name in the destination directory, then renames, which is
atomic on the same filesystem.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

RUNS_ENV = "IMAGEGEN_RUNS"
DEFAULT_RUNS = Path("runs")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def runs_directory() -> Path:
    return Path(os.environ.get(RUNS_ENV, DEFAULT_RUNS)).expanduser()


def _temporary_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(path)
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def atomic_text(path: Path, payload: str) -> Path:
    return atomic_bytes(path, payload.encode("utf-8"))


def atomic_json(path: Path, payload: Any) -> Path:
    return atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_image(path: Path, image, **save_options) -> Path:
    """Save a PIL image, or anything exposing a compatible save method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(path)
    image.save(temporary, **save_options)
    if not temporary.exists():
        raise RuntimeError(f"the image writer produced nothing at {temporary}")
    temporary.replace(path)
    return path


def append_line(path: Path, record: Any) -> None:
    """Append one JSON line and flush it to disk.

    No lock is taken here: Ledger.append owns the lock, because under an ASGI server
    the request handler writes the queued line while the worker writes the rest.
    Call this directly only from a single writer.
    """
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def output_name(stem: str, fingerprint: str, seed: int, suffix: str = ".png") -> str:
    """Content-addressed output name. Changing any input forks the name."""
    return f"{stem}_{fingerprint}_seed{seed}{suffix}"


def sidecar_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.metadata.json")


def highest_numeric_stem(directory: Path) -> int:
    """The largest leading job number already on disk, so ids are never reused."""
    if not directory.is_dir():
        return 0
    seen = [0]
    for path in directory.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        head = path.stem.split("_", 1)[0]
        if head.isdigit():
            seen.append(int(head))
    return max(seen)
