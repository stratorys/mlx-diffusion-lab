"""Content-addressed naming.

The fingerprint covers everything that changes the pixels, and nothing else. Job
identifiers, output paths and timestamps are deliberately excluded, so the same work
requested twice lands on the same filename.

That is what makes a sweep resumable. Rerunning skips what already exists, and a run
killed halfway continues where it stopped. Changing any parameter forks the name
instead of overwriting the earlier image.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from imagegen.manifest import Manifest
from imagegen.storage import sha256_file

LENGTH = 10


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:LENGTH]


def ingredients(manifest: Manifest, identity: dict[str, Any], runner: str) -> dict[str, Any]:
    """Everything that changes the pixels, as plain JSON-safe values.

    The input image contributes its content hash rather than its path. Two different
    paths holding the same bytes must produce the same fingerprint, and the same path
    holding new bytes must not.

    The runner name is in here so a placeholder written in stub mode can never occupy
    the filename a real render would claim, and so be skipped as already done.
    """
    payload: dict[str, Any] = {
        "kind": manifest.kind,
        "runner": runner,
        "prompt": manifest.prompt,
        "seed": manifest.seed,
        "steps": manifest.steps,
        "guidance": manifest.guidance,
        "width": manifest.width,
        "height": manifest.height,
        "identity": identity,
    }
    if manifest.image_strength is not None:
        payload["image_strength"] = manifest.image_strength
    if manifest.input_image is not None:
        payload["input_sha256"] = sha256_file(Path(manifest.input_image))
    if manifest.mask is not None:
        payload["mask"] = manifest.mask.model_dump()
        payload["mask"]["target"] = manifest.segmentation_target
    return payload


def of(manifest: Manifest, identity: dict[str, Any], runner: str) -> str:
    return digest(ingredients(manifest, identity, runner))
