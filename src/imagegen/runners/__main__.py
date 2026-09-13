"""One-shot runner entry: turn a single manifest into an image.

    python -m imagegen.runners --manifest job.json

The queue calls the same runner objects in-process. This entry point exists so a
manifest can be replayed by hand, which is how you debug a job after the fact.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imagegen.config import ConfigError, load_catalogue
from imagegen.manifest import Manifest
from imagegen.runners import build_runner, runner_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner", default=None, help="override IMAGEGEN_RUNNER")
    parser.add_argument("--catalogue", default=None, type=Path)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")

    try:
        manifest = Manifest.read(args.manifest)
        catalogue = load_catalogue(args.catalogue)
        profile = catalogue.resolve(manifest.profile)
        runner = build_runner(profile, args.runner)
        if runner.name != "stub":
            profile.verify_weights()
        runner.load()
        try:
            result = runner.run(manifest)
        finally:
            runner.close()
    except (ConfigError, ValueError) as error:
        print(f"{args.manifest.name}: {error}", file=sys.stderr)
        return 2

    print(f"{result.output}  {result.duration_s:g}s  runner={runner_name() if not args.runner else args.runner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
