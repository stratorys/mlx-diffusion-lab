"""Run a sweep: every prompt crossed with every seed, one model load, resumable.

    imagegen-sweep sweeps/beach.json --out runs/beach
    imagegen-sweep sweeps/beach.json --out runs/beach --dry-run

Outputs are named by fingerprint, so a second run does only what is missing, and a
run killed halfway continues where it stopped. Nothing is ever overwritten: changing
a parameter forks the name instead.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from imagegen import sweep as planning
from imagegen.config import ConfigError, load_catalogue
from imagegen.runners import build_runner, runner_name
from imagegen.storage import atomic_json, runs_directory


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="JSON sweep file")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--profile", default=None, help="override the profile named in the sweep")
    parser.add_argument("--runner", default=None, help="override IMAGEGEN_RUNNER")
    parser.add_argument("--catalogue", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="redo work that already exists")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many images")
    parser.add_argument("--stop-on-error", action="store_true", help="abort instead of continuing")
    return parser.parse_args(argv), parser


def describe(items: list[planning.Item], out_dir: Path) -> None:
    todo = planning.pending(items)
    print(f"{len(items)} image(s) planned, {len(items) - len(todo)} already done, {len(todo)} to run")
    print(f"output: {out_dir}")
    for item in items:
        mark = "skip" if item.done else "run "
        print(f"  {mark}  {item.entry_id}  seed {item.seed}  {item.output.name}")


def main(argv: list[str] | None = None) -> int:
    args, _ = parse_args(argv)
    chosen_runner = (args.runner or runner_name()).lower()

    try:
        sweep = planning.Sweep.read(args.sweep)
        catalogue = load_catalogue(args.catalogue)
        profile = planning.resolve_profile(sweep, catalogue, args.profile)
        out_dir = args.out or runs_directory() / args.sweep.stem
        items = planning.plan(sweep, profile, out_dir, chosen_runner, args.force)
    except (ConfigError, ValueError) as error:
        print(f"{args.sweep}: {error}", file=sys.stderr)
        return 2

    describe(items, out_dir)
    todo = planning.pending(items)
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]

    if args.dry_run:
        print("dry run, nothing written")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    try:
        runner = build_runner(profile, chosen_runner)
        if runner.name != "stub":
            profile.verify_weights()
        started = time.perf_counter()
        runner.load()
        print(f"runner {runner.name} ready in {time.perf_counter() - started:.1f}s", flush=True)
    except (ConfigError, ValueError) as error:
        print(f"cannot start runner: {error}", file=sys.stderr)
        return 2

    records, failures = [], 0
    try:
        for index, item in enumerate(todo, start=1):
            item.manifest.write(item.manifest_path)
            label = f"[{index}/{len(todo)}] {item.entry_id} seed {item.seed}"
            try:
                result = runner.run(item.manifest)
            except Exception as error:  # noqa: BLE001 - one bad prompt must not end a long sweep
                failures += 1
                print(f"{label}  failed: {error}", file=sys.stderr, flush=True)
                records.append({"id": item.entry_id, "seed": item.seed, "status": "failed", "error": str(error)})
                if args.stop_on_error:
                    break
                continue
            print(f"{label}  {result.output.name}  {result.duration_s:g}s", flush=True)
            records.append(
                {
                    "id": item.entry_id,
                    "seed": item.seed,
                    "status": "done",
                    "fingerprint": item.fingerprint,
                    "output": str(result.output),
                    "duration_s": result.duration_s,
                }
            )
    finally:
        runner.close()

    atomic_json(
        out_dir / "index.json",
        {
            "sweep": str(args.sweep),
            "profile": profile.name,
            "runner": runner.name,
            "identity": profile.identity(),
            "finished_at": datetime.now(UTC).isoformat(),
            "planned": len(items),
            "ran": len(records),
            "failed": failures,
            "items": records,
        },
    )
    print(f"{len(records) - failures} produced, {failures} failed, index at {out_dir / 'index.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
