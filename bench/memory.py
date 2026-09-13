"""Peak memory, phase by phase.

Unified memory has no separate pool to spill into. A model that fits is not the
question; the question is what coexists, and for how long. The peak counter is reset
around each phase so the phases are attributable rather than cumulative:

  1. loading the checkpoint             the adapter stack, and any encoder swap
  2. generating one image               the working set at your frame size
  3. release_mlx()                      how much of that was cache, not live tensors
  4. a second model wrapper resident    what IMAGEGEN_RESIDENT=both costs over one

Phase 1 is the one to watch when an engine declares a replacement encoder. Such an
encoder is distributed at full precision and quantised after loading, so for a moment
the full-precision copy and its quantised result are both resident. That is why
runners/mflux.py drops the old encoder and calls release_mlx() *before* the
replacement is materialised, and why a failure there has to be reported as memory
pressure rather than as the authentication error it superficially resembles.

  uv run python bench/memory.py --profile plain-4b
  uv run python bench/memory.py --profile plain-4b --edit

Requires real weights:  uv sync --extra dev --extra mflux
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from imagegen.config import ConfigError, load_catalogue

GB = 1024**3


class Peaks:
    """Thin wrapper over the MLX counters, so each phase reports its own peak."""

    def __init__(self):
        import mlx.core as mx

        self.mx = mx
        self.phases: list[dict] = []

    def reset(self) -> None:
        self.mx.reset_peak_memory()

    def record(self, label: str, elapsed: float) -> dict:
        entry = {
            "phase": label,
            "seconds": round(elapsed, 2),
            "peak_gib": round(self.mx.get_peak_memory() / GB, 2),
            "active_gib": round(self.mx.get_active_memory() / GB, 2),
            "cache_gib": round(self.mx.get_cache_memory() / GB, 2),
        }
        self.phases.append(entry)
        print(
            f"  {label:<28} peak {entry['peak_gib']:>6.2f} GiB   "
            f"active {entry['active_gib']:>6.2f}   cache {entry['cache_gib']:>5.2f}   "
            f"{entry['seconds']:>6.2f}s",
            flush=True,
        )
        return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None, help="default: the engine's own")
    parser.add_argument("--edit", action="store_true", help="also load the edit wrapper")
    parser.add_argument("--out", type=Path, default=Path("bench/results/memory.json"))
    args = parser.parse_args(argv)

    try:
        catalogue = load_catalogue()
        profile = catalogue.resolve(args.profile)
        profile.verify_weights()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        peaks = Peaks()
    except ImportError:
        print("error: mlx is not installed. uv sync --extra mflux", file=sys.stderr)
        return 2

    from imagegen.runners import build_runner
    from imagegen.runners.mflux import EDIT, release_mlx

    engine = profile.engine
    steps = args.steps or engine.default_steps("generate")
    encoder = "none" if engine.encoder is None else f"{engine.encoder.repo} -> Q{engine.encoder.quantize}"
    print(f"\n{args.profile}")
    print(f"  checkpoint {engine.checkpoint}")
    print(f"  encoder    {encoder}")
    print(f"  adapters   {len(profile.adapters)}, bake={profile.bake}\n")

    runner = build_runner(profile, "mflux")

    # Baseline first: whatever the interpreter is already holding is not the model's.
    peaks.reset()
    peaks.record("baseline", 0.0)

    # load() covers the checkpoint, the adapter stack and, when the engine asks for
    # one, the encoder swap. They are one phase here because mflux does not offer a
    # seam between them; the swap's own cost shows up as the jump against baseline.
    peaks.reset()
    started = time.perf_counter()
    runner.load()
    peaks.record("load (+ encoder swap)", time.perf_counter() - started)

    peaks.reset()
    started = time.perf_counter()
    runner.generate(
        prompt="a stone harbour at first light, fishing boats, overcast",
        seed=1,
        steps=steps,
        guidance=engine.guidance,
        width=args.width,
        height=args.height,
    )
    peaks.record(f"generate {args.width}x{args.height}", time.perf_counter() - started)

    # The cache is reusable allocations, not live tensors. Clearing it between jobs
    # gives memory back without dropping the transformer or the encoder, which is the
    # whole reason not to use a destructive teardown callback.
    peaks.reset()
    started = time.perf_counter()
    release_mlx()
    peaks.record("release_mlx()", time.perf_counter() - started)

    if args.edit:
        peaks.reset()
        started = time.perf_counter()
        runner._variant(EDIT)  # private on purpose: measuring the second wrapper is the point
        peaks.record("second wrapper resident", time.perf_counter() - started)

    runner.close()
    peaks.reset()
    peaks.record("after close()", 0.0)

    report = {
        "profile": args.profile,
        "provenance": profile.provenance(),
        "width": args.width,
        "height": args.height,
        "steps": steps,
        "resident": runner.resident,
        "phases": peaks.phases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
