"""Seconds per denoising step.

Seconds per step rather than per image, because it is the only figure comparable
across step counts and frame sizes.

Three inputs change the cost of a step for structural reasons, and each is worth
measuring against its own baseline rather than reasoning about:

  bake      the fold re-promotes the touched layers on a sub-8-bit checkpoint; the
            effect depends on how many layers the adapter reaches
  guidance  above a distilled engine's locked value a real CFG activates, which
            evaluates the model twice per step
  size      cost tracks the latent grid, (width/16) x (height/16) tokens, not pixels

Compare a profile against one that differs in exactly one of them. `bake` is part of
the fingerprint, so a pair of profiles cannot overwrite each other's output.

  uv run python bench/step_time.py --profiles plain-4b
  uv run python bench/step_time.py --profiles plain-4b --sizes 576x576 768x1024

Requires real weights:  uv sync --extra dev --extra mflux
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from imagegen.config import ConfigError, load_catalogue

PROMPT = "a stone harbour at first light, fishing boats, overcast"


@dataclass
class Measurement:
    profile: str
    checkpoint: str
    bake: bool
    width: int
    height: int
    steps: int
    guidance: float
    seconds: list[float] = field(default_factory=list)

    @property
    def latent_tokens(self) -> int:
        """What the transformer actually attends over. Cost tracks this, not pixels."""
        return (self.width // 16) * (self.height // 16)

    @property
    def median_s(self) -> float:
        return statistics.median(self.seconds)

    @property
    def per_step_s(self) -> float:
        return self.median_s / self.steps

    def row(self) -> dict:
        return {
            "profile": self.profile,
            "checkpoint": self.checkpoint,
            "bake": self.bake,
            "width": self.width,
            "height": self.height,
            "latent_tokens": self.latent_tokens,
            "steps": self.steps,
            "guidance": self.guidance,
            "runs": len(self.seconds),
            "median_s": round(self.median_s, 2),
            "per_step_s": round(self.per_step_s, 3),
            "min_s": round(min(self.seconds), 2),
            "max_s": round(max(self.seconds), 2),
        }


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {value!r}") from None
    # mflux crops silently to a multiple of 16. A benchmark that measured a size the
    # model never saw would be worse than no benchmark, so refuse instead.
    for name, side in (("width", width), ("height", height)):
        if side % 16:
            raise argparse.ArgumentTypeError(
                f"{name} {side} is not a multiple of 16; mflux would crop it silently"
            )
    return width, height


def measure(runner, profile, size, steps, guidance, repeats, warmup, out_dir) -> Measurement:
    width, height = size
    record = Measurement(
        profile=profile.name,
        checkpoint=profile.engine.checkpoint,
        bake=profile.bake,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
    )
    # The first image after a load pays for lazy graph construction and kernel
    # compilation. Including it would make every configuration look like the one that
    # happened to run first.
    for index in range(warmup + repeats):
        started = time.perf_counter()
        image = runner.generate(
            prompt=PROMPT,
            seed=1000 + index,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )
        elapsed = time.perf_counter() - started
        warming = index < warmup
        label = "warmup" if warming else f"run {index - warmup + 1}"
        print(
            f"    {label:<8} {elapsed:7.2f}s  {elapsed / steps:6.3f}s/step",
            flush=True,
        )
        if not warming:
            record.seconds.append(elapsed)
            if out_dir is not None:
                name = f"{profile.name}_{width}x{height}_s{steps}_g{guidance:g}_{index}.png"
                image.save(out_dir / name)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", nargs="+", required=True, help="catalogue profiles to compare")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(768, 1024)], metavar="WxH")
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="default: the engine's own")
    parser.add_argument("--guidance", nargs="+", type=float, default=None, help="default: the engine's own")
    parser.add_argument("--repeats", type=int, default=3, help="measured runs per configuration")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured runs before each configuration")
    parser.add_argument("--out", type=Path, default=Path("bench/results/step_time.csv"))
    parser.add_argument("--keep-images", action="store_true", help="also write what was generated")
    args = parser.parse_args(argv)

    try:
        catalogue = load_catalogue()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    image_dir = None
    if args.keep_images:
        image_dir = args.out.parent / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

    results: list[Measurement] = []
    for name in args.profiles:
        try:
            profile = catalogue.resolve(name)
            profile.verify_weights()
        except ConfigError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        engine = profile.engine
        steps_list = args.steps or [engine.default_steps("generate")]
        guidance_list = args.guidance or [engine.guidance]

        # One load per profile. On this stack the load dominates a single image, so
        # reloading per configuration would measure the loader, not the model.
        from imagegen.runners import build_runner

        print(f"\n{name}: loading {engine.checkpoint} (bake={profile.bake})", flush=True)
        runner = build_runner(profile, "mflux")
        load_started = time.perf_counter()
        runner.load()
        print(f"  loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

        try:
            for size in args.sizes:
                for steps in steps_list:
                    for guidance in guidance_list:
                        try:
                            guidance = engine.check_guidance(guidance)
                        except ConfigError as error:
                            print(f"  skipping guidance {guidance:g}: {error}", flush=True)
                            continue
                        print(
                            f"  {size[0]}x{size[1]}  {steps} steps  guidance {guidance:g}",
                            flush=True,
                        )
                        results.append(
                            measure(runner, profile, size, steps, guidance, args.repeats, args.warmup, image_dir)
                        )
        finally:
            runner.close()

    if not results:
        print("nothing measured", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.row() for record in results]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'profile':<20} {'size':>10} {'steps':>6} {'guid':>6} {'s/step':>8} {'median':>8}")
    for record in results:
        print(
            f"{record.profile:<20} {record.width}x{record.height:>4} {record.steps:>6} "
            f"{record.guidance:>6.2f} {record.per_step_s:>8.3f} {record.median_s:>8.2f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
