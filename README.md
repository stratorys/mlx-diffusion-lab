# mlx-diffusion-lab

Diffusion image generation on Apple Silicon, under a unified-memory budget.

Whether a model fits in unified memory is the easy question. The harder one is what
else is resident while it runs, for how long, and what that constraint does to the
shape of the code around it. This repository is one set of answers, small enough to
read in an afternoon.

One queue, one worker, three endpoints, a fingerprint, a ledger and two benchmarks.
Nothing under `src/` knows what it is generating: an engine is a checkpoint and its
limits, a profile binds an engine to an ordered adapter stack, and
`config/profiles.toml` is the only file allowed to name a subject.

```bash
uv sync --extra dev
uv run pytest          # 144 tests, ~3 s, no weights, no torch, no network
```

That command is the point of the repository as much as anything in `src/`. See
[Testing without weights](#testing-without-weights).

## What it does

```
POST /images     upload an image so it can be a parent            -> 201
POST /generate   prompt                       -> image            -> 202
POST /edit       parent image + prompt        -> image            -> 202
POST /mask       parent image + prompt        -> region repainted -> 202
GET  /jobs/{id}  status, metrics, errors
GET  /healthz    profile, model state, queue depth
```

One job runs at a time. A single GPU gains nothing from concurrent denoising, so the
queue exists to keep exactly one job in flight and to make the wait visible rather
than to parallelise anything.

## Engines and profiles

An engine is a checkpoint plus its text encoder and its limits, and knows nothing
about any subject. A profile binds an engine to an ordered stack of LoRA adapters.
Adding a subject is a config entry, never a code change.

| Profile | Engine | Adapters | Runs out of the box |
|---|---|---|---|
| `plain-4b` | FLUX.2 Klein 4B, stock encoder | none | yes |
| `subject-4b` | Klein 4B, replacement encoder | one local adapter | no, fictional paths |
| `subject-4b-baked` | same, `bake = true` | same | no, fictional paths |
| `subject-9b` | Klein 9B, replacement encoder | general, then subject | no, fictional paths |

The `subject-*` profiles are documentation that executes. They exist to exercise the
two config values most able to fail silently — the **order** of an adapter stack, and
the **shape** of a replacement encoder — and the test suite asserts against the real
catalogue file, so a careless edit to it fails the build.

A profile is chosen at startup, not per request: adapters are baked or bound at
construction, so switching profiles means reloading everything. Serving two means two
processes on two ports.

### Two settings that cost more than they look like they cost

**`bake`.** Folding adapters into the weights happens once at load instead of on every
forward pass. On a sub-8-bit checkpoint the fold cannot stay at the original
precision, so the touched layers are re-promoted — mflux names the count in its load
output. Whether that costs or saves time depends on how many layers your adapter
touches, which is why it is a catalogue value rather than an implementation detail,
and why every profile can be paired with an identical one that only differs by this
flag. `bake` is part of the fingerprint, so the two runs cannot overwrite each other.

**`guidance`.** A distilled engine locks it. Above the locked value a real
classifier-free guidance activates and evaluates the model twice per step, so it is
never only a quality knob. The engine refuses other values before anything loads.
There is no negative branch at all, so a negation in a prompt is never encoded as an
exclusion — it only introduces the concept it names.

Neither of those carries a number here on purpose. Both are measurable against their
own baseline in one command: [`bench/RESULTS.md`](bench/RESULTS.md).

## Iterating

The model load dominates the cost of one image, so the unit of work is a sweep rather
than a request. Outputs are named by a fingerprint of everything that changes the
pixels, so a rerun does only what is missing and a run killed halfway continues.

```bash
imagegen-sweep sweeps/example.json --out runs/harbour --dry-run   # print the plan
imagegen-sweep sweeps/example.json --out runs/harbour             # one model load
imagegen-sheet runs/harbour                                       # one page to judge
```

A sweep file crosses prompts with seeds, and an entry may override the seed list:

```json
{
  "profile": "plain-4b",
  "width": 768, "height": 1024,
  "common_prompt": "cinematic photograph",
  "seeds": [1, 2, 3],
  "prompts": [
    { "id": "harbour", "prompt": "a stone harbour at first light" },
    { "id": "market",  "prompt": "a night market", "seeds": [7] }
  ]
}
```

Keep the frame size fixed across a sweep. Drafting small and regenerating large with
the same seed does not give the same image, because the latent shape changes. And
keep resolutions to multiples of 16 — otherwise mflux crops, silently.

`sweeps/one-variable.json` is the other way to use this: same prompt, same seeds, one
thing changing. Two variables at once and the result is unreadable.

## The masked edit

`POST /mask` takes an image and a prompt and derives the region from the text with
CLIPSeg, which runs on the CPU so it never competes with the generator for memory.

Pass `target` as well as `prompt` whenever you can. You segment `the dress` and
generate `a blue silk dress`; CLIPSeg degrades on long styled prompts. Omitting it
returns a warning.

The region is cropped with padding, its pixels are erased with deterministic noise so
the model cannot simply copy what was there, the crop is denoised, and the result is
pasted back through the mask. Every job records `outside_mask_max_delta`, which must
be zero, and writes the heatmap, the mask, the crop, the erased condition and the
composite mask as diagnostics.

A mask that is empty, that covers more of the frame than allowed, or that the
segmenter is not confident about fails the job with a reason.

## Running it

```bash
uv sync --extra dev                                  # no weights, no torch, no mflux
IMAGEGEN_RUNNER=stub imagegen-serve --profile plain-4b
```

For real images, install the extras and point at the weights:

```bash
uv sync --extra dev --extra mflux --extra segment
export IMAGEGEN_WEIGHTS=/path/to/weights
IMAGEGEN_RUNNER=mflux imagegen-serve --profile plain-4b --port 4242
```

| Variable | Meaning |
|---|---|
| `IMAGEGEN_RUNNER` | `stub` or `mflux`. Defaults to `stub`, so weights are opt in. |
| `IMAGEGEN_WEIGHTS` | Directory holding local adapter files. |
| `IMAGEGEN_PROFILE` | Boot profile. `--profile` wins. |
| `IMAGEGEN_RUNS` | Where images, manifests and the ledger go. |
| `IMAGEGEN_SEGMENTER` | `clipseg` or `box`. |
| `IMAGEGEN_RESIDENT` | `both` keeps the text and edit wrappers loaded, `one` evicts. |
| `IMAGEGEN_PROFILES` | Path to the catalogue. |

## Measuring

Two scripts, because the settings that matter here are not the ones that look like
they matter, and nobody should take that on faith from a README.

```bash
uv run python bench/step_time.py --profiles plain-4b --sizes 576x576 768x1024
uv run python bench/memory.py --profile plain-4b
```

`step_time.py` reports seconds per step, comparable across resolutions, and records
`latent_tokens` — `(width/16) x (height/16)`, which is what cost actually tracks. At
768x1024 that is 3 072 tokens for the whole frame, which is why a tight crop beats a
larger canvas when detail is the goal.

`memory.py` reports the MLX peak around each phase, including the encoder swap, which
is the one operation whose transient peak can exceed a machine the steady state fits
in comfortably.

## Durability

Every write is a temporary file followed by a rename, so a reader never sees a partial
file. Every job transition is a line in `runs/ledger.jsonl`, replayed at boot. A job
that was running when the process died comes back as `interrupted` rather than
vanishing, and identifiers are never reused, so a crash cannot overwrite past output.

The fingerprint hashes the pixel-affecting identity of a run: the normalised prompt,
the checkpoint, the encoder, the ordered adapter stack with scales, the bake mode, the
frame, the steps, the guidance, and the seed *mode* rather than a resolved random
seed — so a random slot keeps a stable identity across runs.

## Testing without weights

The stub runner honours the same contract as the real one and writes a deterministic
placeholder. That single seam is what makes the queue, the API, the sweep planner, the
contact sheet and the entire masked-edit pipeline testable on a machine that must not
load a model.

```bash
uv run pytest          # 144 tests, ~3 s
uv run ruff check src tests bench
```

Four things only real weights can confirm, and they are one manual smoke test:

1. Both encoder swaps load. The 4B encoder sits in a subdirectory and the 9B one at
   the repository root, and that difference has no offline proof.
2. The adapter stacks apply in order and visibly change the output.
3. CLIPSeg finds sensible regions on real images, and the default threshold, dilation
   and padding suit your frame sizes.
4. Throughput, and whether both model wrappers fit in memory at once.

## License

License Mozilla Public License v2.0 (MPL v2.0). See [LICENSE](LICENSE).

Copyright (c) 2026, Lucas Jahier - Stratorys

MPL v2.0 is per-file copyleft: changes to these files stay open, but the code can be
combined with differently licensed work without that spreading.
