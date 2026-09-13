# Measurements

This file holds results produced by the scripts beside it, on the machine that ran
them. **It ships empty on purpose.** Performance here depends on the checkpoint, the
adapter, the frame size and the hardware, so a number carried in from somewhere else
would be worse than no number: it would look like a claim about your machine.

Run the scripts, paste the rows, and say what you ran them on.

## What each benchmark measures, and why

### `step_time.py` — seconds per step

Seconds per step rather than seconds per image, because it is the only figure
comparable across step counts and frame sizes.

Three inputs change the cost of a step more than they look like they should, and each
one is a mechanism you can verify in the code rather than take on faith:

**`bake`.** Folding an adapter stack into the weights happens once at load instead of
on every forward pass. On a sub-8-bit checkpoint the fold cannot stay at the original
precision, so the touched layers are re-promoted — mflux says so in its load output,
naming the number of layers. Whether that costs or saves time depends on how many
layers the adapter touches and how big they are, which is exactly why it is a
catalogue value (`Profile.bake`) and measurable against its own baseline:

```bash
uv run python bench/step_time.py --profiles <profile> <the same profile with bake = true>
```

`bake` is part of the fingerprint, so the two runs cannot overwrite each other.

**`guidance`.** A distilled engine locks it (`Engine.check_guidance`). Above the
locked value a real classifier-free guidance activates, which evaluates the model
twice per step rather than once. The engine refuses other values before anything
loads, so on a locked engine this is not a variable you can sweep — it is a wall you
are told about early.

Note that this family has no negative branch at all. A negation in a prompt is never
encoded as an exclusion; it only introduces the concept it names.

**Frame size.** Cost tracks the latent grid, not the pixel count. The grid is
`(width / 16) x (height / 16)`, so 768x1024 is 48x64 = 3 072 tokens for the entire
frame. `step_time.py` records `latent_tokens` in its CSV for exactly this comparison,
and it is the argument for cropping tighter rather than rendering larger when detail
is what you want.

Frame sides must be multiples of 16 or mflux crops silently. `parse_size` refuses a
non-multiple rather than measuring a size the model never saw.

### `memory.py` — peak memory per phase

Unified memory has no second pool. Nothing spills, so a transient allocation that does
not fit does not get slower, it fails. What matters is therefore not the steady state
but what coexists, and for how long.

The script resets the MLX peak counter around each phase separately:

| Phase | What it exposes |
|---|---|
| baseline | what the interpreter already holds, so the rest is attributable |
| load (+ encoder swap) | the checkpoint, the adapter stack, and any encoder replacement |
| generate | the working set of one image at your frame size |
| `release_mlx()` | how much of that was reusable cache rather than live tensors |
| second wrapper resident | the cost of `IMAGEGEN_RESIDENT=both` over `one` |
| after close() | whether anything was retained |

The load phase is the interesting one when an engine declares a replacement encoder.
Such an encoder is distributed at full precision and quantised after loading, so for a
moment the full-precision copy and its quantised result are both resident. That is why
`swap_text_encoder` drops the old encoder and calls `release_mlx()` *before*
materialising the replacement, and why a failure in that window has to be reported as
memory pressure — it superficially resembles a gated-repository authentication error,
and mislabelling it sends you to the wrong problem.

Measure it rather than assume it:

```bash
uv run python bench/memory.py --profile <profile>
uv run python bench/memory.py --profile <profile> --edit
```

## Method

- Report **seconds per step**, and say the frame size and step count alongside it.
- Discard the first image after every model load. It pays for lazy graph construction
  and kernel compilation, and including it makes whichever configuration ran first
  look slowest. `--warmup 1` is the default for this reason.
- Change one thing at a time. Two variables and the result is unreadable.
- One model load per profile. Reloading between configurations measures the loader.
- State the machine, the checkpoint, the adapter rank, and the mflux and MLX versions.
  Without those a row means nothing to anyone else.

## Results

### Apple M4 Pro, 24 GB, macOS 26.6.2 — 2026-09-13

mflux 0.19.1, MLX 0.32.2, `mlx-community/flux2-klein-4b-4bit`, one local rank-64
adapter at scale 1.0, `bake = false`, 4 steps, guidance 1.0. Median of 3 runs after
1 warmup. Source: `bench/results/tokens.csv`.

| Size | Latent tokens | s/step | s/image | ms per 1000 tokens |
|---|---|---|---|---|
| 576x576 | 1 296 | 3.113 | 12.45 | 2.40 |
| 576x768 | 1 728 | 3.991 | 15.96 | 2.31 |
| 768x1024 | 3 072 | 5.897 | 23.59 | 1.92 |
| 832x1216 | 3 952 | 7.371 | 29.48 | 1.87 |

**Cost is linear in latent tokens, over a large fixed per-step overhead.**

```
t_step  ≈  1.17 s  +  1.56 s per 1000 latent tokens        R² = 0.9975
```

Maximum residual across the four points is 123 ms. Adding a quadratic term makes the
fit no better — its coefficient comes out slightly negative and contributes under 1%
at the largest frame — so there is no attention blow-up in this range. Whatever the
theory says about quadratic attention, it is not what this stack pays for at these
sizes.

The fixed 1.17 s is the interesting half. It is paid once per step whatever the frame
holds:

| Size | Share of each step that is fixed overhead |
|---|---|
| 576x576 | 37.5% |
| 576x768 | 29.2% |
| 768x1024 | 19.8% |
| 832x1216 | 15.8% |

**What follows from that, and it is not the obvious thing.** Cropping tighter to put
more pixels on the subject is still right, but not because it is cheap: going from
768x1024 to 576x576 divides the token count by 2.37 and the time by only 1.90. Small
frames are the *least* efficient way to spend a step, because the overhead does not
shrink with them. Per-token throughput improves monotonically with frame size here —
2.40 ms per 1000 tokens at the smallest, 1.87 at the largest.

Useful side effect: the model predicts. Planning a sweep is
`images x steps x (1.17 + 0.00156 x tokens)` seconds — 40 images at 768x1024 and
8 steps is about 31 minutes, and that estimate was accurate to a few percent when
checked against a fourth frame size the fit had not seen.

### Add further sections in this shape:

```markdown
### <machine>, <date>

mflux <version>, MLX <version>, <checkpoint>, <adapter rank if any>.
Source: bench/results/step_time.csv

| Profile | Size | Steps | Guidance | Bake | s/step | Median |
|---|---|---|---|---|---|---|
```

`bench/results/` is gitignored, so the raw CSV and JSON stay local. Copy the rows you
want to publish into this file, where they can be read next to the conditions that
produced them.

## Tooling behaviour worth knowing before you measure

Properties of the surrounding tools, not measurements. Each one is verifiable in a few
seconds and each one can quietly invalidate a benchmark:

- Frame sides that are not multiples of 16 are cropped silently.
- `mx.set_cache_limit` takes **decimal** GB, not binary. A 1.5 meant as GiB is 7%
  smaller than intended.
- `mx.clear_cache()` returns reusable allocations without dropping the transformer or
  the encoder. A destructive teardown callback returns more and costs a full reload.
- mflux bakes adapters by default; this catalogue does not. Compare like with like.
- Prompt unicode is normalised before hashing, so a curly quote does not fork the
  history — but it does mean two visually different prompts can share a fingerprint.
