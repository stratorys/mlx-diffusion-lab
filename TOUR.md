# A tour of the code

Read this with a terminal open. Every experiment below runs against the **stub
runner**, which honours the real contract and loads no weights, so each one finishes
in milliseconds and tells you something true about the design.

```bash
uv sync --extra dev
export IMAGEGEN_RUNNER=stub
```

## The map

Nine modules carry the design. The rest is plumbing.

| Module | Lines | What it decides |
|---|---|---|
| `config.py` | 250 | What an engine is, what a profile is, and what counts as the identity of a render |
| `manifest.py` | — | The one object a runner is given. Validated, frozen |
| `fingerprint.py` | 65 | What changes the pixels, hashed. The cache key and the filename |
| `storage.py` | 101 | Every write is temp-then-rename. Nothing half-written is ever read |
| `jobs.py` | 85 | The durable record, and which fields are allowed to reach the ledger |
| `ledger.py` | 87 | Append-only JSONL, replayed at boot to rebuild state |
| `worker.py` | 213 | One thread, one job in flight, the runner owned by nobody else |
| `sweep.py` | 162 | Planning, kept separate from execution so a dry run is the real plan |
| `runners/base.py` | — | The contract. `generate()` is all a runner must implement |

The dependency direction is strict and worth noticing: `config` and `storage` know
nothing about anything else; `fingerprint` knows `manifest` and `storage`; `worker`
knows `runners`; nothing under `src/` knows what a subject is.

---

## 1. The fingerprint is the cache

```bash
imagegen-sweep sweeps/example.json --out /tmp/tour --dry-run
imagegen-sweep sweeps/example.json --out /tmp/tour
imagegen-sweep sweeps/example.json --out /tmp/tour        # again
```

The third command reports everything as `skip`. Nothing consulted a database: the
name of a file that already exists *is* the record that the work was done.

Now change one character of one prompt in a copy of the sweep and re-plan:

```bash
sed 's/first light/last light/' sweeps/example.json > /tmp/tour.json
imagegen-sweep /tmp/tour.json --out /tmp/tour --dry-run
```

Only the entries whose prompt changed are marked `run`. The others keep their names
because nothing about their pixels changed. This is the whole of `fingerprint.py:33`.

**Why it matters:** resumability is not a feature that was added. It falls out of
naming things by their content.

## 2. The runner is part of the identity

```bash
imagegen-sweep sweeps/example.json --out /tmp/tour --runner mflux --dry-run
```

Every filename differs from the stub run, so a placeholder can never occupy the name
a real render would claim and be skipped as already done. See the docstring at
`fingerprint.py:35`.

**Why it matters:** the cheap escape hatch and the real thing must not be able to
poison each other's output.

## 3. Baking forks the fingerprint, and so does adapter order

Both `subject-4b` and `subject-4b-baked` name the same checkpoint, the same adapter
file and the same scale. Only `bake` differs.

```bash
python - <<'PY'
from imagegen.config import load_catalogue
c = load_catalogue()
a, b = c.resolve("subject-4b"), c.resolve("subject-4b-baked")
print("same adapters:", a.adapters == b.adapters)
print("same identity:", a.identity() == b.identity())
PY
```

Same adapters, different identity. A benchmark of one cannot overwrite the other.

Now try the other one. Swap the two adapter lines in `[profile.subject-9b]` and
compare identities before and after: order is preserved exactly and it is hashed, so
reordering a stack forks every output name.

**Why it matters:** two settings that look like tuning are actually part of what the
image *is*.

## 4. Moving the weights does not invalidate anything

```bash
python - <<'PY'
from pathlib import Path
from imagegen.config import load_catalogue
c = load_catalogue()
here  = c.resolve("subject-4b", Path("/one/place"))
there = c.resolve("subject-4b", Path("/somewhere/else"))
print("same identity:", here.identity() == there.identity())
PY
```

True. `Adapter.identity()` deliberately returns the file *name*, not the resolved
path (`config.py:130`). Relocating a weights directory must not fork every image ever
produced from it.

Contrast with `Adapter.resolved_path`, which refuses to leave the weights directory
at all — and note that it calls `.resolve()`, so **a symlink to a file outside is
rejected**. Point `IMAGEGEN_WEIGHTS` at the real directory instead.

## 5. Crash recovery, in one kill

```bash
export IMAGEGEN_RUNS=/tmp/tour-server
export IMAGEGEN_STUB_DELAY=5          # make jobs slow enough to interrupt
imagegen-serve --profile plain-4b --port 4242 &
sleep 2
for i in 1 2 3; do
  curl -s -X POST localhost:4242/generate -H 'content-type: application/json' \
    -d '{"prompt": "a stone harbour"}' ; echo
done
kill -9 %1                             # die with one job in flight
```

Restart and look:

```bash
imagegen-serve --profile plain-4b --port 4242 &
sleep 2
curl -s localhost:4242/jobs/1 ; echo
curl -s localhost:4242/healthz ; echo
```

Job 1 comes back as `interrupted` with a reason, not as missing. The identifier
counter resumes past the highest number ever used — and `next_identifier` consults
the **disk as well as the ledger** (`ledger.py:83`), so even a deleted ledger cannot
cause an existing image to be overwritten. Try it: delete `ledger.jsonl`, restart,
and watch the numbering continue anyway.

**Why it matters:** the failure mode of a long-running generator is being killed. The
design treats that as normal rather than exceptional.

## 6. The engine refuses what it cannot do

```bash
python - <<'PY'
from imagegen.config import load_catalogue, ConfigError
e = load_catalogue().engine["klein-4b"]
print(e.check_guidance(1.0))
try:
    e.check_guidance(3.5)
except ConfigError as err:
    print("refused:", err)
PY
```

A distilled checkpoint locks its guidance. The refusal is in the config layer, before
anything loads, so the mistake costs nothing.

## 7. The masked edit proves it did no harm

Every masked job records `outside_mask_max_delta`, and it must be zero: the composite
is asserted to be bit-identical outside the mask. `masking.py` runs under the stub
too, so the whole choreography — crop with padding, erase with deterministic noise,
denoise, paste back through the mask — is testable with no model:

```bash
uv run pytest tests/test_masking.py -v
```

## 8. Where the seams are

Three places are worth reading in full because they are where the design is decided,
not merely expressed:

- `config.py:179` `ResolvedProfile.identity()` — the answer to "what makes two
  renders the same render"
- `worker.py:143` `_loop` / `_run_one` — every state transition and its ledger line,
  in forty lines
- `runners/base.py` `run()` — timing, atomic save and sidecar happen here, once, so
  the stub and the real runner cannot drift apart

## Things the tests already assert

```bash
uv run pytest -q                     # 144 tests, no weights, no torch, no network
uv run pytest -q --collect-only | tail -1
```

Read `tests/test_fingerprint.py` first. It is the shortest description of what this
project believes.
