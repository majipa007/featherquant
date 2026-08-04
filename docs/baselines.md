# Baselines 1-2 — RTN, unconstrained and under a ceiling

Per spec §6.1, before any FeatherQuant number is reported, baselines 1 and 2 must be
reproduced and recorded. This document is a runbook: the commands below have **not been
run yet**. No number appears here that was not produced by a manifest committed under
`bench/manifests/`.

## Prerequisites

- A built `llama.cpp` checkout at `~/llama.cpp` (CPU build), tools in `~/llama.cpp/build-cpu/bin`.
- `Qwen/Qwen3-14B` weights converted to bf16 GGUF at `~/models/qwen3-14b-bf16.gguf`.
- Record the exact llama.cpp revision used for these runs:

  ```bash
  git -C ~/llama.cpp rev-parse HEAD
  ```

  Paste the resulting commit hash here once the runs below have been executed:

  llama.cpp revision: `NOT YET RUN`

- Cold vs warm page cache changes I/O numbers substantially (spec §6). Set
  `FQ_DROP_CACHES=1` in the environment before invoking `bench/harness/run_baseline.sh` to
  drop caches (`sync && echo 3 > /proc/sys/vm/drop_caches`, requires `sudo`) before each run,
  and record here whether it was used.

  `FQ_DROP_CACHES` used: `NOT YET RUN`

## Baseline 1 — unconstrained RTN

```bash
LC=~/llama.cpp/build-cpu/bin
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-quantize\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_ref_q4_k_m.gguf\",\"Q4_K_M\"]" \
  m0_rtn_unconstrained rtn_q4_k_m Qwen/Qwen3-14B /tmp/m0_ref_q4_k_m.gguf nvme 0
```

Expected: exit 0, manifest written to `bench/manifests/m0_rtn_unconstrained.json`,
`peak_observed_bytes` recorded.

## Baseline 2 — RTN under descending ceilings

Run each ceiling until one is OOM-killed; every run gets its own manifest:

```bash
for L in 8G 4G 2G 1G 512M; do
  bash bench/harness/run_baseline.sh \
    "[\"bash\",\"bench/harness/run_under_ceiling.sh\",\"$L\",\"$LC/llama-quantize\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_ceil_$L.gguf\",\"Q4_K_M\"]" \
    "m0_rtn_ceiling_$L" rtn_q4_k_m Qwen/Qwen3-14B "/tmp/m0_ceil_$L.gguf" nvme \
    "$(python -c "import sys;print({'8G':8,'4G':4,'2G':2,'1G':1,'512M':0.5}['$L']*2**30)")"
done
```

Expected: the lowest passing ceiling and the first OOM-killed ceiling are both recorded in
`bench/manifests/`.

## Results

All fields below are read directly from the committed manifest for that `run_id`. Until the
runs above have been executed and their manifests committed, every row reads `NOT YET RUN`.

| run_id | method | ceiling | peak_observed | runtime_s | oom_killed | output_sha256 |
|---|---|---|---|---|---|---|
| m0_rtn_unconstrained | rtn_q4_k_m | unconstrained | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| m0_rtn_ceiling_8G | rtn_q4_k_m | 8 GiB | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| m0_rtn_ceiling_4G | rtn_q4_k_m | 4 GiB | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| m0_rtn_ceiling_2G | rtn_q4_k_m | 2 GiB | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| m0_rtn_ceiling_1G | rtn_q4_k_m | 1 GiB | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |
| m0_rtn_ceiling_512M | rtn_q4_k_m | 0.5 GiB | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |

## Why this is not a FeatherQuant result

Per spec §6.1: RTN K-quant conversion (`llama-quantize`, block-local rounding, no calibration
pass) streams the model tensor-by-tensor and never needs more than a small, roughly
constant working set regardless of total model size. Baselines 1 and 2 will therefore likely
**succeed**, including under low memory ceilings — **report that honestly**. It defines the
boundary of the contribution and is the difference between a credible project and one a
reviewer dismisses in the first paragraph. FeatherQuant's contribution is about the memory
behaviour of *calibrated* quantization methods (GPTQ-family, requiring a Hessian over
activations), not RTN — RTN succeeding under a ceiling is expected, not a demonstration of
anything FeatherQuant adds.

---

# Baseline 3 — imatrix + IQ-quant, unconstrained

Per spec §6.1 item 3. This section, like the two above, is a runbook: the commands below
have **not been run yet**. No number appears here that was not produced by a manifest
committed under `bench/manifests/`.

## Calibration corpus

Every perplexity number measured against this corpus — in this section and every later
one — must state its measurement context together:

- dataset: `wikitext-2-raw/wiki.test.raw`
- context length: `512`
- tokenizer: Qwen3

`bench/harness/fetch_calibration_corpus.sh [DEST]` fetches the corpus (default
`bench/data/wiki.test.raw`) and pins its sha256 alongside it at `DEST.sha256`. The
upstream `ggml-org/ci` dataset repo no longer serves `wiki.test.raw` at a raw path — it
only ships `wikitext-2-raw-v1.zip` (the same archive llama.cpp's own
`scripts/get-wikitext-2.sh` uses) — so the script downloads that zip to a temp path,
extracts `wikitext-2-raw/wiki.test.raw` from it to `DEST`, and pins/verifies the sha256 of
the **extracted corpus file**, never the zip, since the extracted file is what
`llama-perplexity` and `llama-imatrix` actually consume. The first run downloads and
writes the pin; every later run re-verifies the extracted file against that pin and fails
loudly on any mismatch, so a silently-changed corpus can never invalidate a comparison
already made against it:

```bash
bash bench/harness/fetch_calibration_corpus.sh
```

Only `bench/data/wiki.test.raw.sha256` is committed; `bench/data/wiki.test.raw` itself is
gitignored (large binary).

`bench/data/wiki.test.raw.sha256` pinned: **yes**, on 04/08/2026 —

```
173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08  bench/data/wiki.test.raw
```

extracted file size: 1,290,590 bytes.

## Step 1 — imatrix pass

```bash
LC=~/llama.cpp/build-cpu/bin
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-imatrix\",\"-m\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"-f\",\"bench/data/wiki.test.raw\",\"-o\",\"/tmp/m0_qwen3_14b.imatrix\",\"--chunks\",\"128\"]" \
  m0_imatrix_unconstrained imatrix Qwen/Qwen3-14B /tmp/m0_qwen3_14b.imatrix nvme 0
```

Expected: exit 0; `peak_observed_bytes` here is the number that shows why imatrix needs
model residency (spec §1.2).

## Step 2 — IQ-quant using that imatrix

```bash
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-quantize\",\"--imatrix\",\"/tmp/m0_qwen3_14b.imatrix\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_iq4_xs.gguf\",\"IQ4_XS\"]" \
  m0_iq4_xs_unconstrained iq4_xs Qwen/Qwen3-14B /tmp/m0_iq4_xs.gguf nvme 0
```

## Step 3 — measure perplexity and fill `quality`

```bash
$LC/llama-perplexity -m /tmp/m0_iq4_xs.gguf -f bench/data/wiki.test.raw -c 512 \
  2>&1 | tee /tmp/m0_iq4_xs_ppl.txt
python - <<'PY'
import re
from featherquant.run_manifest import RunManifest
m = RunManifest.load("bench/manifests/m0_iq4_xs_unconstrained.json")
m.quality = {"ppl": float(re.findall(r"Final estimate: PPL = ([\d.]+)",
                                     open("/tmp/m0_iq4_xs_ppl.txt").read())[-1]),
             "ppl_dataset": "wikitext-2-raw/wiki.test.raw c=512 tokenizer=qwen3",
             "tasks": {}}
m.save("bench/manifests/m0_iq4_xs_unconstrained.json")
PY
```

## Step 4 — confirm coherent generation before the number counts (required gate)

Per spec §11.4 / §10 ("Any benchmark row whose output was never checked for coherent
generation" is a forbidden pattern): a file that loads is not a file that works. This step
is not optional polish — no `quality.ppl` value from Step 3 may be copied into the Results
table below until this gate has been passed and its output pasted here.

```bash
$LC/llama-cli -m /tmp/m0_iq4_xs.gguf -p "The capital of Singapore is" -n 24 --temp 0
```

Expected: coherent continuation.

Coherence check output: `NOT YET RUN`

## Results

All fields below are read directly from the committed manifest for that `run_id`. Until
the runs above have been executed, their manifests committed, and the coherence-check gate
(Step 4) passed, every row reads `NOT YET RUN`.

| run_id | method | ceiling | peak_observed | runtime_s | oom_killed | output_sha256 | ppl |
|---|---|---|---|---|---|---|---|
| m0_imatrix_unconstrained | imatrix | unconstrained | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | n/a |
| m0_iq4_xs_unconstrained | iq4_xs | unconstrained | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |

`ppl` measurement context (required alongside every value above): `wikitext-2-raw/wiki.test.raw`,
context length `512`, tokenizer Qwen3.

---

# Baseline 4 — reference GPTQ, unconstrained

Per spec §6.1 item 4. Like the sections above, this is a runbook: the commands below have
**not been run yet**. No number appears here that was not produced by a manifest committed
under `bench/manifests/`.

**Scope note:** the GPU run itself (Steps 2-3 below) is deferred to a human-run session — it
needs a separate throwaway venv (`gptqmodel` + `torch`) and a GPU (an RTX 5070 Ti for this
project), neither of which exist in the CI/dev environment that wrote this section. What is
committed here is the runner script (`bench/harness/run_gptq_reference.py`) and this runbook;
the perplexity, runtime, and peak-memory numbers below are honestly unfilled until that
session happens.

## Step 1 — the reference runner (committed)

`bench/harness/run_gptq_reference.py` is committed. It runs GPTQ with no memory ceiling on
the GPU, writes a spec §6 run manifest, and separately writes a per-linear reconstruction-
error file that Task 17's calibrator comparison consumes:

- `bench/manifests/<RUN_ID>.json` — the run manifest (`quality.ppl` filled by Step 3 below).
- `bench/manifests/<RUN_ID>_layer_errors.json` — `"<layer_index>.<role>" -> MSE` for the
  seven linear roles (`attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down`), computed
  by snapshotting each `nn.Linear` weight before `model.quantize()` overwrites it in place and
  comparing against the same module after quantization. If the installed `gptqmodel` version
  does not expose the model as a walkable `torch.nn.Module` (or the pre/post shapes never
  line up), this file is written with an explicit `{"unavailable": "<reason>"}` marker instead
  of a fabricated or partial error map — the human running Step 2 should check this file's
  content immediately and record here whether it produced real numbers or the marker.

It is **not** part of the shipped `featherquant` package — it lives in `bench/harness/` only,
is never imported by `featherquant/`, and its dependencies are never added to
`pyproject.toml`.

## Step 2 — run it on Qwen3-0.6B (human, GPU session)

```bash
uv venv /tmp/gptq-venv && /tmp/gptq-venv/bin/uv pip install gptqmodel torch datasets
bash bench/harness/fetch_calibration_corpus.sh   # verifies bench/data/wiki.test.raw
                                                  # against the pinned sha256 first
/tmp/gptq-venv/bin/python bench/harness/run_gptq_reference.py \
  ~/models/qwen3-0.6b /tmp/m0_gptq_qwen3_0.6b m0_gptq_reference
```

Expected: `bench/manifests/m0_gptq_reference.json` and
`bench/manifests/m0_gptq_reference_layer_errors.json` both written.

Calibration texts are read from `bench/data/wiki.test.raw` — the same corpus pinned in
Baseline 3 above (sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`,
1,290,590 bytes) — 128 samples of length > 512, matching spec §3.3's calibration shape.

## Step 3 — fill `quality.ppl`, or record the blocker honestly

Baselines 1-3 all measure `ppl` via `llama-perplexity` against a GGUF file, i.e. llama.cpp's
own tokenizer driven off the GGUF's embedded vocab. GPTQ's native output
(`GPTQModel.save`) is HF-format safetensors, not GGUF, so getting a *comparable* number
(same dataset, context length `512`, and tokenizer per spec §6.1/§11.4) requires the GPTQ
output to go through that same GGUF/llama-perplexity path — which may or may not be possible
depending on whether the installed toolchain's GGUF converter can repack a GPTQ-quantized
checkpoint (as opposed to a full-precision one) without silently re-quantizing it or changing
its tokenizer path.

The human running this step must pick one of the two honest outcomes below and record which:

**(a) Comparable path succeeds** — if the GPTQ output can be converted/loaded such that
`llama-perplexity` runs against it with the same tokenizer as Baselines 1-3, reuse Baseline
3 Step 3's exact procedure, targeting this run's manifest:

```bash
$LC/llama-perplexity -m <gptq-gguf-export> -f bench/data/wiki.test.raw -c 512 \
  2>&1 | tee /tmp/m0_gptq_reference_ppl.txt
python - <<'PY'
import re
from featherquant.run_manifest import RunManifest
m = RunManifest.load("bench/manifests/m0_gptq_reference.json")
m.quality = {"ppl": float(re.findall(r"Final estimate: PPL = ([\d.]+)",
                                     open("/tmp/m0_gptq_reference_ppl.txt").read())[-1]),
             "ppl_dataset": "wikitext-2-raw/wiki.test.raw c=512 tokenizer=qwen3",
             "tasks": {}}
m.save("bench/manifests/m0_gptq_reference.json")
PY
```

**(b) No comparable path** — if the toolchain cannot produce a number measured under the same
tokenizer (e.g. no lossless GPTQ-to-GGUF repack is available for this model/toolchain
version), leave `quality.ppl: null` as `run_gptq_reference.py` already writes it, and record
the blocker in the Results section below instead of publishing an incomparable number. An
honest gap beats a number that looks comparable but is not.

Outcome recorded: `NOT YET RUN`

## Step 4 — confirm coherent generation before the number counts (required gate)

Same requirement as Baseline 3 Step 4 (spec §11.4 / §10): a file that loads is not a file
that works. No `quality.ppl` value from Step 3 may be copied into the Results table below
until this gate has been passed and its output pasted here.

```bash
# whichever runtime loads the GPTQ output directly (gptqmodel's own generate(),
# or the GGUF export if Step 3(a) applies)
```

Expected: coherent continuation.

Coherence check output: `NOT YET RUN`

## Results

All fields below are read directly from the committed manifest for `m0_gptq_reference`.
Until Steps 2-4 above have been executed on a GPU and the manifests committed, every row
reads `NOT YET RUN`.

| run_id | method | ceiling | peak_observed | runtime_s | output_sha256 | ppl |
|---|---|---|---|---|---|---|
| m0_gptq_reference | gptq_reference_4bit_g128 | unconstrained | NOT YET RUN | NOT YET RUN | NOT YET RUN | NOT YET RUN |

`ppl` measurement context (required alongside the value above, once filled):
`wikitext-2-raw/wiki.test.raw`, context length `512`, tokenizer Qwen3 — or, per Step 3(b),
an explicit statement of the blocker if no comparable number could be produced.

Per-linear layer errors (`bench/manifests/m0_gptq_reference_layer_errors.json`, consumed by
Task 17): `NOT YET RUN`.

---

# Milestone M0 gate status

Per spec §6.1, M0 is gated on all four baselines above being run and their manifests
committed. **That gate is NOT MET.** None of the four runs have been executed yet in this
environment — every table above reads `NOT YET RUN`. Outstanding:

1. Baseline 1 (unconstrained RTN) — not run.
2. Baseline 2 (RTN under descending ceilings) — not run.
3. Baseline 3 (imatrix + IQ-quant, unconstrained) — not run.
4. Baseline 4 (reference GPTQ, unconstrained) — not run; additionally requires a human GPU
   session with a throwaway `gptqmodel`/`torch` venv, per the scope note above.

No claim of M0 completion should be made, quoted, or relied upon by any later milestone
until all four manifests exist under `bench/manifests/` and this document's tables are
updated from them.
