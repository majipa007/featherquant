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
`bench/data/wiki.test.raw`) and pins its sha256 alongside it at `DEST.sha256`. The first
run downloads the file and writes the pin; every later run re-verifies the file against
that pin and fails loudly on any mismatch, so a silently-changed corpus can never
invalidate a comparison already made against it:

```bash
bash bench/harness/fetch_calibration_corpus.sh
```

Only `bench/data/wiki.test.raw.sha256` is committed; `bench/data/wiki.test.raw` itself is
gitignored (large binary).

`bench/data/wiki.test.raw.sha256` pinned: `NOT YET RUN` — the URL in this script
(`https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw/wiki.test.raw`)
returned HTTP 404 when tried on 04/08/2026: the upstream `ggml-org/ci` dataset repo now
only carries `wikitext-2-raw-v1.zip`, not a raw path under `wikitext-2-raw/`. Whoever runs
this section for real must first resolve a working source for `wiki.test.raw` (e.g.
llama.cpp's own `scripts/get-wikitext-2.sh`, which downloads and unzips that same archive)
before the pin can be written — do not substitute a different corpus silently; update this
note with whatever URL actually worked.

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
