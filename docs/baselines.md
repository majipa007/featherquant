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
