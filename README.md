# FeatherQuant

**Quantize models larger than your RAM.**

FeatherQuant is a memory-bounded, out-of-core LLM quantizer. It converts an
F16/BF16 GGUF model — or a sharded-safetensors Hugging Face checkpoint
directly — to Q8_0 or Q4_K_M while keeping peak process memory under a
user-configured budget, even when the source model is larger than that
budget. It trades time and disk I/O for memory, checkpoints after every
tensor, and resumes interrupted runs with verified state.

## How it works

```
GGUF metadata (no tensor data)         featherquant/gguf_io.py  TensorSource
        │
        ▼
plan: target type + exact output size/offset per tensor
        │                              featherquant/engine.py   quantize_model
        ▼
per tensor: read N rows → quantize → write at precomputed offset
  (N sized so read buffer + numpy temporaries + packed output
   fit max_ram − runtime RSS − 64 MiB reserve)
        │                              featherquant/q8_0.py     quantize_q8_0
        ▼
valid Q8_0 GGUF, byte-identical to llama-quantize
                                       featherquant/gguf_io.py  IncrementalWriter
```

Key properties:

- **Bounded memory** — tensor data moves through fixed, reusable buffers;
  no full-tensor or full-model materialization, ever.
- **Reference-exact** — the Q8_0 kernel byte-matches llama.cpp's
  `quantize_row_q8_0_ref` (float32 scale math, fp16 storage,
  round-half-away-from-zero). Verified tensor-for-tensor against
  `llama-quantize` output.
- **Deterministic** — same input and config produce a byte-identical file.
- **Externally verifiable** — ships a cgroup-v2 harness that runs the whole
  job under a kernel-enforced `MemoryMax` ceiling.
- **Crash-safe** — an atomic sidecar manifest checkpoints every committed
  tensor (sha256-verified); `--resume` continues an interrupted run and the
  result stays byte-identical (`scripts/kill_resume_test.sh` proves it under
  repeated SIGKILL).
- **Adaptive** — chunk sizes refine from live RSS feedback (EWMA
  controller); the static cost model is only the prior.

## Install

```bash
uv venv .venv
uv pip install -p .venv/bin/python -e '.[dev]'
```

Requires Python ≥ 3.10, Linux. Runtime deps: `numpy`, `gguf`.

## Usage

```bash
featherquant \
  --model  ./model-bf16.gguf \
  --output ./model-q8_0.gguf \
  --format q8_0 \
  --max-ram 1GB \
  --report ./run-report.json
```

`--max-ram` accepts `2GB`, `512M`, `1.5GiB`, or plain bytes. The report JSON
records peak RSS, working budget, bytes read/written, chunk count, budget
violations, adaptive-controller telemetry, and elapsed time. If the budget
cannot fit even one row of the largest tensor, featherquant exits early and
names the minimum feasible working set instead of thrashing.

Other flags: `--format q4_k_m` (K-quant output; kernels come byte-exact from
llama.cpp's `libggml-base.so` via ctypes — point `--ggml-lib`/`$GGML_LIB` at
it), `--resume` (continue an interrupted run), `--vocab-gguf` (required for
safetensors input).

### Quantize a Hugging Face checkpoint directly (no BF16 intermediate)

```bash
scripts/make_vocab_gguf.sh ./hf-model ./vocab.gguf   # tiny metadata-only GGUF
featherquant --model ./hf-model --vocab-gguf ./vocab.gguf \
  --output ./model-q8_0.gguf --format q8_0 --max-ram 1GB
```

Qwen3-family only for now (llama-family needs attn permutation on load —
refused rather than silently wrong). Output is byte-identical to running
`convert_hf_to_gguf.py --outtype bf16` + featherquant, without the ~2x
source-size intermediate file.

### Run under a hard OS ceiling

```bash
scripts/memlimit_run.sh model-bf16.gguf out-q8_0.gguf 1G
```

Runs the job inside a systemd cgroup with `MemoryMax=1G` and swap disabled;
exit 0 means the kernel never had a reason to kill it.

### Validate against llama.cpp

```bash
LLAMA_CPP_DIR=~/llama.cpp scripts/baseline.sh src.gguf ref_q8_0.gguf
python scripts/compare_reference.py ref_q8_0.gguf fq_q8_0.gguf
```

## Measured results (Qwen3-0.6B, BF16 1.5 GB → Q8_0)

| | llama-quantize | featherquant @ 1G ceiling |
|---|---|---|
| Peak RSS | 2.74 GB | 0.59 GB |
| Wall clock | 7.9 s | 30.1 s |
| Output | reference | byte-identical (311/311 tensors) |

Details in `docs/superpowers/plans/2026-07-20-baseline-notes.md`.

## Scope (current prototype)

- Input: little-endian GGUF (F32/F16/BF16 tensors) or sharded safetensors
  (Qwen3-family). Output: Q8_0 or Q4_K_M (byte-identical to
  `llama-quantize` on the validated model).
- Single process, single worker, CPU-only, Linux.
- Minimum feasible budget on 150k-token-vocab models is ~600 MiB: GGUF
  metadata parsing transiently allocates ~0.5 GiB (released before
  streaming; recorded as `rss_metadata_peak` in the report).

Planned next (see `project.md`): calibration/imatrix modes, more
architectures, leaner metadata parsing to lower the budget floor.

## Development

```bash
.venv/bin/pytest          # 59 tests
.venv/bin/ruff check featherquant tests scripts
.venv/bin/mypy featherquant scripts/compare_reference.py
```

TDD is the workflow: every behavior change starts with a failing test.
