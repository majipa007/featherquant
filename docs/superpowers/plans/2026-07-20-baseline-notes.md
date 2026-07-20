# FeatherQuant Phase 0–2 Baseline Notes (2026-07-20)

## Setup

- Model: Qwen/Qwen3-0.6B, downloaded from Hugging Face 2026-07-20.
- Converted to BF16 GGUF with `convert_hf_to_gguf.py --outtype bf16`
  (llama.cpp commit `25a1d63f4346b472e508c6dbd9ab2ed1d81ace2e`).
- Source GGUF: `/home/sukuna/models/qwen3-0.6b-bf16.gguf`, 1.5 GB, 311 tensors.
- Hardware: WSL2 (Linux 6.6.87.2), CPU-only. Working tree on /mnt/c (slow 9p I/O);
  model files kept on ext4 (`~/models`) for honest I/O numbers.
- featherquant commit: see `git log` (Tasks 1–6 of the MVP plan).

## Phase 0 — conventional baseline (llama-quantize Q8_0)

- Command: `llama-quantize qwen3-0.6b-bf16.gguf ref_q8_0.gguf Q8_0`
  (binaries in `build-cpu/bin`, override via `LLAMA_BIN`).
- Peak RSS: **2,739,684 kB (~2.74 GB)** — confirms the full-model allocation.
- Wall clock: **7.93 s**.
- Reference sha256: `75a964d8f5b1404b63086fdec95ae666099a97315d5f04fd8338261a9b07c04c`.

## Phase 2 — featherquant under external ceiling

- Command: `scripts/memlimit_run.sh qwen3-0.6b-bf16.gguf fq_q8_0.gguf 1G`
  (systemd-run cgroup v2, `MemoryMax=1G`, `MemorySwapMax=0`).
- Source (1.5 GB) > ceiling (1 GB): satisfies success criterion 1 at prototype scale.
- Result: **PASS**, exit 0 inside the enforced ceiling.
- Peak RSS (self-measured): **619,163,648 B (~590 MiB)** vs budget 1 GiB —
  `budget_violations: 0`.
- Wall clock: **30.1 s** (~3.8x the unrestricted baseline — expected trade-off:
  RAM for time/I/O).
- 327 chunks; 1.50 GB read, 0.80 GB written.

## Equivalence

- `scripts/compare_reference.py ref_q8_0.gguf fq_q8_0.gguf`:
  **311 tensors, 0 mismatches** — byte-identical output, including tensor
  types (Q8_0 rule `ndim>=2 && ne0%32==0` matched llama-quantize exactly for
  this model) and packed Q8_0 data (fp16-scale + roundf parity confirmed).

## Inference smoke test

- Note: this llama.cpp build's `llama-cli` is chat-first and ignores
  `--no-conversation` (interactive REPL hangs headless runs); use
  `llama-completion` for scripted smoke tests. Also pass `-c 512` — the
  default allocates Qwen3's full 40k context (~5 GB KV cache).
- `llama-completion -m fq_q8_0.gguf -p "The capital of France is" -n 8 --seed 1 -c 512 -t 8`
- Result: **PASS**, exit 0. Load time 428 ms, prompt eval 75.8 tok/s,
  coherent generation (model entered Qwen3 thinking mode: "Okay, so I need to").
  No GGUF validation errors.

## Success criteria status (spec §14, prototype scale)

1. Source > enforced RAM: PASS (1.5 GB source, 1 GB ceiling).
2. Inside external ceiling: PASS (systemd cgroup, exit 0).
3. Tensor larger than working budget: exercised via chunked streaming
   (327 chunks) and unit-tested chunk/full byte equivalence.
4. Valid loadable model: PASS (llama-completion).
5. Matches reference: PASS, byte-identical (311/311 tensors).
6. Recovery after interruption: out of scope (Phase 4).
7. RAM/runtime trade-off: 2.74 GB/7.9 s (baseline) vs 0.59 GB/30.1 s (featherquant).
8. Reproducible manifest: partial (report JSON + deterministic output).
