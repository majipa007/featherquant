# Qwen3-8B direct-safetensors Q4_K_M under a 1 GiB ceiling (2026-08-27)

New scale point on the direct safetensors path (Qwen3-8B, previously only
validated at 0.6B), plus the first run of the multi-threaded ggml kernels.

## Setup

- Source: `Qwen/Qwen3-8B` sharded safetensors, 16.4 GB, 5 shards, BF16.
  Quantized **directly** — no BF16-GGUF intermediate; metadata/tokenizer came
  from a 5.7 MB `--vocab-only` GGUF (`scripts/make_vocab_gguf.sh`).
- Target: Q4_K_M (Q4_K base, Q6_K for output/attn_v/ffn_down per the
  `use_more_bits` schedule).
- Host: WSL2 (Linux 6.6.87.2), 15 GB RAM, i7-1355U, CPU-only. Model files on
  ext4. llama.cpp `25a1d63f4` (`libggml-base.so`).
- Enforcement: `systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0`.
- featherquant commit: `18933f0` (multi-threaded kernels), `--threads 12`.

## Result — PASS

| metric | value |
|---|---|
| source | 16.4 GB safetensors (399 tensors, 36 layers) |
| output | 4.7 GB Q4_K_M GGUF (5.0 GB written) |
| ceiling | 1 GiB, kernel-enforced (swap off) |
| **peak RSS** | **544 MiB** (`rss_metadata_peak` 563 MiB) |
| working budget | 677 MiB (RSS 283 MiB after metadata release + 64 MiB reserve) |
| budget violations | **0** |
| chunks | 546 |
| wall clock | 184 s (12 threads) |
| source-to-ceiling | ~15.3× |
| source-to-peak-RSS | ~30× |
| smoke | `llama-completion` exit 0, coherent generation |

The 563 MiB `rss_metadata_peak` (GGUFReader vocab transient on the 151k-token
vocab) is the true process peak and, as at 0.6B/14B, defines the ~600 MiB
minimum feasible budget on this vocab size — not the 1 GiB ceiling, which was
barely touched.

## Multi-threading (commit 18933f0)

`GgmlLib.quantize_rows(..., threads=N)` splits a row chunk across N workers
(rows independent, ctypes drops the GIL). Isolated kernel scaling on a 20000×1024
block, i7-1355U (12 logical cores):

| format | threads=1 | threads=8 | threads=12 | speedup |
|---|---|---|---|---|
| Q4_K | 1.371 s | 0.636 s | 0.562 s | **2.44×** |
| Q6_K | 0.872 s | 0.251 s | 0.260 s | **3.47×** |
| Q8_0 | 0.126 s | 0.035 s | 0.062 s | **3.56×** |

Bytes identical across every thread count and split (uneven splits and
workers > rows included; `tests/test_ggml_backend.py::test_threads_do_not_change_bytes`).

**End-to-end caveat:** this 8B run is I/O-bound — 16.4 GB read through a
~677 MiB window — so wall clock is dominated by disk and the single-threaded
BF16→F32 numpy conversion, not the kernel. Threading speeds up the compute
fraction only; do not read the 184 s as a threaded-vs-serial wall-clock win
(that comparison was not run). The isolated numbers above are the honest
scaling claim.

## Equivalence

Byte-equivalence vs `llama-quantize` is NOT re-verified at 8B here: producing a
reference needs a BF16 GGUF (16 GB) + a reference Q4_K_M (5 GB), which does not
fit the machine's free disk alongside the source. Kernel byte-parity was proven
at 0.6B where the reference exists (311/311 tensors, Q8_0 and Q4_K_M), and the
K-quant kernels are byte-exact by construction (`ggml_quantize_chunk` from
`libggml-base.so`). Coherent generation was confirmed (forbidden pattern: a
benchmark row whose output was never checked).

## Portable lib discovery (commit 18933f0)

The hardcoded `/home/sukuna/.../libggml-base.so` default is gone:
`default_lib_path()` resolves `$GGML_LIB`, then `~/llama.cpp/build*/bin/libggml-base.so`,
then errors naming the search. `q8_0` falls back to the numpy kernel when no
ggml library is present; K-quants still refuse loudly.
