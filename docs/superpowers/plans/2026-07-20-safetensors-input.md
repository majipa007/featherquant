# Safetensors Input Implementation Plan (Phase 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `featherquant --model <hf-dir>` quantizes a sharded-Safetensors Hugging Face model directly to GGUF — no ~30 GB BF16 intermediate file, process stays memory-bounded.

**Architecture:** Two-source design. (1) Model METADATA (architecture KVs + tokenizer) comes from a tiny `--vocab-only` GGUF produced by llama.cpp's battle-tested `convert_hf_to_gguf.py` — megabytes, seconds, memory-cheap, and removes the need to reimplement tokenizer conversion. (2) Tensor DATA is read directly from the safetensors shards with an explicit parser (the format is trivial: 8-byte little-endian header length, JSON header of `{name: {dtype, shape, data_offsets}}`, raw row-major data), sliced row-wise into the existing reusable-buffer pipeline. HF→GGUF tensor names map via `gguf.tensor_mapping.get_tensor_name_map`. `SafetensorsSource` implements the same duck-typed interface as `TensorSource`, so the engine loop does not change.

**Tech Stack:** stdlib `json`/`struct`, existing `gguf` package (`tensor_mapping`, `MODEL_ARCH`). No `safetensors` dependency — the parser is ~40 lines and gives exact control over reads.

## Global Constraints

- All MVP global constraints apply (budget contract, no full-tensor materialization, determinism).
- Scope: **Qwen3 dense architecture only** in v1. Llama-family needs attn_q/attn_k rope permutation on load — explicitly out of scope; the source detector must REFUSE unknown architectures with a clear message, never emit silently-wrong weights.
- Acceptance bar: `featherquant hf-dir → q8_0.gguf` must be byte-identical to the two-step pipeline (`convert_hf_to_gguf --outtype bf16` then `featherquant q8_0`) on Qwen3-0.6B — same KVs, same tensor names/order/types/bytes.
- Tensor iteration order must be deterministic: sort by (shard filename, header order) at plan time and record it.
- Gates green per task; commits by majipa007. Execute after resume + adaptive plans; rebase engine touches.

## File Structure

```
featherquant/st_source.py    — safetensors shard parser + SafetensorsSource (new)
featherquant/engine.py       — source dispatch: dir -> SafetensorsSource (modify, small)
featherquant/cli.py          — accept a directory as --model, --vocab-gguf flag (modify)
scripts/make_vocab_gguf.sh   — wraps convert_hf_to_gguf.py --vocab-only (new)
tests/test_st_source.py      — parser + slicing unit tests on synthetic shards (new)
tests/test_st_e2e.py         — equivalence vs two-step pipeline (real model, skippable)
```

---

### Task 1: Safetensors shard parser

**Files:**
- Create: `featherquant/st_source.py` (parser half)
- Test: `tests/test_st_source.py`

**Interfaces:**
- Consumes: nothing from featherquant.
- Produces: `parse_shard_header(path: str) -> tuple[dict[str, StTensor], int]` where `StTensor = (name, dtype: str, shape: tuple[int, ...], start: int, end: int)` and the int is the absolute file offset where data begins; `read_st_rows(f: BinaryIO, t: StTensor, data_base: int, row_start: int, n_rows: int, buf: bytearray) -> np.ndarray` returning float32 (BF16/F16/F32 supported, reusing `q8_0.bf16_to_f32`). Rows are along the LAST dim (row-major), matching GGUF ne0.

- [ ] **Step 1: Write failing tests** — build a synthetic single-file shard in the test (pure stdlib):

```python
"""Safetensors parser tests against a hand-built shard."""
import json
import struct

import numpy as np

from featherquant.st_source import parse_shard_header, read_st_rows


def write_shard(path, tensors):
    """Minimal safetensors writer: header-len u64, JSON header, raw data."""
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        dtype = {"<f4": "F32", "<f2": "F16", "<u2": "BF16"}[arr.dtype.str]
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj")))
        f.write(hj)
        for b in blobs:
            f.write(b)


def test_parse_and_slice(tmp_path):
    a = (np.arange(6 * 32, dtype=np.float32) / 8).reshape(6, 32).astype(np.float16)
    p = tmp_path / "s.safetensors"
    write_shard(p, {"model.layers.0.q.weight": a})
    tensors, data_base = parse_shard_header(str(p))
    t = tensors["model.layers.0.q.weight"]
    assert t.shape == (6, 32) and t.dtype == "F16"
    with open(p, "rb") as f:
        buf = bytearray(2 * 32 * 2)
        x = read_st_rows(f, t, data_base, 2, 2, buf)
    assert np.array_equal(x, a[2:4].astype(np.float32).ravel())
```

(Fix the obvious `hj"` typo when transcribing; add a BF16 case mirroring `test_read_rows_bf16`.)

- [ ] **Step 2: Watch fail. Step 3: Implement** (parser: `struct.unpack("<Q", f.read(8))`, `json.loads`, skip the `__metadata__` key; slicing mirrors `TensorSource.read_rows_f32` with `row_len = shape[-1]`). **Step 4: Pass, gates, commit** — `feat: minimal safetensors shard parser with row slicing`.

### Task 2: SafetensorsSource (engine-compatible)

**Files:**
- Modify: `featherquant/st_source.py`
- Test: extend `tests/test_st_source.py`

**Interfaces:**
- Consumes: Task 1; `gguf.tensor_mapping.get_tensor_name_map`, `gguf.MODEL_ARCH`; a vocab-only GGUF path for KV metadata.
- Produces: `class SafetensorsSource` duck-typing what the engine uses from `TensorSource`:
  - `reader` — a `GGUFReader` opened on the VOCAB-ONLY GGUF (gives `IncrementalWriter` its KV metadata verbatim; `general.architecture` must read `qwen3` or init raises).
  - `tensors` — list of lightweight entries each exposing `.name` (GGUF name from the name map + `.weight` suffix), `.shape` (ggml ne-order = reversed HF shape), `.tensor_type` (`GGMLQuantizationType.BF16/F16/F32` from the shard dtype), `.n_elements`, `.n_bytes`, plus private shard-routing fields. Deterministic order: `model.safetensors.index.json` weight_map sorted by (shard, header position); single-file models use header order.
  - `read_rows_f32(tensor, row_start, n_rows, buf)` / `read_raw(...)` / `close()` — same signatures as `TensorSource`.
  - Init takes `(model_dir: str, vocab_gguf: str)`; raises `RuntimeError` listing any HF tensor the name map cannot translate (refuse, don't skip).

- [ ] **Step 1: Failing test** — synthetic “model dir”: two shards + index.json + a vocab-only GGUF faked with `make_gguf(..., arch="qwen3")` from conftest; assert name translation (`model.layers.0.self_attn.q_proj.weight` → `blk.0.attn_q.weight`), ne-order shapes, cross-shard ordering, and `read_rows_f32` equality against source arrays.
- [ ] **Step 2: Watch fail. Step 3: Implement. Step 4: Pass, gates, commit** — `feat: SafetensorsSource with HF-to-GGUF name mapping`.

### Task 3: Engine + CLI dispatch

**Files:**
- Modify: `featherquant/engine.py`, `featherquant/cli.py`
- Create: `scripts/make_vocab_gguf.sh`

**Interfaces:**
- Consumes: Task 2.
- Produces: `quantize_model` gains `vocab_gguf: str | None`; source selection: `os.path.isdir(src)` → `SafetensorsSource(src, vocab_gguf)` (missing `vocab_gguf` → `SystemExit` telling the user to run `scripts/make_vocab_gguf.sh`), else `TensorSource(src)`. CLI: `--vocab-gguf` flag. Script:

```bash
#!/usr/bin/env bash
# Produce the tiny metadata/tokenizer-only GGUF featherquant needs for
# safetensors input. Usage: scripts/make_vocab_gguf.sh HF_DIR OUT.gguf
set -euo pipefail
CONVERT_PY=${CONVERT_PY:-/home/sukuna/models/.convert-venv/bin/python}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-/home/sukuna/llama.cpp}
"$CONVERT_PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$1" --outfile "$2" --vocab-only
```

- [ ] **Step 1: Failing test** — CLI end-to-end on the synthetic model dir from Task 2 (vocab GGUF from conftest): output opens in `GGUFReader`, quantized tensor bytes equal `quantize_q8_0` of the source rows.
- [ ] **Step 2: Watch fail. Step 3: Implement. Step 4: Pass, gates, commit** — `feat: quantize sharded safetensors directly`.

### Task 4: Real-model equivalence gate

- [ ] **Step 1:** `scripts/make_vocab_gguf.sh ~/models/qwen3-0.6b ~/models/qwen3-0.6b-vocab.gguf` (seconds; file ~10 MB).
- [ ] **Step 2:** `featherquant --model ~/models/qwen3-0.6b --vocab-gguf ~/models/qwen3-0.6b-vocab.gguf --output ~/models/fq_st_q8_0.gguf --max-ram 1GB`.
- [ ] **Step 3:** `python scripts/compare_reference.py ~/models/fq_q8_0.gguf ~/models/fq_st_q8_0.gguf` → `0 mismatches` (two-step pipeline output is the reference). KV diff too: dump both KV sets and diff — allow only ordering-neutral differences; investigate anything else.
- [ ] **Step 4:** Ceiling run via `memlimit_run.sh` on the directory input; smoke via `llama-completion`. Record: peak temp-storage saved (~29 GB intermediate eliminated at 14B scale).
- [ ] **Step 5:** Update README input-formats section; commit docs — `docs: safetensors input results`.
