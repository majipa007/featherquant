# Memory model

Per spec §9, this document records the budget equation the planner and calibrator key
off, plus the provenance for the numbers behind it. Only Task 6 (the indexer) has landed
so far, so this file currently holds only the indexer gate below; the budget equation
itself lands with the planner (Task 9).

## Indexer gate (Task 6)

`featherquant index` (`featherquant/indexer.py`) reads checkpoint metadata only —
`config.json` / shard JSON headers for safetensors, the GGUF KV block for `.gguf` — and
never a weight byte (enforced by `tests/integration/test_index_no_weight_reads.py`, an
strace-gated test that is skipped on this machine — no `strace` binary and no root to
install one). Below are the real-model runs performed for the M1 gate, each with
`largest_tensor_bytes` verified by hand against `vocab_size × hidden_size × dtype_bytes`
from the checkpoint's own `config.json`.

### Qwen3-0.6B, safetensors (`~/models/qwen3-0.6b`)

```
$ .venv/bin/featherquant index ~/models/qwen3-0.6b -o /tmp/idx_qwen3_0.6b.json
311 tensors, 28 layers, largest tensor 296.8 MiB -> /tmp/idx_qwen3_0.6b.json
```

`config.json`: `vocab_size=151936`, `hidden_size=1024`, dtype BF16 (2 bytes).
By hand: `151936 × 1024 × 2 = 311,164,928 B = 296.75 MiB`.
`largest_tensor_bytes` in the emitted index: `311164928`. Match.

### Qwen3-14B, sharded safetensors (`~/models/qwen3-14b`, 8 shards)

```
$ .venv/bin/featherquant index ~/models/qwen3-14b -o /tmp/idx_qwen3_14b.json
443 tensors, 40 layers, largest tensor 1483.8 MiB -> /tmp/idx_qwen3_14b.json
```

`config.json`: `vocab_size=151936`, `hidden_size=5120`, dtype BF16 (2 bytes).
By hand: `151936 × 5120 × 2 = 1,555,824,640 B = 1483.75 MiB ≈ 1.449 GiB`.
`largest_tensor_bytes` in the emitted index: `1555824640`. Match.

### Qwen3-0.6B, GGUF (`~/models/qwen3-0.6b-bf16.gguf`)

```
$ .venv/bin/featherquant index ~/models/qwen3-0.6b-bf16.gguf -o /tmp/idx_gguf.json
311 tensors, 28 layers, largest tensor 296.8 MiB -> /tmp/idx_gguf.json
```

Same checkpoint as the first run, converted to GGUF: `vocab_size=151936`,
`hidden_size=1024`, BF16. Same hand computation: `311,164,928 B = 296.75 MiB`. Match, and
identical to the safetensors run of the same weights — the GGUF path's `ne`-order shape
reversal (`TensorInfo.shape` stored in source/HF row-major order) round-trips correctly.
`head_dims` from the GGUF KV block (`{"n_heads": 16, "n_kv_heads": 8, "head_dim": 128}`)
also matches `config.json`'s explicit `head_dim: 128` — the indexer reads the GGUF's
`{arch}.attention.key_length` KV rather than assuming `hidden_size / n_heads` (which would
give 64, wrong for this GQA architecture).

No `classify_hf`/`classify_gguf` failure occurred on any of the three runs above.

### Outstanding: a third, Llama-architecture family

Not run. No Llama-architecture checkpoint exists on this machine
(`model.layers.N.self_attn.*` naming with `gate_proj`/`up_proj` and no `q_norm`), and Task
6's instructions were explicit not to download one speculatively. `classify_hf`'s rule
table (`featherquant/roles.py`) already has Llama-shaped patterns (`gate_proj`, `up_proj`,
`down_proj`, `o_proj`) covered by the same rules Qwen3 uses, but that has not been
exercised end-to-end against a real Llama checkpoint's `config.json` key names (e.g.
whether it also sets `model_type`, `num_key_value_heads`, `head_dim`). This remains
outstanding until such a checkpoint is available.

## M2 gate: cgroup-enforced bit-identity (Task 7)

`tests/memory/test_rtn_under_ceiling.py::test_bit_identical_under_ceiling` runs the
featherquant RTN path against a real model wrapped in
`bench/harness/run_under_ceiling.sh` (`systemd-run --user`, cgroup v2 `MemoryMax`,
`MemorySwapMax=0`), then diffs the output against `llama-quantize`'s reference tensor-for-
tensor with `featherquant.validator.compare_gguf`. A test that passes without a ceiling is
not a memory test (spec §9); this one enforces the ceiling as a kernel-level constraint,
not just an accounting one — exit 137 (OOM-killed) fails the test.

```
$ .venv/bin/pytest tests/memory -v -m memory
tests/memory/test_rtn_under_ceiling.py::test_bit_identical_under_ceiling[q8_0-Q8_0] PASSED
tests/memory/test_rtn_under_ceiling.py::test_bit_identical_under_ceiling[q4_k_m-Q4_K_M] PASSED
2 passed in 143.96s (0:02:23)
```

Model: `~/models/qwen3-0.6b-bf16.gguf` (1.5 GB on disk). Ceiling: 1 GiB (`1G`), the value
in the brief — both formats passed on the first attempt, no OOM-kill, so no smaller
ceiling was searched for and 1G was not raised. `structural_check` and `compare_gguf` both
returned `[]` for both formats: featherquant's RTN output is bit-identical to
`llama-quantize`'s, under a real memory ceiling, for both Q8_0 and Q4_K_M.
