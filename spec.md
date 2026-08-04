# FeatherQuant — Build Instruction Set

**Audience:** any coding agent or contributor working in this repository.
**Status:** authoritative. Where this file conflicts with a task prompt, ask before deviating.
**Scope:** bounded-memory post-training quantization of decoder-only transformers to GGUF.

---

## 1. The thesis

FeatherQuant makes **calibration-aware** quantization possible under a hard memory ceiling.

### 1.1 What is NOT the contribution

Streaming round-to-nearest quantization is **already solved** by existing tooling.
`convert_hf_to_gguf.py` loads safetensors lazily, tensor by tensor. `llama-quantize`
operates on bounded buffers. A 20 GB BF16 model already converts to `Q4_K_M` on a
low-RAM machine today.

Any claim of the form *"we quantized a 20 GB model in 2 GB of RAM"* that refers to RTN
K-quants is not a result. It is a reproduction of the baseline. Do not write it as a result.
Do not let it become the headline in the README, the paper, or any benchmark table.

### 1.2 What IS the contribution

Methods that need **statistics across the calibration set** are the ones that break under a
memory ceiling:

| Method | Memory blocker |
|---|---|
| imatrix (IQ-quants) | requires forward passes → model residency during inference |
| GPTQ | per-layer Hessian, `d_in × d_in` |
| AWQ | activation scales + grid search with layer held resident |

FeatherQuant's job is to make these run inside a declared budget, and to **measure the
quality cost of every approximation taken to get there**. The deliverable is a
quality-vs-memory frontier, not a single success flag.

### 1.3 The primary research question

> Given a memory ceiling **B**, what is the best achievable quantization quality, and how
> does that frontier compare to unconstrained calibration-aware quantization?

Every experiment must answer that question or be cut.

---

## 2. Non-negotiable invariants

1. **Declared budget is a hard ceiling.** The process runs inside a cgroup with
   `memory.max = B`. It either completes or is OOM-killed. There is no "approximately."
2. **Never measure with peak RSS alone.** RSS under `mmap` hides weight bytes in the page
   cache and any reviewer will say so. Cgroup enforcement is the measurement.
3. **No `mmap` for weight reads on any measured path.** Use explicit `pread` into owned
   buffers. `mmap` is permitted only in throwaway inspection scripts, never in
   `featherquant/` proper.
4. **Bit-exact determinism.** Same input + same config + same seed → byte-identical output
   file. This is a test, not an aspiration.
5. **Buffers are pre-allocated and reused.** No allocation inside a tensor-processing loop.
6. **Fail loudly.** If the plan cannot fit in `B`, refuse before doing any work and report
   the binding constraint. Never silently degrade a method, reduce sample count, or fall
   back to RTN.
7. **Every approximation is logged as a first-class field** in the run manifest, with the
   quality delta attributable to it.

---

## 3. Memory model

### 3.1 The budget equation

Peak memory during layer-wise calibrated quantization:

```
peak = layer_weights
     + activation_cache
     + hessian
     + output_buffer
     + runtime_overhead
```

The planner must compute this **before** processing and refuse if `peak > B`.

### 3.2 Component formulas

Let `h` = hidden size, `i` = FFN intermediate size, `v` = vocab size,
`n` = calibration samples, `s` = sequence length, and let all working math be fp32.

```
layer_weights  ≈ params_in_layer × bytes_per_source_element
activation_cache = n × s × h × bytes_per_activation_element
hessian        = d_in² × 4          # per linear layer, fp32, d_in = layer input dim
output_buffer  = one superblock row group, small and fixed
```

The Hessian is sized by the **largest `d_in` in the layer**, which is the `down_proj`
input, i.e. `i`, not `h`. This is the single most commonly mis-sized term.

### 3.3 Worked example

Target: `h = 4096`, `i = 12288`, `v = 151936`, BF16 source (~8B params, ~16 GB on disk).
Budget `B = 2 GiB`. Calibration `n = 128`, `s = 512`, activations fp16.

```
layer_weights     ~191M params × 2 B          =  382 MB
activation_cache  128 × 512 × 4096 × 2 B      =  537 MB
hessian (down)    12288² × 4 B                =  604 MB
output + overhead                             ~  100 MB
                                              ----------
peak                                          ~ 1.62 GB
```

Fits — but with under 400 MB of headroom, and **only** if the layer is not upcast to fp32
wholesale. Upcasting the full layer costs another 382 MB and breaks the budget. Upcast
per-column-block inside the fixed buffer instead.

Two consequences to internalise:

- **The activation cache is a first-class budget consumer, not an afterthought.** It is
  also the most tunable term. `n`, `s`, and spill-to-disk are the primary knobs, and each
  has a quality cost that must be measured, not assumed.
- **The embedding tensor sets a hard floor.** `151936 × 4096` in BF16 is 1.24 GB in a
  single tensor. It must be processed in row groups. The planner must know the largest
  single tensor before it commits to a budget, and must report it in the refusal message
  when the budget is infeasible.

### 3.4 Derive, never hardcode

All dimensions come from `config.json` and the safetensors index. Any hardcoded shape,
layer count, or vocab size is a bug. Model families change dimensions between releases.

---

## 4. Architecture

Eight modules. Each has one job and a serialisable contract at its boundary, so any stage
can be tested and replayed in isolation.

### 4.1 `indexer`

Reads `config.json` + `model.safetensors.index.json` + shard headers only. Never reads
weight bytes.

Emits `manifest.json`:

```
model_arch, n_layers, hidden_size, intermediate_size, vocab_size, head_dims
tensors: [{ name, shape, dtype, shard_path, byte_offset, byte_length,
            quant_eligible, layer_index, role }]
largest_tensor_bytes, total_bytes
```

`role` is one of `embed | attn_q | attn_k | attn_v | attn_o | ffn_gate | ffn_up |
ffn_down | norm | output`. Downstream logic keys off `role`, never off name-string
matching. Name conventions differ across model families; keep that knowledge in one place.

### 4.2 `planner`

Input: `manifest.json`, budget `B`, method, calibration config.
Output: `plan.json` — per-tensor slicing strategy, processing order, buffer sizes,
computed peak, and the binding constraint.

Must refuse infeasible plans with a specific, actionable message:

```
INFEASIBLE: budget 1.0 GiB < required 1.62 GiB
  binding term: hessian (604 MB, d_in=12288)
  options: --hessian-approx=diagonal (-598 MB, measured PPL cost +0.31)
           --calib-samples=64        (-268 MB, measured PPL cost +0.12)
```

Never guess the quality cost. Populate those numbers from the calibrated table in
`docs/approximation_costs.md`, and mark them `UNMEASURED` until they exist.

### 4.3 `reader`

Slice-level `pread` into caller-owned buffers. Signature takes a destination buffer; the
reader never allocates. Handles BF16/FP16 → fp32 conversion per block, inside the buffer.

### 4.4 `calibrator`

Sequential layer-wise calibration. This is the core of the project.

```
1. Run embeddings over calibration batch → activation_cache
2. For each layer L:
     a. Load L's weights into the fixed layer buffer
     b. Accumulate statistics for L from activation_cache
     c. Quantize L (delegates to quantizer)
     d. Forward activation_cache through the *quantized* L, in place
     e. Release L
3. activation_cache now holds inputs for L+1
```

Step **(d) must use the quantized layer, not the original.** Propagating quantization
error forward is what makes sequential calibration work. Using original weights is a
silent correctness bug that shows up only as mysteriously poor output quality.

The activation cache is a ring of fixed buffers with optional disk spill. Spilling is a
logged approximation with a measured runtime cost, not a hidden fallback.

### 4.5 `quantizer`

Fixed-buffer block quantization. Superblock formats:

| Format | Weights/superblock | Bytes | bpw |
|---|---|---|---|
| `Q8_0` | 32 | 34 | 8.5 |
| `Q4_K` | 256 | 144 | 4.5 |
| `Q6_K` | 256 | 210 | 6.5625 |

Verify these against upstream `ggml-quants.h` at implementation time; treat the table as a
starting point, not gospect. Row groups must align to superblock boundaries — 256 for
K-quants. A row-group size that isn't a multiple of 256 will produce a file that loads and
emits garbage.

### 4.6 `writer`

Incremental GGUF writer. Default alignment 32 bytes (`general.alignment`). Tracks offsets,
sizes, per-tensor checksums, and completion state. Writes to a temp path and atomically
renames on success so a partial file can never be mistaken for a finished one.

### 4.7 `checkpoint`

Tensor-granular. On resume: verify checksums of completed tensors, detect and truncate
partial writes, continue. Also persists calibration state — resuming at layer 40 requires
the layer-40 activation cache, which is the expensive part of checkpointing and must be
sized into the budget when enabled.

### 4.8 `validator`

- Structural: tensor count, names, shapes, offsets, checksums, alignment
- Loadability: model loads in `llama.cpp` and produces coherent output
- Numerical: per-tensor error vs unconstrained reference quantization
- Determinism: two runs → identical bytes

---

## 5. The approximation ladder

The research core. Each rung trades memory for quality. **Every rung must be measured, and
the cost table is the primary deliverable.**

**Hessian handling**, most to least memory:

1. Full fp32 in-memory — reference, no approximation
2. Blocked out-of-core — full fidelity, disk-backed, panel Cholesky
3. Low-rank + diagonal — rank `r` correction
4. Diagonal only — degenerates toward scaled RTN

**Calibration set:** samples `n`, sequence length `s`, spill vs resident.

**Working precision:** fp32 vs bf16 accumulation for statistics.

Produce `docs/approximation_costs.md`:

| Rung | Peak Δ | Runtime Δ | PPL Δ | Downstream task Δ |
|---|---|---|---|---|

The claim to establish or refute: **blocked out-of-core Hessian gives full-fidelity GPTQ
quality at a runtime cost, inside a budget where the in-memory method cannot run at all.**
A negative result here is still publishable and still worth having. Do not tune the
experiment until it agrees with you.

---

## 6. Measurement protocol

Every run emits a manifest. No number enters a table or the README without one.

```json
{
  "run_id": "...", "date": "DD/MM/YYYY",
  "model": { "id": "...", "revision": "...", "sha256": "..." },
  "method": "...", "approximations": [...],
  "budget_bytes": 2147483648,
  "enforcement": "cgroup_v2_memory_max",
  "peak_observed_bytes": ..., "oom_killed": false,
  "runtime_seconds": ..., "bytes_read": ..., "bytes_written": ...,
  "storage": "nvme|sata_ssd|hdd",
  "output_sha256": "...",
  "quality": { "ppl": ..., "ppl_dataset": "...", "tasks": {...} },
  "host": { "cpu": "...", "ram_gb": ..., "kernel": "..." }
}
```

Rules:

- Ceilings in binary units (GiB), file sizes in whatever the tool reports, labelled.
- Perplexity is only comparable within one dataset, context length, and tokenizer. State
  all three or the number is meaningless.
- Report runtime **and** the ceiling together, always. Memory reduction with unbounded
  runtime is not a result.
- Cold vs warm page cache changes I/O numbers substantially. Drop caches between runs and
  say so.

### 6.1 Baselines to reproduce first

Before any FeatherQuant number is reported, reproduce and record:

1. `convert_hf_to_gguf.py` + `llama-quantize` RTN, unconstrained
2. Same, under cgroup ceiling `B` — find where it actually breaks, if it does
3. `llama-imatrix` + IQ-quant, unconstrained
4. Reference GPTQ, unconstrained

Baselines 1 and 2 will likely succeed. **Report that honestly.** It defines the boundary of
the contribution and is the difference between a credible project and one a reviewer
dismisses in the first paragraph.

---

## 7. Build sequence

Each milestone has a falsifiable exit gate. Do not start the next until the gate passes.

**M0 — Baselines.** Reproduce all four above. Gate: four run manifests committed, with the
cgroup ceiling at which each baseline breaks documented.

**M1 — Indexer.** Gate: manifest for three model families (different naming conventions),
`largest_tensor_bytes` correct, zero weight bytes read (verify with `strace` byte counts).

**M2 — Reader + writer + RTN.** Gate: RTN `Q4_K_M` output is bit-identical to
`llama-quantize` on the same input, under a cgroup ceiling. Bit-identical, not "close."

**M3 — Planner.** Gate: predicted peak within 10% of observed on ten model/budget pairs;
every infeasible case refuses before allocating, naming the binding term.

**M4 — Calibrator, in-memory Hessian.** Gate: matches reference GPTQ perplexity within
noise on a small model. Explicitly test that quantized-layer propagation (§4.4d) is active
— a unit test asserting the activation cache differs from the fp32-path result.

**M5 — Out-of-core Hessian.** Gate: same quality as M4, inside a budget where M4 is
OOM-killed. **This is the project's central claim.**

**M6 — Approximation ladder.** Gate: `docs/approximation_costs.md` fully populated, no
`UNMEASURED` rows.

**M7 — Checkpoint/resume.** Gate: `SIGKILL` at 100 random points; every resume produces
output bit-identical to the uninterrupted run.

**M8 — Scale.** Gate: frontier measured across budget × model-size ratios × storage tiers.

---

## 8. CLI surface

```
featherquant index   <model_path> -o manifest.json
featherquant plan    manifest.json --budget 2GiB --method gptq \
                     --calib-samples 128 --calib-seqlen 512 -o plan.json
featherquant run     plan.json -o model-Q4_K_M.gguf [--resume]
featherquant verify  model-Q4_K_M.gguf --reference ref.gguf
featherquant bench   --sweep budgets.yaml
```

Split `plan` from `run` deliberately: the plan is inspectable, diffable, and committable
before hours of compute are spent. Flags use `--kebab-case`; Python identifiers use
`snake_case`; artifacts use `underscore_naming`.

---

## 9. Repository layout

```
featherquant/
  indexer/  planner/  reader/  calibrator/  quantizer/
  writer/   checkpoint/  validator/  cli/
tests/
  unit/  integration/  determinism/  memory/
bench/
  harness/  sweeps/  manifests/
docs/
  approximation_costs.md
  memory_model.md
  baselines.md
```

`tests/memory/` runs under cgroup enforcement in CI. A test that passes without a ceiling
is not a memory test.

---

## 10. Forbidden patterns

- `mmap` on any measured path
- Allocation inside a tensor or block loop
- Silent method degradation on memory pressure
- Peak-RSS-only measurement
- Reporting a memory number without its runtime
- Row groups not aligned to superblock size
- Forward-propagating original weights instead of quantized ones during calibration
- Hardcoded model dimensions
- `try/except` that swallows a memory error and continues
- Any benchmark row whose output was never checked for coherent generation

---

## 11. Agent working rules

1. **Read `docs/memory_model.md` before touching the planner or calibrator.** If your
   change alters the budget equation, update that doc in the same commit.
2. **Numbers require provenance.** Never write a benchmark figure you did not produce from
   a committed run manifest. `UNMEASURED` is an acceptable value; a plausible guess is not.
3. **Verify format constants against upstream source at implementation time.** Block sizes
   and byte layouts in this document are a starting point and may drift.
4. **Confirm a quantized output actually generates coherent text before it enters a table.**
   A file that loads is not a file that works. If a format name is not in upstream
   `ggml`'s type enum, it does not exist — do not benchmark it.
5. **When a result contradicts the thesis, report the result.** Do not adjust the
   experiment until it agrees. The frontier is the deliverable; where it sits is an
   empirical question.
6. **Ask before changing anything in §2.** Those invariants are what make the project's
   claims defensible. Everything else is negotiable.
