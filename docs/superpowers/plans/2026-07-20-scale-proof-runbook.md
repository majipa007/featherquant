# Scale Proof Runbook (10x source-to-RAM ratio)

> **For agentic workers:** This is an experiment runbook, not a code plan — no
> production code changes, so no TDD cycle. Execute steps in order, record
> every number. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the existing pipeline quantizes a source model ~10x larger than the enforced RAM ceiling, producing reference-identical output.

**Architecture:** No new code. Uses the shipped CLI, `scripts/memlimit_run.sh`, `scripts/baseline.sh`, `scripts/compare_reference.py`.

**Tech Stack:** featherquant (as committed), llama.cpp at `~/llama.cpp` (commit `25a1d63f4`, binaries in `build-cpu/bin`), systemd-run cgroup v2, uv-managed venvs.

## Global Constraints

- Model files live on ext4 (`/home/sukuna/models/`), never on /mnt/c (9p I/O skews results).
- Target ratio ≥ 8x: ~30 GB BF16 source vs 3 GB ceiling (or 16 GB vs 2 GB if disk is tight).
- Disk preflight: need source (~30 GB) + reference Q8_0 (~16 GB) + featherquant Q8_0 (~16 GB) ≈ 65 GB free.
- Record numbers in `docs/superpowers/plans/2026-07-20-scale-proof-results.md`; commit results as majipa007.

---

### Task 1: Acquire and convert the large model

- [ ] **Step 1: Disk preflight**

Run: `df -h /home/sukuna | tail -1`
Expected: ≥ 100 GB available. If < 100 GB, drop to Qwen3-8B (~16 GB BF16) and a 2 GB ceiling.

- [ ] **Step 2: Download Qwen3-14B**

```bash
mkdir -p ~/models/qwen3-14b && cd ~/models/qwen3-14b
for f in config.json tokenizer.json tokenizer_config.json generation_config.json \
         model.safetensors.index.json; do
  curl -sfLO "https://huggingface.co/Qwen/Qwen3-14B/resolve/main/$f"
done
# Shard list comes from the index; download each shard listed there:
python3 - <<'EOF'
import json, subprocess
shards = sorted(set(json.load(open("model.safetensors.index.json"))["weight_map"].values()))
for s in shards:
    subprocess.run(["curl", "-sfLO",
                    f"https://huggingface.co/Qwen/Qwen3-14B/resolve/main/{s}"], check=True)
    print("done", s)
EOF
```

Expected: shards totalling ~29.5 GB. Verify: `du -sh ~/models/qwen3-14b`.

- [ ] **Step 3: Convert to BF16 GGUF**

```bash
/home/sukuna/models/.convert-venv/bin/python /home/sukuna/llama.cpp/convert_hf_to_gguf.py \
  ~/models/qwen3-14b --outfile ~/models/qwen3-14b-bf16.gguf --outtype bf16
```

Expected: `Model successfully exported`, file ~29.5 GB. The converter reads lazily; if its own RSS exceeds workstation RAM, that is a finding to record, not a blocker (workstation is high-memory).

### Task 2: Conventional baseline

- [ ] **Step 1: Run llama-quantize with RSS capture**

```bash
cd "<repo>"   # featherQuant repo root
LLAMA_CPP_DIR=/home/sukuna/llama.cpp LLAMA_BIN=/home/sukuna/llama.cpp/build-cpu/bin \
  scripts/baseline.sh ~/models/qwen3-14b-bf16.gguf ~/models/ref14b_q8_0.gguf
```

Expected: peak RSS around the full model size (~30 GB). Record commit hash, `Maximum resident set size`, `Elapsed`, output sha256.

- [ ] **Step 2: Baseline under the same 3G ceiling (expected to FAIL)**

```bash
systemd-run --user --scope --collect -p MemoryMax=3G -p MemorySwapMax=0 \
  /home/sukuna/llama.cpp/build-cpu/bin/llama-quantize \
  ~/models/qwen3-14b-bf16.gguf /tmp/should-fail.gguf Q8_0; echo "exit=$?"
rm -f /tmp/should-fail.gguf
```

Expected: non-zero exit (137 = OOM-killed). This is the headline contrast: the conventional tool cannot complete where featherquant can. Record the exit code.

### Task 3: featherquant under the ceiling

- [ ] **Step 1: Run at 3G ceiling (~10x ratio)**

```bash
source .venv/bin/activate
scripts/memlimit_run.sh ~/models/qwen3-14b-bf16.gguf ~/models/fq14b_q8_0.gguf 3G
```

Expected: `PASS: completed inside 3G external ceiling`. Record full report JSON (`~/models/fq14b_q8_0.report.json`): peak_rss, working_budget, chunks, bytes read/written, elapsed_s, budget_violations (must be 0).

- [ ] **Step 2 (stretch): Tighten to 2G**

```bash
scripts/memlimit_run.sh ~/models/qwen3-14b-bf16.gguf ~/models/fq14b_2g.gguf 2G
```

Record PASS or the minimum-feasible-budget error verbatim — either result is a data point. Delete `fq14b_2g.gguf` after recording.

### Task 4: Validate and record

- [ ] **Step 1: Reference equivalence**

Run: `.venv/bin/python scripts/compare_reference.py ~/models/ref14b_q8_0.gguf ~/models/fq14b_q8_0.gguf`
Expected: `N tensors, 0 mismatches`, exit 0. Any mismatch: stop, extract the failing block, add a regression test to `tests/test_q8_0.py` (see Task 7 Step 5 of the MVP plan for the procedure).

- [ ] **Step 2: Inference smoke test**

```bash
/home/sukuna/llama.cpp/build-cpu/bin/llama-completion -m ~/models/fq14b_q8_0.gguf \
  -p "The capital of France is" -n 8 --seed 1 -c 512 -t 8 </dev/null
```

Expected: exit 0, coherent tokens. (Use `llama-completion`, not `llama-cli` — the chat REPL hangs headless; keep `-c 512` or the KV cache balloons.)

- [ ] **Step 3: Write results doc and commit**

Create `docs/superpowers/plans/2026-07-20-scale-proof-results.md` with: model, sizes, ratio, baseline RSS/time, baseline-under-ceiling exit code, featherquant report JSON, compare result, smoke result, disk peak. Then:

```bash
git add docs/superpowers/plans/2026-07-20-scale-proof-results.md
git commit -m "docs: 10x scale-proof results (Qwen3-14B under 3G ceiling)"
```
