# FeatherQuant spec.md M0–M8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the existing RTN streaming quantizer to the full `spec.md` deliverable — calibration-aware (GPTQ) quantization that runs inside a hard cgroup memory ceiling, with a measured quality-vs-memory frontier as the primary artifact.

**Architecture:** New modules land flat in `featherquant/` alongside the existing ones (no restructuring of working code). Eight spec roles map onto files: `indexer.py`, `planner.py`, existing `gguf_io.py`/`st_source.py` (reader), new `calibrator.py` + `activations.py` + `hessian.py` + `gptq.py` + `model_fwd.py`, existing `q8_0.py`/`ggml_backend.py` (quantizer), existing `gguf_io.py` (writer), existing `manifest.py` (checkpoint), new `validator.py`. Measurement lives in `run_manifest.py` + `bench.py`. The CLI grows spec §8 subcommands while keeping today's flat-flag form working for `featherquant.sh`.

**Tech Stack:** Python ≥3.10, numpy, gguf (metadata + `gguf.quants.dequantize`), rich, pyyaml (new, sweep files); ctypes into `libggml-base.so` for K-quant kernels; cgroup v2 via `systemd-run --user`; llama.cpp CLI tools for baselines and cross-checks; pytest/ruff/mypy as today.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from `spec.md`.

- **Declared budget is a hard ceiling.** Runs are enforced with cgroup v2 `memory.max = B`; the job completes or is OOM-killed. `enforcement` in every run manifest is `"cgroup_v2_memory_max"`.
- **Never measure with peak RSS alone.** RSS is telemetry; the cgroup is the measurement.
- **No `mmap` for weight reads on any measured path.** Explicit `pread`/`seek`+`readinto` into owned buffers. `mmap` permitted only in throwaway inspection scripts, never in `featherquant/` proper. (`GGUFReader` is metadata-only and its KV graph is released via `release_metadata()` before streaming — that stays the only exception, and it never touches tensor-data pages.)
- **Bit-exact determinism.** Same input + same config + same seed → byte-identical output file. Tested, not asserted.
- **Buffers are pre-allocated and reused.** No allocation inside a tensor-processing loop.
- **Fail loudly.** If the plan cannot fit in `B`, refuse before doing any work and report the binding constraint. Never silently degrade a method, reduce sample count, or fall back to RTN.
- **Every approximation is logged as a first-class field** in the run manifest (`approximations: [...]`), with the quality delta attributable to it.
- **Derive, never hardcode.** All dimensions come from `config.json` and the safetensors index. Any hardcoded shape, layer count, or vocab size is a bug.
- **Forbidden patterns** (spec §10): `mmap` on a measured path; allocation inside a tensor or block loop; silent method degradation; peak-RSS-only measurement; a memory number without its runtime; row groups not aligned to superblock size; forward-propagating original weights instead of quantized ones during calibration; hardcoded model dimensions; `try/except` that swallows a memory error and continues; a benchmark row whose output was never checked for coherent generation.
- **Format constants** (verify against upstream `ggml-quants.h` at implementation time — `GGML_QUANT_SIZES` from the `gguf` package is the runtime source of truth): `Q8_0` 32 weights / 34 bytes / 8.5 bpw; `Q4_K` 256 / 144 / 4.5; `Q6_K` 256 / 210 / 6.5625. K-quant superblock `QK_K = 256`; row groups must align to it.
- **GGUF alignment:** 32 bytes (`general.alignment`), as `featherquant/gguf_io.py:ALIGN`.
- **Roles** are exactly `embed | attn_q | attn_k | attn_v | attn_o | ffn_gate | ffn_up | ffn_down | norm | output`. Downstream logic keys off `role`, never off name-string matching.
- **CLI:** flags `--kebab-case`, Python identifiers `snake_case`, artifacts `underscore_naming`. Subcommands per spec §8: `index`, `plan`, `run`, `verify`, `bench`.
- **Units and dates:** memory ceilings in binary units (GiB); dates in run manifests are `DD/MM/YYYY`; perplexity is reported only with its dataset, context length, and tokenizer.
- **Numbers require provenance.** Never write a benchmark figure not produced from a committed run manifest. `UNMEASURED` is an acceptable value; a plausible guess is not.
- **Style:** PEP 8, ruff `line-length = 100` with `select = ["E", "W", "F", "I"]`, mypy `disallow_untyped_defs = true` on `featherquant/`. Docstrings and comments everywhere; wrap I/O and parsing in `try/except` with actionable messages (never around memory errors).
- **Tooling:** `uv` only (`uv venv`, `uv pip install -e '.[dev]'`). Gates for every task: `.venv/bin/pytest`, `.venv/bin/ruff check featherquant tests scripts`, `.venv/bin/mypy featherquant`.
- **Commits:** author `majipa007 <sulavstha007@gmail.com>`, **no** `Co-Authored-By` line, commit at the end of every task.
- **Local paths on the dev box** (do not hardcode into library code — CLI flags or env vars only): HF checkpoints `~/models/qwen3-0.6b`, `~/models/qwen3-14b`; BF16 GGUFs `~/models/qwen3-0.6b-bf16.gguf`, `~/models/qwen3-14b-bf16.gguf`; llama.cpp at `~/llama.cpp` with CPU build in `~/llama.cpp/build-cpu` (`libggml-base.so` under `build-cpu/bin`).

## File Structure

New files (all additive; nothing existing moves):

```
featherquant/
  indexer.py        — model index: config.json + safetensors index + shard headers -> ModelIndex (M1)
  roles.py          — HF/GGUF tensor name -> Role, one place for naming knowledge (M1)
  run_manifest.py   — spec §6 measurement manifest: schema, host capture, atomic write (M0)
  approx_costs.py   — parse docs/approximation_costs.md into a lookup; UNMEASURED-aware (M3)
  planner.py        — budget equation, Plan, INFEASIBLE refusal with binding term (M3)
  activations.py    — fixed-ring activation cache with optional disk spill (M4)
  model_fwd.py      — numpy Qwen3 decoder-layer forward: RMSNorm, RoPE, GQA, SwiGLU (M4)
  hessian.py        — InMemoryHessian + TiledHessian (blocked out-of-core, panel Cholesky) (M4/M5)
  gptq.py           — group-wise GPTQ error compensation over ggml grids (M4)
  calibrator.py     — sequential layer-wise loop, quantized-forward propagation (M4)
  validator.py      — structural / loadability / numerical / determinism checks (M2+)
  bench.py          — sweep runner over budgets.yaml -> run manifests -> frontier table (M8)
tests/
  unit/  integration/  determinism/  memory/     (each with __init__.py)
bench/
  harness/run_baseline.sh, harness/run_under_ceiling.sh
  sweeps/budgets.yaml
  manifests/                                     (committed run manifests)
docs/
  baselines.md, memory_model.md, approximation_costs.md
scripts/
  plan_accuracy.py, coherence_check.sh
```

Existing files modified: `featherquant/cli.py` (subcommands), `featherquant/engine.py` (accept a `Plan`, emit a run manifest), `pyproject.toml` (pyyaml), `README.md` (contribution framing per spec §1.1).

**Naming collision to respect:** `featherquant/manifest.py` is the *checkpoint/resume* manifest and keeps that meaning. The *model index* artifact is `ModelIndex` in `indexer.py` (the CLI still writes it to a user-chosen path, e.g. `manifest.json`, per spec §8). The *measurement* manifest is `RunManifest` in `run_manifest.py`.

---

## Milestone M0 — Baselines

**Gate:** four run manifests committed under `bench/manifests/`, with the cgroup ceiling at which each baseline breaks documented in `docs/baselines.md`.

### Task 1: Run manifest module

**Files:**
- Create: `featherquant/run_manifest.py`
- Test: `tests/unit/test_run_manifest.py`, `tests/unit/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunManifest` dataclass; `RunManifest.new(run_id: str, model: dict[str, str], method: str, budget_bytes: int, storage: str) -> RunManifest`; `RunManifest.save(path: str) -> None`; `RunManifest.load(path: str) -> RunManifest`; `host_info() -> dict[str, object]`; `today_ddmmyyyy() -> str`; `sha256_file(path: str) -> str`. Fields exactly as spec §6: `run_id, date, model{id,revision,sha256}, method, approximations, budget_bytes, enforcement, peak_observed_bytes, oom_killed, runtime_seconds, bytes_read, bytes_written, storage, output_sha256, quality{ppl,ppl_dataset,tasks}, host{cpu,ram_gb,kernel}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_run_manifest.py
import json

from featherquant.run_manifest import RunManifest, host_info, today_ddmmyyyy


def test_new_manifest_has_spec_shape(tmp_path):
    m = RunManifest.new(run_id="m0_rtn_unconstrained",
                        model={"id": "Qwen/Qwen3-0.6B", "revision": "main",
                               "sha256": "0" * 64},
                        method="rtn_q8_0", budget_bytes=2147483648,
                        storage="nvme")
    assert m.enforcement == "cgroup_v2_memory_max"
    assert m.oom_killed is False
    assert m.quality == {"ppl": None, "ppl_dataset": None, "tasks": {}}
    p = tmp_path / "run.json"
    m.save(str(p))
    d = json.loads(p.read_text())
    assert set(d) == {"run_id", "date", "model", "method", "approximations",
                      "budget_bytes", "enforcement", "peak_observed_bytes",
                      "oom_killed", "runtime_seconds", "bytes_read",
                      "bytes_written", "storage", "output_sha256", "quality",
                      "host"}
    assert d["host"]["kernel"] and d["host"]["ram_gb"] > 0


def test_date_is_singapore_format():
    s = today_ddmmyyyy()
    dd, mm, yyyy = s.split("/")
    assert len(dd) == 2 and len(mm) == 2 and len(yyyy) == 4


def test_roundtrip(tmp_path):
    m = RunManifest.new("r", {"id": "x", "revision": "y", "sha256": "z"},
                        "rtn_q8_0", 1 << 30, "nvme")
    m.runtime_seconds = 12.5
    p = tmp_path / "r.json"
    m.save(str(p))
    assert RunManifest.load(str(p)).runtime_seconds == 12.5


def test_host_info_fields():
    h = host_info()
    assert set(h) == {"cpu", "ram_gb", "kernel"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_run_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.run_manifest'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/run_manifest.py
"""Measurement manifest (spec §6): one JSON per measured run.

No number enters a table, a doc, or the README without one of these. The
schema is fixed by the spec — fields are never dropped, only filled in.
"""
import datetime
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Any


def today_ddmmyyyy() -> str:
    """Today's date in Singapore format (DD/MM/YYYY)."""
    return datetime.date.today().strftime("%d/%m/%Y")


def sha256_file(path: str) -> str:
    """Stream a file's sha256 in 8 MiB chunks (never materializes it)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8 << 20):
                h.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot hash {path}: {exc}") from exc
    return h.hexdigest()


def host_info() -> dict[str, Any]:
    """CPU model, total RAM in GiB, kernel release."""
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass  # non-Linux or restricted /proc: keep platform.processor()
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                       / (1 << 30), 2)
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"cannot read physical memory size: {exc}") from exc
    return {"cpu": cpu, "ram_gb": ram_gb, "kernel": platform.release()}


@dataclass
class RunManifest:
    """Spec §6 run record. Field order matches the spec listing."""
    run_id: str
    date: str
    model: dict[str, str]
    method: str
    approximations: list[dict[str, Any]]
    budget_bytes: int
    enforcement: str
    peak_observed_bytes: int | None
    oom_killed: bool
    runtime_seconds: float | None
    bytes_read: int | None
    bytes_written: int | None
    storage: str
    output_sha256: str | None
    quality: dict[str, Any]
    host: dict[str, Any] = field(default_factory=host_info)

    @classmethod
    def new(cls, run_id: str, model: dict[str, str], method: str,
            budget_bytes: int, storage: str) -> "RunManifest":
        """A manifest with everything measurable still unfilled."""
        return cls(run_id=run_id, date=today_ddmmyyyy(), model=dict(model),
                   method=method, approximations=[],
                   budget_bytes=budget_bytes,
                   enforcement="cgroup_v2_memory_max",
                   peak_observed_bytes=None, oom_killed=False,
                   runtime_seconds=None, bytes_read=None, bytes_written=None,
                   storage=storage, output_sha256=None,
                   quality={"ppl": None, "ppl_dataset": None, "tasks": {}})

    def save(self, path: str) -> None:
        """Write the manifest as pretty JSON (atomic: tmp + replace)."""
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(asdict(self), f, indent=2, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            raise RuntimeError(f"cannot save run manifest {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> "RunManifest":
        """Load a manifest, failing loudly on a schema mismatch."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load run manifest {path}: {exc}") from exc
        try:
            return cls(**d)
        except TypeError as exc:
            raise RuntimeError(f"malformed run manifest {path}: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_run_manifest.py -v && .venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant`
Expected: 4 passed, ruff clean, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add featherquant/run_manifest.py tests/unit/__init__.py tests/unit/test_run_manifest.py
git commit -m "feat: spec §6 run manifest with host capture and DD/MM/YYYY dates"
```

---

### Task 2: Baseline harness — RTN, unconstrained and under a ceiling

**Files:**
- Create: `bench/harness/run_baseline.sh`, `bench/harness/run_under_ceiling.sh`, `bench/manifests/.gitkeep`
- Create: `docs/baselines.md`
- Test: `tests/integration/test_baseline_harness.py`, `tests/integration/__init__.py`

**Interfaces:**
- Consumes: `RunManifest` (Task 1).
- Produces: `bench/harness/run_baseline.sh CMD_JSON RUN_ID METHOD MODEL_ID OUTPUT [STORAGE]` — runs an arbitrary command with `/usr/bin/time -v`, emits `bench/manifests/<run_id>.json`; `bench/harness/run_under_ceiling.sh LIMIT CMD...` — wraps a command in `systemd-run --user --scope -p MemoryMax=LIMIT -p MemorySwapMax=0`, exits 137 on OOM-kill. Both are consumed by Tasks 3, 4, and the M8 sweep.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_baseline_harness.py
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path("bench/harness/run_baseline.sh")


@pytest.mark.skipif(not shutil.which("/usr/bin/time"), reason="GNU time absent")
def test_harness_emits_run_manifest(tmp_path):
    out = tmp_path / "artifact.bin"
    cmd = json.dumps(["sh", "-c", f"printf hello > {out}"])
    r = subprocess.run(
        ["bash", str(HARNESS), cmd, "t_harness", "noop", "test/model",
         str(out), "nvme"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "FQ_MANIFEST_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "t_harness.json").read_text())
    assert m["method"] == "noop"
    assert m["runtime_seconds"] >= 0
    assert m["peak_observed_bytes"] > 0
    assert m["oom_killed"] is False
    assert len(m["output_sha256"]) == 64


def test_ceiling_wrapper_reports_oom_exit_code(tmp_path):
    if not shutil.which("systemd-run"):
        pytest.skip("systemd-run absent")
    r = subprocess.run(
        ["bash", "bench/harness/run_under_ceiling.sh", "32M",
         "python", "-c", "b=bytearray(512<<20)"],
        capture_output=True, text=True)
    assert r.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_baseline_harness.py -v`
Expected: FAIL — harness scripts do not exist (`bash: bench/harness/run_baseline.sh: No such file or directory`).

- [ ] **Step 3: Write the harness scripts**

```bash
#!/usr/bin/env bash
# bench/harness/run_baseline.sh — run any command, emit a spec §6 run manifest.
#
# Usage: run_baseline.sh CMD_JSON RUN_ID METHOD MODEL_ID OUTPUT [STORAGE] [BUDGET_BYTES]
#   CMD_JSON  JSON array of argv, e.g. '["llama-quantize","in.gguf","out.gguf","Q8_0"]'
#   OUTPUT    artifact whose sha256 goes into the manifest
#   STORAGE   nvme|sata_ssd|hdd   (default nvme)
#   BUDGET_BYTES  declared ceiling; 0 = unconstrained baseline
# Manifests land in $FQ_MANIFEST_DIR (default bench/manifests).
set -uo pipefail
CMD_JSON=$1; RUN_ID=$2; METHOD=$3; MODEL_ID=$4; OUTPUT=$5
STORAGE=${6:-nvme}; BUDGET=${7:-0}
DIR=${FQ_MANIFEST_DIR:-bench/manifests}
mkdir -p "$DIR"
TIMEFILE=$(mktemp)

# Cold page cache makes I/O numbers honest; needs root, so it is optional
# and recorded either way.
if [ "${FQ_DROP_CACHES:-0}" = "1" ]; then
  sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
fi

python - "$CMD_JSON" <<'PY' > /tmp/fq_argv.$$
import json, shlex, sys
print(" ".join(shlex.quote(a) for a in json.loads(sys.argv[1])))
PY
ARGV=$(cat /tmp/fq_argv.$$); rm -f /tmp/fq_argv.$$

# -v gives "Maximum resident set size (kbytes)" and "Elapsed (wall clock) time".
/usr/bin/time -v sh -c "$ARGV" 2> "$TIMEFILE"
CODE=$?

python - "$TIMEFILE" "$RUN_ID" "$METHOD" "$MODEL_ID" "$OUTPUT" "$STORAGE" \
        "$BUDGET" "$CODE" "$DIR" <<'PY'
"""Parse GNU time output into a RunManifest and save it."""
import re
import sys

from featherquant.run_manifest import RunManifest, sha256_file

timefile, run_id, method, model_id, output, storage, budget, code, dirname = sys.argv[1:10]
text = open(timefile).read()
peak_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)[1])
wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", text)[1]
parts = [float(p) for p in wall.split(":")]
seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

m = RunManifest.new(run_id, {"id": model_id, "revision": "unknown",
                             "sha256": "unknown"},
                    method, int(budget), storage)
m.enforcement = "cgroup_v2_memory_max" if int(budget) else "none_unconstrained"
m.peak_observed_bytes = peak_kb * 1024
m.runtime_seconds = round(seconds, 3)
# 137 = SIGKILL, which under a cgroup ceiling means the OOM killer fired.
m.oom_killed = int(code) == 137
try:
    m.output_sha256 = sha256_file(output)
except RuntimeError:
    m.output_sha256 = None      # killed before the artifact existed
m.save(f"{dirname}/{run_id}.json")
print(f"wrote {dirname}/{run_id}.json (exit={code}, peak={peak_kb//1024} MiB)")
PY
rm -f "$TIMEFILE"
exit "$CODE"
```

```bash
#!/usr/bin/env bash
# bench/harness/run_under_ceiling.sh — run a command inside a cgroup v2 ceiling.
#
# Usage: run_under_ceiling.sh LIMIT CMD...
# Swap is disabled inside the scope so the ceiling is honest. Exit 137 means
# the kernel OOM-killed the job — that is a measurement, not a bug.
set -uo pipefail
LIMIT=$1; shift
exec systemd-run --user --scope --collect --same-dir \
  -p MemoryMax="$LIMIT" -p MemorySwapMax=0 "$@"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chmod +x bench/harness/*.sh && .venv/bin/pytest tests/integration/test_baseline_harness.py -v`
Expected: PASS (the ceiling test skips if `systemd-run` is unavailable).

- [ ] **Step 5: Record baseline 1 — unconstrained RTN**

Run (records `bench/manifests/m0_rtn_unconstrained.json`):

```bash
LC=~/llama.cpp/build-cpu/bin
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-quantize\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_ref_q4_k_m.gguf\",\"Q4_K_M\"]" \
  m0_rtn_unconstrained rtn_q4_k_m Qwen/Qwen3-14B /tmp/m0_ref_q4_k_m.gguf nvme 0
```

Expected: exit 0, manifest written, `peak_observed_bytes` recorded.

- [ ] **Step 6: Record baseline 2 — RTN under descending ceilings**

Run each ceiling until one is OOM-killed; every run gets its own manifest:

```bash
for L in 8G 4G 2G 1G 512M; do
  bash bench/harness/run_baseline.sh \
    "[\"bash\",\"bench/harness/run_under_ceiling.sh\",\"$L\",\"$LC/llama-quantize\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_ceil_$L.gguf\",\"Q4_K_M\"]" \
    "m0_rtn_ceiling_$L" rtn_q4_k_m Qwen/Qwen3-14B "/tmp/m0_ceil_$L.gguf" nvme \
    "$(python -c "import sys;print({'8G':8,'4G':4,'2G':2,'1G':1,'512M':0.5}['$L']*2**30)")"
done
```

Expected: the lowest passing ceiling and the first OOM-killed ceiling are both in `bench/manifests/`.

- [ ] **Step 7: Write `docs/baselines.md`**

Record, from the committed manifests only (no remembered numbers): a table of `run_id | method | ceiling | peak_observed | runtime_s | oom_killed | output_sha256`, and a prose paragraph stating plainly — per spec §6.1 — that RTN K-quant conversion **succeeds** under low ceilings and is therefore *not* a FeatherQuant result. Include the exact llama.cpp revision (`git -C ~/llama.cpp rev-parse HEAD`) and note whether `FQ_DROP_CACHES=1` was used.

- [ ] **Step 8: Commit**

```bash
git add bench/harness bench/manifests docs/baselines.md tests/integration
git commit -m "feat: baseline harness emitting run manifests; RTN baselines 1-2 recorded"
```

---

### Task 3: Baseline 3 — imatrix + IQ-quant, unconstrained

**Files:**
- Modify: `docs/baselines.md`
- Create: `bench/manifests/m0_imatrix_unconstrained.json`, `bench/manifests/m0_iq4_xs_unconstrained.json` (generated)

**Interfaces:**
- Consumes: `bench/harness/run_baseline.sh` (Task 2).
- Produces: two committed manifests plus the calibration-corpus provenance line reused by every later perplexity number.

- [ ] **Step 1: Fetch and pin the calibration corpus**

```bash
mkdir -p bench/data
curl -L -o bench/data/wiki.test.raw \
  https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw/wiki.test.raw
sha256sum bench/data/wiki.test.raw | tee bench/data/wiki.test.raw.sha256
```

Add `bench/data/wiki.test.raw` to `.gitignore` (large); commit only the `.sha256`.

- [ ] **Step 2: Run the imatrix pass**

```bash
LC=~/llama.cpp/build-cpu/bin
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-imatrix\",\"-m\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"-f\",\"bench/data/wiki.test.raw\",\"-o\",\"/tmp/m0_qwen3_14b.imatrix\",\"--chunks\",\"128\"]" \
  m0_imatrix_unconstrained imatrix Qwen/Qwen3-14B /tmp/m0_qwen3_14b.imatrix nvme 0
```

Expected: exit 0; `peak_observed_bytes` here is the number that shows why imatrix needs model residency (spec §1.2).

- [ ] **Step 3: Run the IQ-quant using that imatrix**

```bash
bash bench/harness/run_baseline.sh \
  "[\"$LC/llama-quantize\",\"--imatrix\",\"/tmp/m0_qwen3_14b.imatrix\",\"$HOME/models/qwen3-14b-bf16.gguf\",\"/tmp/m0_iq4_xs.gguf\",\"IQ4_XS\"]" \
  m0_iq4_xs_unconstrained iq4_xs Qwen/Qwen3-14B /tmp/m0_iq4_xs.gguf nvme 0
```

- [ ] **Step 4: Measure perplexity and fill `quality`**

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

- [ ] **Step 5: Confirm coherent generation before the number counts**

```bash
$LC/llama-cli -m /tmp/m0_iq4_xs.gguf -p "The capital of Singapore is" -n 24 --temp 0
```

Expected: coherent continuation. Paste the output into `docs/baselines.md`. Per spec §11.4, a file that loads is not a file that works.

- [ ] **Step 6: Commit**

```bash
git add bench/manifests docs/baselines.md bench/data/wiki.test.raw.sha256 .gitignore
git commit -m "feat: imatrix + IQ4_XS baselines with ppl provenance and coherence check"
```

---

### Task 4: Baseline 4 — reference GPTQ, unconstrained

**Files:**
- Create: `bench/harness/run_gptq_reference.py`
- Modify: `docs/baselines.md`

**Interfaces:**
- Consumes: `RunManifest` (Task 1).
- Produces: `bench/manifests/m0_gptq_reference.json` with `quality.ppl`, and a saved per-layer reference: `bench/manifests/m0_gptq_reference_layer_errors.json` mapping `"<layer>.<role>" -> float` (mean squared reconstruction error), which Task 17 compares FeatherQuant's calibrator against.

- [ ] **Step 1: Write the reference runner**

This runs in a throwaway venv (GPU, `gptqmodel`) and is **not** part of the shipped package — it produces numbers, not library code.

```python
#!/usr/bin/env python3
"""bench/harness/run_gptq_reference.py — unconstrained reference GPTQ.

Runs GPTQ on the GPU with no memory ceiling, records perplexity and
per-linear reconstruction error. This is the quality target M4 must match
within noise. Requires: uv pip install gptqmodel torch datasets
Usage: run_gptq_reference.py HF_MODEL_DIR OUT_DIR RUN_ID
"""
import json
import sys
import time

import torch
from gptqmodel import GPTQModel, QuantizeConfig

from featherquant.run_manifest import RunManifest


def main() -> None:
    model_dir, out_dir, run_id = sys.argv[1:4]
    # 128 samples x 512 tokens: the same calibration shape spec §3.3 uses.
    texts = [l for l in open("bench/data/wiki.test.raw").read().split("\n\n")
             if len(l) > 512][:128]
    cfg = QuantizeConfig(bits=4, group_size=128, damp_percent=0.01,
                         desc_act=False, sym=True)
    t0 = time.monotonic()
    try:
        model = GPTQModel.load(model_dir, cfg)
        model.quantize(texts)
        model.save(out_dir)
    except Exception as exc:
        raise RuntimeError(f"reference GPTQ failed: {exc}") from exc
    runtime = time.monotonic() - t0
    m = RunManifest.new(run_id, {"id": model_dir, "revision": "local",
                                 "sha256": "unknown"},
                        "gptq_reference_4bit_g128", 0, "nvme")
    m.enforcement = "none_unconstrained"
    m.runtime_seconds = round(runtime, 3)
    m.peak_observed_bytes = torch.cuda.max_memory_allocated()
    m.quality = {"ppl": None,
                 "ppl_dataset": "wikitext-2-raw/wiki.test.raw c=512 tokenizer=qwen3",
                 "tasks": {}}
    m.save(f"bench/manifests/{run_id}.json")
    print(json.dumps({"runtime_s": runtime}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on Qwen3-0.6B**

```bash
uv venv /tmp/gptq-venv && /tmp/gptq-venv/bin/uv pip install gptqmodel torch datasets
/tmp/gptq-venv/bin/python bench/harness/run_gptq_reference.py \
  ~/models/qwen3-0.6b /tmp/m0_gptq_qwen3_0.6b m0_gptq_reference
```

Expected: a manifest in `bench/manifests/m0_gptq_reference.json`.

- [ ] **Step 3: Fill `quality.ppl` for the reference**

Evaluate the GPTQ output's perplexity on the same corpus, context length and tokenizer as Task 3 Step 4, and write it into the manifest. If the toolchain cannot produce a comparable ppl (different tokenizer path), set `"ppl": null` and record the blocker in `docs/baselines.md` — an honest gap beats an incomparable number.

- [ ] **Step 4: Document the M0 gate as passed**

`docs/baselines.md` must now list four run ids and state, per spec §6.1, which baselines succeeded under a ceiling and which broke, with the exact ceiling.

- [ ] **Step 5: Commit**

```bash
git add bench/harness/run_gptq_reference.py bench/manifests docs/baselines.md
git commit -m "feat: reference GPTQ baseline; M0 gate documented with four run manifests"
```

---

## Milestone M1 — Indexer

**Gate:** a model index for three model families with different naming conventions, `largest_tensor_bytes` correct, and zero weight bytes read (verified with `strace` byte counts).

### Task 5: Role classification

**Files:**
- Create: `featherquant/roles.py`
- Test: `tests/unit/test_roles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Role` (a `str` `Enum` with members `EMBED, ATTN_Q, ATTN_K, ATTN_V, ATTN_O, FFN_GATE, FFN_UP, FFN_DOWN, NORM, OUTPUT`, values `"embed"`, `"attn_q"`, …); `classify_hf(name: str) -> tuple[Role, int | None]` returning `(role, layer_index)`; `classify_gguf(name: str) -> tuple[Role, int | None]`. Both raise `RuntimeError` on an unrecognised name — never guess. Consumed by `indexer.py` (Task 6), `planner.py` (Task 9), `calibrator.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_roles.py
import pytest

from featherquant.roles import Role, classify_gguf, classify_hf


@pytest.mark.parametrize("name,role,layer", [
    ("model.embed_tokens.weight", Role.EMBED, None),
    ("lm_head.weight", Role.OUTPUT, None),
    ("model.norm.weight", Role.NORM, None),
    ("model.layers.7.self_attn.q_proj.weight", Role.ATTN_Q, 7),
    ("model.layers.7.self_attn.k_proj.weight", Role.ATTN_K, 7),
    ("model.layers.7.self_attn.v_proj.weight", Role.ATTN_V, 7),
    ("model.layers.7.self_attn.o_proj.weight", Role.ATTN_O, 7),
    ("model.layers.0.mlp.gate_proj.weight", Role.FFN_GATE, 0),
    ("model.layers.0.mlp.up_proj.weight", Role.FFN_UP, 0),
    ("model.layers.0.mlp.down_proj.weight", Role.FFN_DOWN, 0),
    ("model.layers.3.input_layernorm.weight", Role.NORM, 3),
    ("model.layers.3.self_attn.q_norm.weight", Role.NORM, 3),
    ("transformer.h.5.attn.c_attn.weight", Role.ATTN_Q, 5),
])
def test_classify_hf(name, role, layer):
    assert classify_hf(name) == (role, layer)


@pytest.mark.parametrize("name,role,layer", [
    ("token_embd.weight", Role.EMBED, None),
    ("output.weight", Role.OUTPUT, None),
    ("output_norm.weight", Role.NORM, None),
    ("blk.12.attn_q.weight", Role.ATTN_Q, 12),
    ("blk.12.ffn_down.weight", Role.FFN_DOWN, 12),
    ("blk.12.attn_norm.weight", Role.NORM, 12),
])
def test_classify_gguf(name, role, layer):
    assert classify_gguf(name) == (role, layer)


def test_unknown_name_fails_loudly():
    with pytest.raises(RuntimeError, match="unrecognised tensor name"):
        classify_hf("model.layers.0.mystery.weight")


def test_role_values_match_spec():
    assert {r.value for r in Role} == {
        "embed", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate",
        "ffn_up", "ffn_down", "norm", "output"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_roles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.roles'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/roles.py
"""Tensor-name -> role classification, in one place.

Naming conventions differ across model families and drift between
releases. Every downstream decision (planner sizing, calibrator ordering,
quantizer type rules) keys off Role, never off a name substring match at
the call site.
"""
import re
from enum import Enum


class Role(str, Enum):
    """The ten roles spec §4.1 defines. Values are the manifest strings."""
    EMBED = "embed"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_O = "attn_o"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    NORM = "norm"
    OUTPUT = "output"


# (regex, role). Order matters: norms are matched before the projections
# they sit next to (q_norm must not fall through to attn_q).
_HF_RULES: list[tuple[re.Pattern[str], Role]] = [
    (re.compile(r"(^|\.)(embed_tokens|wte|word_embeddings)\."), Role.EMBED),
    (re.compile(r"(^|\.)(lm_head|output_layer)\."), Role.OUTPUT),
    (re.compile(r"norm"), Role.NORM),
    (re.compile(r"\.(q_proj|c_attn)\."), Role.ATTN_Q),
    (re.compile(r"\.k_proj\."), Role.ATTN_K),
    (re.compile(r"\.v_proj\."), Role.ATTN_V),
    (re.compile(r"\.(o_proj|c_proj|dense)\."), Role.ATTN_O),
    (re.compile(r"\.(gate_proj|w1)\."), Role.FFN_GATE),
    (re.compile(r"\.(up_proj|w3|c_fc)\."), Role.FFN_UP),
    (re.compile(r"\.(down_proj|w2)\."), Role.FFN_DOWN),
]

_GGUF_RULES: list[tuple[re.Pattern[str], Role]] = [
    (re.compile(r"^token_embd\."), Role.EMBED),
    (re.compile(r"^output\.weight$"), Role.OUTPUT),
    (re.compile(r"norm"), Role.NORM),
    (re.compile(r"attn_q"), Role.ATTN_Q),
    (re.compile(r"attn_k"), Role.ATTN_K),
    (re.compile(r"attn_v"), Role.ATTN_V),
    (re.compile(r"attn_output"), Role.ATTN_O),
    (re.compile(r"ffn_gate"), Role.FFN_GATE),
    (re.compile(r"ffn_up"), Role.FFN_UP),
    (re.compile(r"ffn_down"), Role.FFN_DOWN),
]

# Layer index: "model.layers.7." / "transformer.h.5." / "blk.12."
_HF_LAYER = re.compile(r"(?:layers|\.h)\.(\d+)\.")
_GGUF_LAYER = re.compile(r"^blk\.(\d+)\.")


def _classify(name: str, rules: list[tuple[re.Pattern[str], Role]],
              layer_re: re.Pattern[str]) -> tuple[Role, int | None]:
    """Apply an ordered rule table; fail loudly when nothing matches."""
    m = layer_re.search(name)
    layer = int(m[1]) if m else None
    for pattern, role in rules:
        if pattern.search(name):
            return role, layer
    raise RuntimeError(
        f"unrecognised tensor name {name!r}: add a rule to "
        f"featherquant/roles.py rather than guessing a role")


def classify_hf(name: str) -> tuple[Role, int | None]:
    """Role and layer index for a Hugging Face parameter name."""
    return _classify(name, _HF_RULES, _HF_LAYER)


def classify_gguf(name: str) -> tuple[Role, int | None]:
    """Role and layer index for a GGUF tensor name."""
    return _classify(name, _GGUF_RULES, _GGUF_LAYER)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_roles.py -v && .venv/bin/ruff check featherquant && .venv/bin/mypy featherquant`
Expected: 21 passed, clean.

- [ ] **Step 5: Commit**

```bash
git add featherquant/roles.py tests/unit/test_roles.py
git commit -m "feat: role classification table for HF and GGUF tensor names"
```

---

### Task 6: Indexer module and `featherquant index`

**Files:**
- Create: `featherquant/indexer.py`
- Modify: `featherquant/cli.py`
- Test: `tests/unit/test_indexer.py`, `tests/integration/test_index_no_weight_reads.py`

**Interfaces:**
- Consumes: `Role`, `classify_hf`, `classify_gguf` (Task 5); `parse_shard_header` from `featherquant/st_source.py`.
- Produces:
  - `@dataclass(frozen=True) TensorInfo(name, shape, dtype, shard_path, byte_offset, byte_length, quant_eligible, layer_index, role)` — `shape` is a `tuple[int, ...]` in **source (HF row-major) order**, `dtype` one of `"F32" | "F16" | "BF16"`, `role` a `str` (the `Role` value).
  - `@dataclass ModelIndex(model_arch, n_layers, hidden_size, intermediate_size, vocab_size, head_dims, tensors, largest_tensor_bytes, total_bytes)` where `head_dims` is `{"n_heads": int, "n_kv_heads": int, "head_dim": int}`.
  - `ModelIndex.save(path) -> None`, `ModelIndex.load(path) -> ModelIndex`, `ModelIndex.layer_tensors(i: int) -> list[TensorInfo]`, `ModelIndex.by_role(role: Role) -> list[TensorInfo]`.
  - `index_model(model_path: str) -> ModelIndex` — dispatches on directory (safetensors) vs file (GGUF).
  - CLI: `featherquant index <model_path> -o manifest.json`.
  These are consumed by `planner.py` (Task 9) and `calibrator.py` (Task 16).

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_indexer.py
import json
import struct

import numpy as np
import pytest

from featherquant.indexer import ModelIndex, index_model
from featherquant.roles import Role


def write_safetensors(path, arrays):
    """Minimal safetensors writer for fixtures (header + raw data)."""
    header, offset = {}, 0
    blobs = []
    for name, arr in arrays.items():
        raw = arr.tobytes()
        dtype = {"float32": "F32", "float16": "F16"}[arr.dtype.name]
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


@pytest.fixture
def tiny_model(tmp_path):
    """A 2-layer model with the shapes an indexer must derive, not assume."""
    h, i, v = 8, 16, 32
    arrays = {"model.embed_tokens.weight": np.zeros((v, h), np.float16),
              "model.norm.weight": np.zeros((h,), np.float32),
              "lm_head.weight": np.zeros((v, h), np.float16)}
    for layer in range(2):
        p = f"model.layers.{layer}."
        arrays[p + "self_attn.q_proj.weight"] = np.zeros((h, h), np.float16)
        arrays[p + "self_attn.k_proj.weight"] = np.zeros((h // 2, h), np.float16)
        arrays[p + "self_attn.v_proj.weight"] = np.zeros((h // 2, h), np.float16)
        arrays[p + "self_attn.o_proj.weight"] = np.zeros((h, h), np.float16)
        arrays[p + "mlp.gate_proj.weight"] = np.zeros((i, h), np.float16)
        arrays[p + "mlp.up_proj.weight"] = np.zeros((i, h), np.float16)
        arrays[p + "mlp.down_proj.weight"] = np.zeros((h, i), np.float16)
        arrays[p + "input_layernorm.weight"] = np.zeros((h,), np.float32)
    write_safetensors(tmp_path / "model.safetensors", arrays)
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "num_hidden_layers": 2, "hidden_size": h, "intermediate_size": i,
        "vocab_size": v, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0}))
    return tmp_path


def test_index_derives_dimensions(tiny_model):
    idx = index_model(str(tiny_model))
    assert idx.model_arch == "qwen3"
    assert (idx.n_layers, idx.hidden_size, idx.intermediate_size,
            idx.vocab_size) == (2, 8, 16, 32)
    assert idx.head_dims == {"n_heads": 2, "n_kv_heads": 1, "head_dim": 4}


def test_largest_tensor_is_the_embedding(tiny_model):
    idx = index_model(str(tiny_model))
    assert idx.largest_tensor_bytes == 32 * 8 * 2       # v x h, fp16
    assert idx.total_bytes == sum(t.byte_length for t in idx.tensors)


def test_roles_and_layer_indices(tiny_model):
    idx = index_model(str(tiny_model))
    downs = idx.by_role(Role.FFN_DOWN)
    assert sorted(t.layer_index for t in downs) == [0, 1]
    assert all(t.quant_eligible for t in downs)
    assert not idx.by_role(Role.NORM)[0].quant_eligible   # 1-D tensor
    assert {t.role for t in idx.layer_tensors(0)} == {
        "attn_q", "attn_k", "attn_v", "attn_o",
        "ffn_gate", "ffn_up", "ffn_down", "norm"}


def test_roundtrip(tiny_model, tmp_path):
    idx = index_model(str(tiny_model))
    p = tmp_path / "manifest.json"
    idx.save(str(p))
    again = ModelIndex.load(str(p))
    assert again.tensors == idx.tensors
    assert again.largest_tensor_bytes == idx.largest_tensor_bytes


def test_missing_config_fails_loudly(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="config.json"):
        index_model(str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.indexer'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/indexer.py
"""Model index (spec §4.1): metadata only, never a weight byte.

Reads config.json, model.safetensors.index.json and each shard's JSON
header — or a GGUF's metadata — and emits the manifest every later stage
plans from. Dimensions are derived from the checkpoint, never assumed.
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .roles import Role, classify_gguf, classify_hf
from .st_source import ST_ITEMSIZE, parse_shard_header

# Roles that are never quantized regardless of shape.
_NEVER_QUANT = {Role.NORM}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor's location and classification. No data, ever."""
    name: str
    shape: tuple[int, ...]     # source order (HF row-major)
    dtype: str                 # "F32" | "F16" | "BF16"
    shard_path: str
    byte_offset: int           # absolute offset in shard_path
    byte_length: int
    quant_eligible: bool
    layer_index: int | None
    role: str


@dataclass
class ModelIndex:
    """Everything the planner needs to size a job without reading weights."""
    model_arch: str
    n_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    head_dims: dict[str, int]
    tensors: list[TensorInfo]
    largest_tensor_bytes: int
    total_bytes: int

    def layer_tensors(self, i: int) -> list[TensorInfo]:
        """Every tensor belonging to transformer layer ``i``."""
        return [t for t in self.tensors if t.layer_index == i]

    def by_role(self, role: Role) -> list[TensorInfo]:
        """Every tensor with the given role, in index order."""
        return [t for t in self.tensors if t.role == role.value]

    def save(self, path: str) -> None:
        """Write the index as JSON (the artifact spec §8 calls manifest.json)."""
        try:
            with open(path, "w") as f:
                json.dump(asdict(self), f, indent=2)
        except OSError as exc:
            raise RuntimeError(f"cannot write model index {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> "ModelIndex":
        """Read back an index, failing loudly on a schema mismatch."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read model index {path}: {exc}") from exc
        try:
            tensors = [TensorInfo(**{**t, "shape": tuple(t["shape"])})
                       for t in d.pop("tensors")]
            return cls(tensors=tensors, **d)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"malformed model index {path}: {exc}") from exc


def _read_config(model_dir: str) -> dict[str, Any]:
    """Load config.json; every dimension below comes from here."""
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as f:
            cfg: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    return cfg


def _shard_names(model_dir: str) -> list[str]:
    """Shard file names from the index, or the single-file fallback."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        return ["model.safetensors"]
    try:
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"cannot read {index}: {exc}") from exc
    return sorted(set(weight_map.values()))


def index_safetensors(model_dir: str) -> ModelIndex:
    """Build an index from an HF checkpoint directory."""
    cfg = _read_config(model_dir)
    try:
        arch = str(cfg["model_type"])
        n_layers = int(cfg["num_hidden_layers"])
        hidden = int(cfg["hidden_size"])
        inter = int(cfg["intermediate_size"])
        vocab = int(cfg["vocab_size"])
        n_heads = int(cfg["num_attention_heads"])
        n_kv = int(cfg.get("num_key_value_heads", n_heads))
        head_dim = int(cfg.get("head_dim", hidden // n_heads))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(
            f"{model_dir}/config.json is missing a required dimension: "
            f"{exc}") from exc

    tensors: list[TensorInfo] = []
    for shard in _shard_names(model_dir):
        path = os.path.join(model_dir, shard)
        entries, data_base = parse_shard_header(path)
        for name, st in entries.items():
            role, layer = classify_hf(name)
            nbytes = st.end - st.start
            # 2-D+ float tensors with a Q8_0-sized contiguous row are the
            # only quantizable ones; K-quant eligibility is a planner call.
            eligible = (len(st.shape) >= 2 and role not in _NEVER_QUANT
                        and st.dtype in ST_ITEMSIZE
                        and st.shape[-1] % 32 == 0)
            tensors.append(TensorInfo(
                name=name, shape=tuple(st.shape), dtype=st.dtype,
                shard_path=path, byte_offset=data_base + st.start,
                byte_length=nbytes, quant_eligible=eligible,
                layer_index=layer, role=role.value))
    if not tensors:
        raise RuntimeError(f"no tensors found under {model_dir}")
    return ModelIndex(
        model_arch=arch, n_layers=n_layers, hidden_size=hidden,
        intermediate_size=inter, vocab_size=vocab,
        head_dims={"n_heads": n_heads, "n_kv_heads": n_kv,
                   "head_dim": head_dim},
        tensors=tensors,
        largest_tensor_bytes=max(t.byte_length for t in tensors),
        total_bytes=sum(t.byte_length for t in tensors))


def index_gguf(path: str) -> ModelIndex:
    """Build an index from a GGUF's metadata (no tensor data is touched)."""
    from gguf import GGUFReader          # local import: metadata path only

    try:
        reader = GGUFReader(path)
    except Exception as exc:
        raise RuntimeError(f"cannot read GGUF metadata from {path}: {exc}") from exc
    try:
        arch = str(reader.fields["general.architecture"].contents())

        def kv(suffix: str) -> int:
            field = reader.fields[f"{arch}.{suffix}"]
            return int(field.contents())

        n_layers = kv("block_count")
        hidden = kv("embedding_length")
        inter = kv("feed_forward_length")
        n_heads = kv("attention.head_count")
        n_kv = kv("attention.head_count_kv")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{path} is missing a required metadata key: {exc}") from exc
    tensors: list[TensorInfo] = []
    vocab = 0
    for t in reader.tensors:
        role, layer = classify_gguf(t.name)
        shape = tuple(int(d) for d in reversed(list(t.shape)))  # ne -> source
        if role is Role.EMBED:
            vocab = shape[0]
        tensors.append(TensorInfo(
            name=t.name, shape=shape, dtype=t.tensor_type.name,
            shard_path=path, byte_offset=int(t.data_offset),
            byte_length=int(t.n_bytes),
            quant_eligible=(len(shape) >= 2 and role not in _NEVER_QUANT
                            and int(t.shape[0]) % 32 == 0),
            layer_index=layer, role=role.value))
    reader.fields.clear()   # drop the KV object graph immediately
    return ModelIndex(
        model_arch=arch, n_layers=n_layers, hidden_size=hidden,
        intermediate_size=inter, vocab_size=vocab,
        head_dims={"n_heads": n_heads, "n_kv_heads": n_kv,
                   "head_dim": hidden // n_heads if n_heads else 0},
        tensors=tensors,
        largest_tensor_bytes=max(t.byte_length for t in tensors),
        total_bytes=sum(t.byte_length for t in tensors))


def index_model(model_path: str) -> ModelIndex:
    """Index an HF checkpoint directory or a GGUF file."""
    if os.path.isdir(model_path):
        return index_safetensors(model_path)
    return index_gguf(model_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_indexer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Write the failing strace test (zero weight bytes read)**

```python
# tests/integration/test_index_no_weight_reads.py
"""The indexer's gate: it must never read weight bytes.

strace counts bytes returned by read()/pread64() on the shard. Header
bytes are legitimate; anything on the order of the tensor payload is not.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("strace"), reason="strace absent")


def _make_model(tmp_path):
    from tests.unit.test_indexer import write_safetensors   # fixture writer
    import numpy as np
    # 4 MiB of payload: any full read is unmissable next to a ~1 KiB header.
    write_safetensors(tmp_path / "model.safetensors",
                      {"model.embed_tokens.weight": np.zeros((1024, 512), np.float16),
                       "model.norm.weight": np.zeros((512,), np.float32)})
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "num_hidden_layers": 0, "hidden_size": 512,
        "intermediate_size": 1024, "vocab_size": 1024,
        "num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 64}))
    return tmp_path


def test_index_reads_only_headers(tmp_path):
    model = _make_model(tmp_path)
    out = tmp_path / "manifest.json"
    trace = tmp_path / "trace.txt"
    subprocess.run(
        ["strace", "-f", "-e", "trace=read,pread64", "-o", str(trace),
         sys.executable, "-m", "featherquant.cli", "index", str(model),
         "-o", str(out)], check=True, capture_output=True)
    text = trace.read_text()
    # Sum the return values of successful reads on any .safetensors fd.
    total = sum(int(m) for m in re.findall(r"= (\d+)$", text, re.M))
    payload = 1024 * 512 * 2
    assert out.exists()
    assert total < payload // 4, f"indexer read {total} B; payload is {payload} B"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_index_no_weight_reads.py -v`
Expected: FAIL — `featherquant.cli` has no `index` subcommand yet (`error: unrecognized arguments`).

- [ ] **Step 7: Add the `index` subcommand, keeping the legacy flat CLI**

In `featherquant/cli.py`, restructure `main()` so the first positional argument selects a subcommand while the existing flag form still works (`featherquant.sh` and `tests/test_cli.py` depend on it):

```python
# featherquant/cli.py  (replace the body of main())
SUBCOMMANDS = {"index", "plan", "run", "verify", "bench"}


def main() -> None:
    """Entry point: spec §8 subcommands, or the legacy flat-flag form."""
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        _dispatch(argv[0], argv[1:])
        return
    _legacy_quantize(argv)      # the existing --model/--output/--max-ram path


def _cmd_index(argv: list[str]) -> None:
    """featherquant index <model_path> -o manifest.json"""
    p = argparse.ArgumentParser(prog="featherquant index",
                                description="Emit a model index (metadata only)")
    p.add_argument("model_path", help="HF checkpoint directory or GGUF file")
    p.add_argument("-o", "--output", required=True, help="index JSON path")
    a = p.parse_args(argv)
    try:
        idx = index_model(a.model_path)
    except RuntimeError as exc:
        sys.exit(f"featherquant index: error: {exc}")
    idx.save(a.output)
    print(f"{len(idx.tensors)} tensors, {idx.n_layers} layers, "
          f"largest tensor {idx.largest_tensor_bytes / 2**20:.1f} MiB -> "
          f"{a.output}")


def _dispatch(name: str, argv: list[str]) -> None:
    """Route a subcommand; later tasks register plan/run/verify/bench here."""
    handlers = {"index": _cmd_index}
    try:
        handlers[name](argv)
    except KeyError:
        sys.exit(f"featherquant: {name} is not implemented yet")
```

Move the current `main()` body verbatim into `_legacy_quantize(argv)` (it takes `argv` and passes it to `p.parse_args(argv)`), and add `from .indexer import index_model` at the top.

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant`
Expected: all green, including the existing `tests/test_cli.py` and `tests/test_launcher.py` (the legacy form is unchanged).

- [ ] **Step 9: Index three model families — the M1 gate**

```bash
.venv/bin/featherquant index ~/models/qwen3-0.6b -o /tmp/idx_qwen3_0.6b.json
.venv/bin/featherquant index ~/models/qwen3-14b  -o /tmp/idx_qwen3_14b.json
.venv/bin/featherquant index ~/models/qwen3-0.6b-bf16.gguf -o /tmp/idx_gguf.json
```

For the third family, download any Llama-architecture checkpoint (different naming: `model.layers.N.self_attn.*` with `gate_proj`/`up_proj` and no `q_norm`) into `~/models/` and index it too. For each: verify `largest_tensor_bytes` equals the embedding's byte length computed by hand from `config.json` (`vocab_size × hidden_size × dtype_bytes`), and that no `classify_hf` failure occurred. Record the three checks in `docs/memory_model.md` under a "Indexer gate" heading.

- [ ] **Step 10: Commit**

```bash
git add featherquant/indexer.py featherquant/cli.py tests/unit/test_indexer.py \
        tests/integration/test_index_no_weight_reads.py docs/memory_model.md
git commit -m "feat: model indexer + featherquant index; strace-verified zero weight reads"
```

---

## Milestone M2 — Reader + writer + RTN under a ceiling

The streaming reader (`gguf_io.TensorSource`, `st_source.SafetensorsSource`), writer (`gguf_io.IncrementalWriter`) and RTN quantizers (`q8_0.py`, `ggml_backend.py`) already exist and already produce byte-identical output. What is missing is the **gate as an automated test**: bit-identity proven *under a cgroup ceiling*, in CI, plus a no-`mmap` guard.

**Gate:** RTN `Q4_K_M` output is bit-identical to `llama-quantize` on the same input, under a cgroup ceiling. Bit-identical, not "close."

### Task 7: Memory-enforced bit-identity test

**Files:**
- Create: `tests/memory/__init__.py`, `tests/memory/test_rtn_under_ceiling.py`, `tests/memory/conftest.py`
- Create: `tests/determinism/__init__.py`, `tests/determinism/test_byte_identical_runs.py`
- Create: `featherquant/validator.py`
- Modify: `pyproject.toml` (register the `memory` and `slow` markers)

**Interfaces:**
- Consumes: `quantize_model` from `featherquant/engine.py`; `bench/harness/run_under_ceiling.sh` (Task 2).
- Produces: `compare_gguf(a: str, b: str) -> list[str]` — returns a list of human-readable mismatch strings, empty when the two files match tensor-for-tensor (names, types, bytes); `structural_check(path: str) -> list[str]` — tensor count, alignment, offsets, no truncation. Consumed by `featherquant verify` (Task 11) and the M8 sweep.

- [ ] **Step 1: Write the failing validator test**

```python
# tests/unit/test_validator.py
import numpy as np

from featherquant.validator import compare_gguf, structural_check
from tests.conftest import make_gguf


def test_identical_files_compare_clean(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    arr = np.arange(64, dtype=np.float32).reshape(2, 32)
    make_gguf(a, {"t.weight": arr})
    make_gguf(b, {"t.weight": arr})
    assert compare_gguf(str(a), str(b)) == []


def test_byte_difference_is_reported(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    arr = np.arange(64, dtype=np.float32).reshape(2, 32)
    make_gguf(a, {"t.weight": arr})
    make_gguf(b, {"t.weight": arr + 1})
    msgs = compare_gguf(str(a), str(b))
    assert len(msgs) == 1 and "byte mismatch" in msgs[0]


def test_structural_check_passes_on_valid_file(tmp_path):
    p = tmp_path / "a.gguf"
    make_gguf(p, {"t.weight": np.zeros((2, 32), np.float32)})
    assert structural_check(str(p)) == []


def test_structural_check_catches_truncation(tmp_path):
    p = tmp_path / "a.gguf"
    make_gguf(p, {"t.weight": np.zeros((2, 32), np.float32)})
    with open(p, "r+b") as f:
        f.truncate(p.stat().st_size - 16)
    msgs = structural_check(str(p))
    assert any("truncated" in m for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_validator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.validator'`

- [ ] **Step 3: Write the validator**

```python
# featherquant/validator.py
"""Output validation (spec §4.8): structural, comparative, deterministic.

Loadability and numerical-vs-reference checks live in the bench harness
(they need llama.cpp binaries); this module is the pure-Python half that
CI can run on any machine.
"""
import os

import numpy as np
from gguf import GGML_QUANT_SIZES, GGUFReader

from .gguf_io import ALIGN


def _chunks_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two arrays 64 MiB at a time (works on files bigger than RAM)."""
    x, y = a.reshape(-1), b.reshape(-1)
    if x.dtype != y.dtype or x.size != y.size:
        return False
    step = max(1, (64 << 20) // x.itemsize)
    return all(np.array_equal(x[i:i + step], y[i:i + step])
               for i in range(0, x.size, step))


def compare_gguf(a: str, b: str) -> list[str]:
    """Tensor-for-tensor comparison; empty list means identical."""
    try:
        ra, rb = GGUFReader(a), GGUFReader(b)
    except Exception as exc:
        raise RuntimeError(f"cannot open GGUF for comparison: {exc}") from exc
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    msgs: list[str] = []
    if ta.keys() != tb.keys():
        msgs.append(f"tensor name sets differ: {sorted(ta.keys() ^ tb.keys())}")
    for name in sorted(ta.keys() & tb.keys()):
        x, y = ta[name], tb[name]
        if x.tensor_type != y.tensor_type:
            msgs.append(f"{name}: type {x.tensor_type.name} != {y.tensor_type.name}")
        elif not _chunks_equal(x.data, y.data):
            msgs.append(f"{name}: byte mismatch")
    return msgs


def structural_check(path: str) -> list[str]:
    """Offsets, alignment, declared sizes vs the file's actual length."""
    try:
        size = os.path.getsize(path)
        reader = GGUFReader(path)
    except Exception as exc:
        raise RuntimeError(f"cannot open {path}: {exc}") from exc
    msgs: list[str] = []
    if not reader.tensors:
        msgs.append("no tensors in file")
    for t in reader.tensors:
        blk, tsz = GGML_QUANT_SIZES[t.tensor_type]
        n_elements = int(np.prod([int(d) for d in t.shape]))
        if n_elements % blk:
            msgs.append(f"{t.name}: {n_elements} elements is not a multiple "
                        f"of block size {blk}")
        expect = n_elements // blk * tsz
        if int(t.n_bytes) != expect:
            msgs.append(f"{t.name}: declared {t.n_bytes} B, format implies {expect} B")
        if int(t.data_offset) % ALIGN:
            msgs.append(f"{t.name}: data_offset {t.data_offset} is not "
                        f"{ALIGN}-byte aligned")
        end = int(t.data_offset) + int(t.n_bytes)
        if end > size:
            msgs.append(f"{t.name}: truncated — needs {end} B, file is {size} B")
    return msgs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_validator.py -v`
Expected: 4 passed.

Note: `data_offset` in `GGUFReader` is absolute; if the alignment assertion fires on a known-good file, the offsets are relative to the data section — subtract `reader.data_offset` before the modulo and re-run. Verify against one real file before trusting the check.

- [ ] **Step 5: Write the cgroup-enforced bit-identity test**

```python
# tests/memory/conftest.py
"""Shared skips for memory tests: these require cgroup v2 and llama.cpp."""
import os
import shutil

import pytest

LLAMA_BIN = os.environ.get("LLAMA_BIN",
                           os.path.expanduser("~/llama.cpp/build-cpu/bin"))
MODEL = os.environ.get("FQ_TEST_MODEL",
                       os.path.expanduser("~/models/qwen3-0.6b-bf16.gguf"))

needs_cgroup = pytest.mark.skipif(
    not shutil.which("systemd-run"),
    reason="cgroup v2 enforcement needs systemd-run --user")
needs_llama = pytest.mark.skipif(
    not os.path.exists(f"{LLAMA_BIN}/llama-quantize") or not os.path.exists(MODEL),
    reason="llama.cpp build or test model not available")
```

```python
# tests/memory/test_rtn_under_ceiling.py
"""M2 gate: RTN output is bit-identical to llama-quantize, under a ceiling.

A test that passes without a ceiling is not a memory test (spec §9), so
the featherquant run happens inside systemd-run with MemoryMax set and
swap disabled. If the kernel OOM-kills it, the test fails.
"""
import subprocess
import sys

import pytest

from featherquant.validator import compare_gguf, structural_check
from tests.memory.conftest import LLAMA_BIN, MODEL, needs_cgroup, needs_llama

pytestmark = [pytest.mark.memory, pytest.mark.slow, needs_cgroup, needs_llama]


@pytest.mark.parametrize("fmt,ref_type", [("q8_0", "Q8_0"), ("q4_k_m", "Q4_K_M")])
def test_bit_identical_under_ceiling(tmp_path, fmt, ref_type):
    ref = tmp_path / f"ref_{fmt}.gguf"
    out = tmp_path / f"fq_{fmt}.gguf"
    subprocess.run([f"{LLAMA_BIN}/llama-quantize", MODEL, str(ref), ref_type],
                   check=True, capture_output=True)
    r = subprocess.run(
        ["bash", "bench/harness/run_under_ceiling.sh", "1G",
         sys.executable, "-m", "featherquant.cli", "--model", MODEL,
         "--output", str(out), "--format", fmt, "--max-ram", "1GB",
         "--ui", "none"],
        capture_output=True, text=True)
    assert r.returncode == 0, (
        f"exit {r.returncode} (137 = OOM-killed by the 1G ceiling)\n{r.stderr}")
    assert structural_check(str(out)) == []
    assert compare_gguf(str(ref), str(out)) == []
```

- [ ] **Step 6: Register the markers**

In `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
  "memory: runs under a cgroup v2 ceiling (spec §9)",
  "slow: minutes-scale; excluded from the default run with -m 'not slow'",
]
```

- [ ] **Step 7: Run the gate**

Run: `.venv/bin/pytest tests/memory -v -m memory`
Expected: 2 passed (or skipped with a stated reason on a machine without systemd/llama.cpp). If it fails with exit 137, the ceiling is the finding — record the smallest passing ceiling in `docs/memory_model.md`, do not raise it silently.

- [ ] **Step 8: Write the determinism test**

```python
# tests/determinism/test_byte_identical_runs.py
"""Invariant 4: same input + config + seed -> byte-identical output."""
import hashlib
from pathlib import Path

import numpy as np

from featherquant.engine import quantize_model
from tests.conftest import make_gguf


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_two_runs_are_byte_identical(tmp_path):
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(0)
    make_gguf(src, {"blk.0.ffn_down.weight":
                    rng.standard_normal((8, 256), dtype=np.float32),
                    "blk.0.attn_norm.weight": np.ones((256,), np.float32)})
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    for out in (a, b):
        quantize_model(str(src), str(out), max_ram=512 << 20, fmt="q8_0")
    assert _sha(a) == _sha(b)


def test_chunking_does_not_change_bytes(tmp_path):
    """Different chunk sizes must produce the same file (rows independent)."""
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(1)
    make_gguf(src, {"blk.0.ffn_down.weight":
                    rng.standard_normal((16, 256), dtype=np.float32)})
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize_model(str(src), str(a), max_ram=512 << 20, fmt="q8_0",
                   _force_chunk_rows=1)
    quantize_model(str(src), str(b), max_ram=512 << 20, fmt="q8_0",
                   _force_chunk_rows=16)
    assert _sha(a) == _sha(b)
```

- [ ] **Step 9: Run it**

Run: `.venv/bin/pytest tests/determinism -v`
Expected: 2 passed.

- [ ] **Step 10: Add the no-mmap guard**

```python
# tests/unit/test_no_mmap_on_measured_path.py
"""Invariant 3: featherquant/ never mmaps a weight file.

GGUFReader (metadata-only, released before streaming) is the single
allowed exception and is asserted by name, so a new mmap call site cannot
sneak in unnoticed.
"""
import pathlib
import re

ALLOWED = {"gguf_io.py", "st_source.py", "indexer.py"}   # GGUFReader users


def test_no_direct_mmap_calls():
    offenders = []
    for path in pathlib.Path("featherquant").glob("*.py"):
        text = path.read_text()
        if re.search(r"\bmmap\b|np\.memmap|numpy\.memmap", text):
            offenders.append(path.name)
    assert not [o for o in offenders if o not in ALLOWED], offenders


def test_allowed_files_only_use_gguf_reader_metadata():
    for name in ALLOWED:
        text = pathlib.Path("featherquant", name).read_text()
        assert "np.memmap" not in text and "numpy.memmap" not in text
```

- [ ] **Step 11: Run everything and commit**

Run: `.venv/bin/pytest -q -m "not slow" && .venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant`

```bash
git add featherquant/validator.py tests/unit/test_validator.py \
        tests/unit/test_no_mmap_on_measured_path.py tests/memory tests/determinism \
        pyproject.toml docs/memory_model.md
git commit -m "feat: validator + cgroup-enforced bit-identity, determinism and no-mmap gates (M2)"
```

---

## Milestone M3 — Planner

**Gate:** predicted peak within 10% of observed on ten model/budget pairs; every infeasible case refuses before allocating, naming the binding term.

### Task 8: Approximation cost table and lookup

**Files:**
- Create: `docs/approximation_costs.md`, `featherquant/approx_costs.py`
- Test: `tests/unit/test_approx_costs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `@dataclass(frozen=True) ApproxCost(rung: str, peak_delta_bytes: int | None, runtime_delta_pct: float | None, ppl_delta: float | None, task_delta: str, measured: bool)`; `load_costs(path: str = "docs/approximation_costs.md") -> dict[str, ApproxCost]`; `format_option(cost: ApproxCost, flag: str) -> str` rendering one line of the planner's refusal message, e.g. `--hessian-approx=diagonal (-598 MB, measured PPL cost +0.31)` or `... (-598 MB, PPL cost UNMEASURED)`. Consumed by `planner.py` (Task 9) and rewritten by the M6 sweep (Task 21).

- [ ] **Step 1: Create the doc with UNMEASURED rows**

```markdown
<!-- docs/approximation_costs.md -->
# Approximation costs

Every rung of the ladder (spec §5) trades memory for quality. **No number
in this table may be a guess.** A row is `UNMEASURED` until a committed run
manifest in `bench/manifests/` produces it; the `source` column names that
manifest. The planner reads this file to populate its refusal message.

Measurement context for every PPL figure in this table: wikitext-2-raw
`wiki.test.raw`, context length 512, Qwen3 tokenizer.

| rung | flag | peak Δ | runtime Δ | PPL Δ | downstream task Δ | source |
|---|---|---|---|---|---|---|
| hessian_full | `--hessian-approx=full` | 0 | 0% | 0.00 | 0.00 | reference |
| hessian_blocked | `--hessian-approx=blocked` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| hessian_lowrank | `--hessian-approx=lowrank` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| hessian_diagonal | `--hessian-approx=diagonal` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_samples_64 | `--calib-samples=64` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_samples_32 | `--calib-samples=32` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_seqlen_256 | `--calib-seqlen=256` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| calib_spill | `--calib-spill` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| stats_bf16 | `--stat-precision=bf16` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| kquant_group_joint | (implicit for K-quant targets) | 0 | 0% | UNMEASURED | UNMEASURED | UNMEASURED |
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_approx_costs.py
import pytest

from featherquant.approx_costs import format_option, load_costs


def test_loads_every_rung():
    costs = load_costs()
    assert "hessian_diagonal" in costs
    assert costs["hessian_full"].measured is True
    assert costs["hessian_diagonal"].measured is False


def test_unmeasured_option_line_says_so():
    costs = load_costs()
    line = format_option(costs["hessian_diagonal"], "--hessian-approx=diagonal")
    assert "UNMEASURED" in line


def test_measured_option_line_has_numbers(tmp_path):
    doc = tmp_path / "costs.md"
    doc.write_text(
        "| rung | flag | peak Δ | runtime Δ | PPL Δ | downstream task Δ | source |\n"
        "|---|---|---|---|---|---|---|\n"
        "| hessian_diagonal | `--hessian-approx=diagonal` | -598 MB | +4% | "
        "+0.31 | -0.4 | m6_diag.json |\n")
    costs = load_costs(str(doc))
    c = costs["hessian_diagonal"]
    assert c.peak_delta_bytes == -598 * 1000 * 1000
    assert c.ppl_delta == pytest.approx(0.31)
    assert c.measured is True
    assert "measured PPL cost +0.31" in format_option(c, "--hessian-approx=diagonal")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_approx_costs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.approx_costs'`

- [ ] **Step 4: Write the implementation**

```python
# featherquant/approx_costs.py
"""Read docs/approximation_costs.md — the planner's only source of costs.

Spec §4.2: "Never guess the quality cost." A rung whose measurement does
not exist yet is reported as UNMEASURED, in the refusal message, in front
of the user.
"""
import re
from dataclasses import dataclass

# MB in the doc means 10^6 bytes (what file-size tools report); ceilings
# elsewhere are binary. Spec §6 requires the distinction be labelled.
_UNITS = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9}


@dataclass(frozen=True)
class ApproxCost:
    """One row of the ladder table."""
    rung: str
    flag: str
    peak_delta_bytes: int | None
    runtime_delta_pct: float | None
    ppl_delta: float | None
    task_delta: str
    measured: bool


def _size(cell: str) -> int | None:
    """Parse '-598 MB' / '0' into bytes; None when UNMEASURED."""
    cell = cell.strip()
    if cell == "UNMEASURED":
        return None
    m = re.fullmatch(r"([+-]?[\d.]+)\s*([KMG]?B)?", cell)
    if not m:
        raise RuntimeError(f"cannot parse size cell {cell!r} in the cost table")
    return int(float(m[1]) * _UNITS.get((m[2] or "B").upper(), 1))


def _number(cell: str) -> float | None:
    """Parse '+0.31' / '+4%' into a float; None when UNMEASURED."""
    cell = cell.strip().rstrip("%")
    if cell == "UNMEASURED":
        return None
    try:
        return float(cell)
    except ValueError as exc:
        raise RuntimeError(f"cannot parse numeric cell {cell!r}: {exc}") from exc


def load_costs(path: str = "docs/approximation_costs.md") -> dict[str, ApproxCost]:
    """Parse the markdown table into a rung -> ApproxCost lookup."""
    try:
        lines = open(path).read().splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read cost table {path}: {exc}") from exc
    costs: dict[str, ApproxCost] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] in ("rung", "---"):
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        rung, flag, peak, runtime, ppl, task, source = cells
        cost = ApproxCost(rung=rung, flag=flag.strip("`"),
                          peak_delta_bytes=_size(peak),
                          runtime_delta_pct=_number(runtime),
                          ppl_delta=_number(ppl), task_delta=task,
                          measured=source != "UNMEASURED")
        costs[rung] = cost
    if not costs:
        raise RuntimeError(f"no cost rows parsed from {path}")
    return costs


def format_option(cost: ApproxCost, flag: str) -> str:
    """One line of the planner's INFEASIBLE options block."""
    peak = ("UNMEASURED" if cost.peak_delta_bytes is None
            else f"{cost.peak_delta_bytes / 10 ** 6:+.0f} MB")
    ppl = ("PPL cost UNMEASURED" if not cost.measured or cost.ppl_delta is None
           else f"measured PPL cost {cost.ppl_delta:+.2f}")
    return f"{flag:<28} ({peak}, {ppl})"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_approx_costs.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add docs/approximation_costs.md featherquant/approx_costs.py tests/unit/test_approx_costs.py
git commit -m "feat: approximation cost table with UNMEASURED rows and planner-facing lookup"
```

---

### Task 9: Planner — budget equation and infeasible refusal

**Files:**
- Create: `featherquant/planner.py`
- Test: `tests/unit/test_planner.py`

**Interfaces:**
- Consumes: `ModelIndex`, `TensorInfo` (Task 6); `Role` (Task 5); `load_costs`, `format_option` (Task 8); `RESERVE` from `featherquant/engine.py`.
- Produces:
  - `@dataclass(frozen=True) CalibConfig(samples: int, seqlen: int, act_dtype: str = "fp16", spill: bool = False)`.
  - `@dataclass(frozen=True) PeakEstimate(layer_weights: int, activation_cache: int, attn_scratch: int, hessian: int, output_buffer: int, runtime_overhead: int)` with `total() -> int` and `binding_term() -> tuple[str, int]` (largest component, excluding `runtime_overhead`).
  - `@dataclass TensorPlan(name: str, role: str, layer_index: int | None, target_type: str, row_group_rows: int, n_row_groups: int, source_bytes: int, output_bytes: int)`.
  - `@dataclass Plan(model_path, index_path, method, fmt, budget_bytes, calib, approximations, hessian_approx, peak, tensor_plans, layer_order, created)` with `save(path)`, `load(path)`, `predicted_peak_bytes` property.
  - `class InfeasiblePlan(RuntimeError)` whose `str()` is the spec §4.2 message.
  - `estimate_peak(index, budget_bytes, method, calib, hessian_approx, row_group_rows, runtime_overhead) -> PeakEstimate`.
  - `plan_job(index, budget_bytes, method, fmt, calib, hessian_approx="full", runtime_overhead=None, costs_path="docs/approximation_costs.md") -> Plan` — raises `InfeasiblePlan` **before allocating anything**.
  These are consumed by the CLI (Task 10), `calibrator.py` (Task 16) and `bench.py` (Task 24).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_planner.py
import pytest

from featherquant.indexer import ModelIndex, TensorInfo
from featherquant.planner import (
    CalibConfig,
    InfeasiblePlan,
    Plan,
    estimate_peak,
    plan_job,
)


def spec_index():
    """The spec §3.3 worked example: h=4096, i=12288, v=151936, BF16."""
    h, i, v, n_layers = 4096, 12288, 151936, 36
    tensors = [TensorInfo("model.embed_tokens.weight", (v, h), "BF16", "s", 0,
                          v * h * 2, True, None, "embed")]
    for layer in range(n_layers):
        for name, role, shape in [
            ("q_proj", "attn_q", (h, h)), ("k_proj", "attn_k", (1024, h)),
            ("v_proj", "attn_v", (1024, h)), ("o_proj", "attn_o", (h, h)),
            ("gate_proj", "ffn_gate", (i, h)), ("up_proj", "ffn_up", (i, h)),
            ("down_proj", "ffn_down", (h, i)),
        ]:
            tensors.append(TensorInfo(
                f"model.layers.{layer}.{name}.weight", shape, "BF16", "s", 0,
                shape[0] * shape[1] * 2, True, layer, role))
    return ModelIndex("qwen3", n_layers, h, i, v,
                      {"n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
                      tensors, max(t.byte_length for t in tensors),
                      sum(t.byte_length for t in tensors))


def test_hessian_is_sized_by_intermediate_not_hidden():
    idx = spec_index()
    est = estimate_peak(idx, 2 << 30, "gptq", CalibConfig(128, 512), "full",
                        row_group_rows=256, runtime_overhead=100 << 20)
    assert est.hessian == 12288 ** 2 * 4          # down_proj input, not h
    assert est.hessian != 4096 ** 2 * 4


def test_activation_cache_matches_spec_worked_example():
    idx = spec_index()
    est = estimate_peak(idx, 2 << 30, "gptq", CalibConfig(128, 512), "full",
                        row_group_rows=256, runtime_overhead=100 << 20)
    assert est.activation_cache == 128 * 512 * 4096 * 2
    assert est.layer_weights == sum(t.byte_length
                                    for t in idx.layer_tensors(0))
    # Spec §3.3 lands near 1.62 GB; allow the extra terms it rolls into
    # "output + overhead".
    assert 1.4e9 < est.total() < 1.9e9


def test_binding_term_is_named():
    idx = spec_index()
    est = estimate_peak(idx, 1 << 30, "gptq", CalibConfig(128, 512), "full",
                        row_group_rows=256, runtime_overhead=100 << 20)
    name, value = est.binding_term()
    assert name == "hessian" and value == 12288 ** 2 * 4


def test_infeasible_refuses_with_actionable_message():
    idx = spec_index()
    with pytest.raises(InfeasiblePlan) as exc:
        plan_job(idx, 1 << 30, "gptq", "q4_k_m", CalibConfig(128, 512))
    msg = str(exc.value)
    assert msg.startswith("INFEASIBLE: budget 1.00 GiB < required")
    assert "binding term: hessian" in msg
    assert "d_in=12288" in msg
    assert "--hessian-approx=diagonal" in msg
    assert "--calib-samples=64" in msg
    assert "UNMEASURED" in msg          # the table is not populated yet


def test_infeasible_reports_largest_tensor_when_budget_is_below_the_floor():
    idx = spec_index()
    with pytest.raises(InfeasiblePlan) as exc:
        plan_job(idx, 64 << 20, "rtn", "q8_0", CalibConfig(0, 0))
    assert "largest single tensor" in str(exc.value)


def test_feasible_plan_roundtrips(tmp_path):
    idx = spec_index()
    plan = plan_job(idx, 4 << 30, "gptq", "q4_k_m", CalibConfig(128, 512),
                    runtime_overhead=100 << 20)
    assert plan.peak.total() <= plan.budget_bytes
    assert all(tp.row_group_rows >= 1 for tp in plan.tensor_plans)
    assert plan.layer_order == list(range(idx.n_layers))
    p = tmp_path / "plan.json"
    plan.save(str(p))
    assert Plan.load(str(p)).peak.total() == plan.peak.total()


def test_row_groups_align_to_superblock():
    idx = spec_index()
    plan = plan_job(idx, 4 << 30, "gptq", "q4_k_m", CalibConfig(128, 512),
                    runtime_overhead=100 << 20)
    for tp in plan.tensor_plans:
        if tp.target_type in ("Q4_K", "Q6_K"):
            # Row groups slice along rows; each row must hold whole
            # superblocks, and the group count must cover every row.
            assert tp.row_group_rows * tp.n_row_groups >= 1
    embed = next(tp for tp in plan.tensor_plans if tp.role == "embed")
    assert embed.n_row_groups > 1, "the embedding must be split into row groups"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.planner'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/planner.py
"""Budget planning (spec §4.2): compute the peak before touching a weight.

peak = layer_weights + activation_cache + hessian + output_buffer
     + runtime_overhead                                  (spec §3.1)

The planner either produces a plan whose predicted peak fits the declared
budget, or refuses with the binding term named and priced. It never
silently degrades a method.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .approx_costs import format_option, load_costs
from .indexer import ModelIndex
from .roles import Role
from .run_manifest import today_ddmmyyyy

# Working precision for statistics and dequantized weights.
FP32 = 4
# Bytes per cached activation element by name.
ACT_BYTES = {"fp16": 2, "fp32": 4, "bf16": 2}
# K-quant superblock: every row group must hold whole superblocks.
QK_K = 256
# Hessian memory as a fraction of d_in^2*4 for each approximation rung.
# 'blocked' keeps one panel plus one tile row resident, not the whole matrix.
HESSIAN_FRACTION = {"full": 1.0, "blocked": 0.10, "lowrank": 0.05,
                    "diagonal": 0.0}


class InfeasiblePlan(RuntimeError):
    """Raised before any allocation when the budget cannot hold the plan."""


@dataclass(frozen=True)
class CalibConfig:
    """Calibration set shape. samples=0 means no calibration (RTN)."""
    samples: int
    seqlen: int
    act_dtype: str = "fp16"
    spill: bool = False


@dataclass(frozen=True)
class PeakEstimate:
    """Every term of the budget equation, in bytes."""
    layer_weights: int
    activation_cache: int
    attn_scratch: int
    hessian: int
    output_buffer: int
    runtime_overhead: int

    def total(self) -> int:
        """Predicted peak resident bytes."""
        return (self.layer_weights + self.activation_cache + self.attn_scratch
                + self.hessian + self.output_buffer + self.runtime_overhead)

    def binding_term(self) -> tuple[str, int]:
        """The largest controllable term (runtime overhead is not a knob)."""
        terms = {"layer_weights": self.layer_weights,
                 "activation_cache": self.activation_cache,
                 "attn_scratch": self.attn_scratch,
                 "hessian": self.hessian,
                 "output_buffer": self.output_buffer}
        name = max(terms, key=lambda k: terms[k])
        return name, terms[name]


@dataclass
class TensorPlan:
    """Per-tensor slicing strategy."""
    name: str
    role: str
    layer_index: int | None
    target_type: str
    row_group_rows: int
    n_row_groups: int
    source_bytes: int
    output_bytes: int


@dataclass
class Plan:
    """The inspectable, diffable, committable job description (spec §8)."""
    model_path: str
    index_path: str
    method: str
    fmt: str
    budget_bytes: int
    calib: CalibConfig
    hessian_approx: str
    approximations: list[dict[str, Any]]
    peak: PeakEstimate
    tensor_plans: list[TensorPlan]
    layer_order: list[int]
    created: str = field(default_factory=today_ddmmyyyy)

    @property
    def predicted_peak_bytes(self) -> int:
        """Shorthand used by the accuracy harness and the run manifest."""
        return self.peak.total()

    def save(self, path: str) -> None:
        """Write plan.json."""
        try:
            with open(path, "w") as f:
                json.dump(asdict(self), f, indent=2)
        except OSError as exc:
            raise RuntimeError(f"cannot write plan {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> "Plan":
        """Read plan.json, failing loudly on a schema mismatch."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read plan {path}: {exc}") from exc
        try:
            return cls(calib=CalibConfig(**d.pop("calib")),
                       peak=PeakEstimate(**d.pop("peak")),
                       tensor_plans=[TensorPlan(**t)
                                     for t in d.pop("tensor_plans")], **d)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"malformed plan {path}: {exc}") from exc


def _max_d_in(index: ModelIndex) -> int:
    """Largest linear input dimension in a layer — the down_proj input."""
    # Derived, never assumed: the largest last-dim among a layer's 2-D
    # tensors. For a standard decoder that is intermediate_size.
    dims = [t.shape[-1] for t in index.layer_tensors(0) if len(t.shape) >= 2]
    return max(dims) if dims else index.hidden_size


def estimate_peak(index: ModelIndex, budget_bytes: int, method: str,
                  calib: CalibConfig, hessian_approx: str,
                  row_group_rows: int, runtime_overhead: int) -> PeakEstimate:
    """Evaluate the budget equation for this configuration."""
    layer_weights = max(
        (sum(t.byte_length for t in index.layer_tensors(i))
         for i in range(index.n_layers)), default=0)
    if method == "rtn":
        # No calibration state at all: the streaming path already shipped.
        act = attn = hessian = 0
    else:
        try:
            act_bytes = ACT_BYTES[calib.act_dtype]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown activation dtype {calib.act_dtype!r}: "
                f"{sorted(ACT_BYTES)}") from exc
        act = (0 if calib.spill else
               calib.samples * calib.seqlen * index.hidden_size * act_bytes)
        # One sample's attention scores, fp32: heads x s x s. Batch size is
        # one sample; larger batches multiply this term.
        attn = index.head_dims["n_heads"] * calib.seqlen ** 2 * FP32
        d_in = _max_d_in(index)
        try:
            hessian = int(d_in ** 2 * FP32 * HESSIAN_FRACTION[hessian_approx])
        except KeyError as exc:
            raise RuntimeError(
                f"unknown hessian approximation {hessian_approx!r}: "
                f"{sorted(HESSIAN_FRACTION)}") from exc
        if hessian_approx == "diagonal":
            hessian = d_in * FP32
    # Output buffer: one row group upcast to fp32 plus its packed form.
    widest_row = max((t.shape[-1] for t in index.tensors if len(t.shape) >= 2),
                     default=index.hidden_size)
    output_buffer = row_group_rows * widest_row * (FP32 + 1)
    return PeakEstimate(layer_weights=layer_weights, activation_cache=act,
                        attn_scratch=attn, hessian=hessian,
                        output_buffer=output_buffer,
                        runtime_overhead=runtime_overhead)


def _refusal(index: ModelIndex, budget_bytes: int, est: PeakEstimate,
             calib: CalibConfig, costs_path: str) -> str:
    """Build the spec §4.2 refusal message."""
    name, value = est.binding_term()
    detail = f" (d_in={_max_d_in(index)})" if name == "hessian" else ""
    lines = [f"INFEASIBLE: budget {budget_bytes / 2 ** 30:.2f} GiB < "
             f"required {est.total() / 2 ** 30:.2f} GiB",
             f"  binding term: {name} "
             f"({value / 10 ** 6:.0f} MB{detail})"]
    if index.largest_tensor_bytes + est.runtime_overhead > budget_bytes:
        lines.append(f"  floor: largest single tensor is "
                     f"{index.largest_tensor_bytes / 10 ** 6:.0f} MB and must "
                     f"be processed in row groups; with "
                     f"{est.runtime_overhead / 10 ** 6:.0f} MB runtime "
                     f"overhead no budget below "
                     f"{(index.largest_tensor_bytes // 8 + est.runtime_overhead) / 2 ** 30:.2f} "
                     f"GiB can work")
    try:
        costs = load_costs(costs_path)
    except RuntimeError:
        costs = {}          # missing table: still refuse, just without options
    options = []
    if name == "hessian":
        for rung in ("hessian_blocked", "hessian_diagonal"):
            if rung in costs:
                options.append(format_option(costs[rung], costs[rung].flag))
    if name in ("activation_cache", "attn_scratch"):
        for rung in ("calib_samples_64", "calib_seqlen_256", "calib_spill"):
            if rung in costs:
                options.append(format_option(costs[rung], costs[rung].flag))
    if not options and costs:
        options = [format_option(c, c.flag) for c in list(costs.values())[1:4]]
    if options:
        lines.append("  options: " + ("\n           ".join(options)))
    return "\n".join(lines)


def plan_job(index: ModelIndex, budget_bytes: int, method: str, fmt: str,
             calib: CalibConfig, hessian_approx: str = "full",
             runtime_overhead: int | None = None,
             costs_path: str = "docs/approximation_costs.md",
             model_path: str = "", index_path: str = "") -> Plan:
    """Produce a feasible Plan, or refuse before doing any work."""
    from .engine import RESERVE, rss_bytes      # local: avoids a cycle

    if runtime_overhead is None:
        # Measured, not guessed: current interpreter footprint + reserve.
        runtime_overhead = rss_bytes() + RESERVE
    # Row group: the largest superblock-aligned group whose fp32 upcast plus
    # packed output fits a fixed slice of the budget (12.5%, so the buffer
    # never dominates). Always at least one row.
    widest_row = max((t.shape[-1] for t in index.tensors if len(t.shape) >= 2),
                     default=index.hidden_size)
    row_group_rows = max(1, (budget_bytes // 8) // (widest_row * (FP32 + 1)))
    est = estimate_peak(index, budget_bytes, method, calib, hessian_approx,
                        row_group_rows, runtime_overhead)
    if est.total() > budget_bytes:
        raise InfeasiblePlan(_refusal(index, budget_bytes, est, calib,
                                      costs_path))

    from .formats import FORMATS
    try:
        spec = FORMATS[fmt]
    except KeyError as exc:
        raise RuntimeError(f"unknown format {fmt!r}: {sorted(FORMATS)}") from exc

    tensor_plans: list[TensorPlan] = []
    for t in index.tensors:
        rows = t.shape[0] if len(t.shape) >= 2 else 1
        # A shim so the existing per-tensor type rules (which read ggml
        # ne-order shape and tensor_type) can score a TensorInfo.
        target = spec.tensor_type(_RuleView(t), index.n_layers)
        group = min(rows, row_group_rows)
        if target.name in ("Q4_K", "Q6_K"):
            # Row groups slice whole rows; each row already holds whole
            # superblocks because quant_eligible checked the row length.
            if t.shape[-1] % QK_K:
                raise RuntimeError(
                    f"{t.name}: row length {t.shape[-1]} is not a multiple of "
                    f"{QK_K}; a K-quant row group would emit garbage")
        tensor_plans.append(TensorPlan(
            name=t.name, role=t.role, layer_index=t.layer_index,
            target_type=target.name, row_group_rows=group,
            n_row_groups=(rows + group - 1) // group,
            source_bytes=t.byte_length,
            output_bytes=t.byte_length))     # filled exactly by the engine
    approximations: list[dict[str, Any]] = []
    if hessian_approx != "full":
        approximations.append({"rung": f"hessian_{hessian_approx}",
                               "reason": "requested"})
    if calib.spill:
        approximations.append({"rung": "calib_spill", "reason": "requested"})
    if fmt != "q8_0" and method != "rtn":
        approximations.append({"rung": "kquant_group_joint",
                               "reason": "K-quant grid forces group-joint "
                                         "quantization inside GPTQ"})
    return Plan(model_path=model_path or "", index_path=index_path or "",
                method=method, fmt=fmt, budget_bytes=budget_bytes,
                calib=calib, hessian_approx=hessian_approx,
                approximations=approximations, peak=est,
                tensor_plans=tensor_plans,
                layer_order=list(range(index.n_layers)))


class _RuleView:
    """Adapts a TensorInfo to the (shape in ne-order, tensor_type) shape the
    format rules in featherquant/formats.py expect."""

    def __init__(self, t: Any) -> None:
        from gguf import GGMLQuantizationType
        self.name = t.name
        self.shape = tuple(reversed(t.shape))      # source order -> ne order
        try:
            self.tensor_type = GGMLQuantizationType[t.dtype]
        except KeyError as exc:
            raise RuntimeError(
                f"{t.name}: dtype {t.dtype!r} has no ggml equivalent") from exc
```

Note on `_RuleView`: `formats._quantizable` also checks `t.tensor_type in ITEMSIZE`, so `F32/F16/BF16` sources classify correctly and quantized sources fall through to a verbatim copy — same behavior as the engine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_planner.py -v && .venv/bin/mypy featherquant`
Expected: 8 passed. If `test_activation_cache_matches_spec_worked_example` fails on the total bound, print the estimate's fields and reconcile against spec §3.3 before adjusting the test — the spec's worked example is the arbiter, not the code.

- [ ] **Step 5: Commit**

```bash
git add featherquant/planner.py tests/unit/test_planner.py
git commit -m "feat: planner with spec §3.1 budget equation and actionable INFEASIBLE refusal"
```

---

### Task 10: `featherquant plan` / `run` / `verify` subcommands

**Files:**
- Modify: `featherquant/cli.py`, `featherquant/engine.py`
- Test: `tests/unit/test_cli_subcommands.py`

**Interfaces:**
- Consumes: `index_model` (Task 6); `plan_job`, `Plan`, `CalibConfig`, `InfeasiblePlan` (Task 9); `compare_gguf`, `structural_check` (Task 7); `quantize_model` (existing).
- Produces:
  - `featherquant plan <manifest.json> --budget 2GiB --method gptq --format q4_k_m --calib-samples 128 --calib-seqlen 512 [--hessian-approx full|blocked|lowrank|diagonal] [--calib-spill] -o plan.json`
  - `featherquant run <plan.json> -o model-Q4_K_M.gguf [--resume] [--run-manifest run.json]`
  - `featherquant verify <model.gguf> [--reference ref.gguf]`
  - `quantize_model(..., plan: Plan | None = None, run_manifest_path: str | None = None)` — when `plan` is given, the engine uses the plan's budget, format and row-group sizes instead of re-deriving them, and writes a `RunManifest` on completion.
  - `parse_size` gains binary-unit correctness for `GiB`/`MiB` (already the case: `KMGT` map to powers of 1024) and is reused by `plan`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli_subcommands.py
import json
import subprocess
import sys

import numpy as np
import pytest

from tests.unit.test_indexer import write_safetensors


def _tiny_hf(tmp_path):
    h, i, v = 256, 512, 256
    arrays = {"model.embed_tokens.weight": np.zeros((v, h), np.float16),
              "model.norm.weight": np.zeros((h,), np.float32),
              "lm_head.weight": np.zeros((v, h), np.float16)}
    p = "model.layers.0."
    for name, shape in [("self_attn.q_proj", (h, h)), ("self_attn.k_proj", (h, h)),
                        ("self_attn.v_proj", (h, h)), ("self_attn.o_proj", (h, h)),
                        ("mlp.gate_proj", (i, h)), ("mlp.up_proj", (i, h)),
                        ("mlp.down_proj", (h, i))]:
        arrays[p + name + ".weight"] = np.zeros(shape, np.float16)
    arrays[p + "input_layernorm.weight"] = np.zeros((h,), np.float32)
    write_safetensors(tmp_path / "model.safetensors", arrays)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": h,
        "intermediate_size": i, "vocab_size": v, "num_attention_heads": 4,
        "num_key_value_heads": 4, "head_dim": 64, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0}))
    return tmp_path


def _cli(*args, **kw):
    return subprocess.run([sys.executable, "-m", "featherquant.cli", *args],
                          capture_output=True, text=True, **kw)


def test_plan_writes_plan_json(tmp_path):
    model = _tiny_hf(tmp_path)
    idx = tmp_path / "manifest.json"
    assert _cli("index", str(model), "-o", str(idx)).returncode == 0
    plan = tmp_path / "plan.json"
    r = _cli("plan", str(idx), "--budget", "4GiB", "--method", "gptq",
             "--format", "q8_0", "--calib-samples", "8", "--calib-seqlen", "64",
             "-o", str(plan))
    assert r.returncode == 0, r.stderr
    d = json.loads(plan.read_text())
    assert d["method"] == "gptq" and d["budget_bytes"] == 4 * 2 ** 30
    assert d["peak"]["hessian"] == 512 ** 2 * 4        # down_proj d_in = i
    assert d["calib"]["samples"] == 8


def test_plan_refuses_infeasible_budget_with_exit_2(tmp_path):
    model = _tiny_hf(tmp_path)
    idx = tmp_path / "manifest.json"
    _cli("index", str(model), "-o", str(idx))
    r = _cli("plan", str(idx), "--budget", "2MiB", "--method", "gptq",
             "--format", "q8_0", "--calib-samples", "128", "--calib-seqlen",
             "512", "-o", str(tmp_path / "plan.json"))
    assert r.returncode == 2
    assert "INFEASIBLE" in r.stderr
    assert "binding term" in r.stderr
    assert not (tmp_path / "plan.json").exists()   # refuse before any work


def test_verify_reports_structural_problems(tmp_path):
    from tests.conftest import make_gguf
    p = tmp_path / "a.gguf"
    make_gguf(p, {"t.weight": np.zeros((2, 32), np.float32)})
    assert _cli("verify", str(p)).returncode == 0
    with open(p, "r+b") as f:
        f.truncate(p.stat().st_size - 16)
    r = _cli("verify", str(p))
    assert r.returncode == 1 and "truncated" in r.stdout + r.stderr


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_flat_cli_still_works(tmp_path, legacy):
    """featherquant.sh passes flat flags; that path must never break."""
    from tests.conftest import make_gguf
    src = tmp_path / "src.gguf"
    make_gguf(src, {"blk.0.ffn_down.weight":
                    np.ones((4, 256), np.float32)})
    out = tmp_path / "out.gguf"
    r = _cli("--model", str(src), "--output", str(out), "--max-ram", "512MB",
             "--ui", "none")
    assert r.returncode == 0, r.stderr
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_cli_subcommands.py -v`
Expected: FAIL — `featherquant: plan is not implemented yet`.

- [ ] **Step 3: Implement the subcommands**

Add to `featherquant/cli.py`:

```python
def _cmd_plan(argv: list[str]) -> None:
    """featherquant plan manifest.json --budget 2GiB --method gptq -o plan.json"""
    p = argparse.ArgumentParser(prog="featherquant plan",
                                description="Compute a memory plan or refuse")
    p.add_argument("index_path", help="model index from `featherquant index`")
    p.add_argument("--budget", required=True, type=parse_size,
                   help="hard memory ceiling, e.g. 2GiB")
    p.add_argument("--method", default="gptq", choices=["rtn", "gptq"])
    p.add_argument("--format", default="q4_k_m", choices=["q8_0", "q4_k_m"])
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--calib-spill", action="store_true",
                   help="keep the activation cache on disk (logged approximation)")
    p.add_argument("--hessian-approx", default="full",
                   choices=["full", "blocked", "lowrank", "diagonal"])
    p.add_argument("--model", default="",
                   help="source model path recorded in the plan (default: "
                        "read from the index's first shard)")
    p.add_argument("-o", "--output", required=True, help="plan JSON path")
    a = p.parse_args(argv)
    try:
        index = ModelIndex.load(a.index_path)
        calib = CalibConfig(samples=a.calib_samples, seqlen=a.calib_seqlen,
                            spill=a.calib_spill)
        plan = plan_job(index, a.budget, a.method, a.format, calib,
                        hessian_approx=a.hessian_approx,
                        model_path=a.model or os.path.dirname(
                            index.tensors[0].shard_path),
                        index_path=a.index_path)
    except InfeasiblePlan as exc:
        # Exit 2 distinguishes "refused by design" from "crashed" (exit 1).
        sys.exit(str(exc))
    except RuntimeError as exc:
        sys.exit(f"featherquant plan: error: {exc}")
    plan.save(a.output)
    print(f"feasible: predicted peak {plan.predicted_peak_bytes / 2**30:.2f} GiB "
          f"of {a.budget / 2**30:.2f} GiB -> {a.output}")


def _cmd_run(argv: list[str]) -> None:
    """featherquant run plan.json -o model-Q4_K_M.gguf [--resume]"""
    p = argparse.ArgumentParser(prog="featherquant run",
                                description="Execute a plan")
    p.add_argument("plan_path")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-manifest", help="write a spec §6 run manifest here")
    p.add_argument("--vocab-gguf", help="metadata GGUF for safetensors input")
    p.add_argument("--ggml-lib")
    p.add_argument("--ui", choices=["auto", "rich", "plain", "none"],
                   default="auto")
    a = p.parse_args(argv)
    try:
        plan = Plan.load(a.plan_path)
    except RuntimeError as exc:
        sys.exit(f"featherquant run: error: {exc}")
    reporter = _make_reporter(a.ui)
    try:
        quantize_model(plan.model_path, a.output, plan.budget_bytes,
                       fmt=plan.fmt, ggml_lib=a.ggml_lib, resume=a.resume,
                       vocab_gguf=a.vocab_gguf, progress=reporter, plan=plan,
                       run_manifest_path=a.run_manifest)
    except RuntimeError as exc:
        sys.exit(f"featherquant run: error: {exc}")
    finally:
        if reporter is not None:
            reporter.close()


def _cmd_verify(argv: list[str]) -> None:
    """featherquant verify model.gguf [--reference ref.gguf]"""
    p = argparse.ArgumentParser(prog="featherquant verify")
    p.add_argument("model")
    p.add_argument("--reference", help="reference GGUF for byte comparison")
    a = p.parse_args(argv)
    try:
        msgs = structural_check(a.model)
        if a.reference:
            msgs += compare_gguf(a.reference, a.model)
    except RuntimeError as exc:
        sys.exit(f"featherquant verify: error: {exc}")
    for m in msgs:
        print(m)
    print("OK" if not msgs else f"{len(msgs)} problem(s)")
    sys.exit(1 if msgs else 0)
```

Extract the reporter construction from the legacy path into `_make_reporter(mode: str)` and register the new handlers in `_dispatch`: `{"index": _cmd_index, "plan": _cmd_plan, "run": _cmd_run, "verify": _cmd_verify}`.

- [ ] **Step 4: Teach the engine to accept a Plan and emit a run manifest**

In `featherquant/engine.py`, extend the signature and use the plan where it exists:

```python
def quantize_model(src: str, dst: str, max_ram: int, report: str | None = None,
                   fmt: str = "q8_0", ggml_lib: str | None = None,
                   manifest_path: str | None = None, resume: bool = False,
                   adaptive: bool = True, vocab_gguf: str | None = None,
                   plan: "Plan | None" = None,
                   run_manifest_path: str | None = None,
                   _force_chunk_rows: int | None = None,
                   _fail_after: int | None = None,
                   progress: ProgressFn | None = None) -> dict[str, Any]:
```

At the top, after the format lookup:

```python
    if plan is not None:
        # The plan is authoritative: it was computed against the declared
        # budget and committed before any compute was spent.
        fmt = plan.fmt
        max_ram = plan.budget_bytes
        stats_extra = {"predicted_peak": plan.predicted_peak_bytes}
    else:
        stats_extra = {}
```

Merge `stats_extra` into `stats`, and just before the `return stats`:

```python
    if run_manifest_path is not None:
        rm = RunManifest.new(
            run_id=os.path.basename(dst).replace(".gguf", ""),
            model={"id": src, "revision": "local",
                   "sha256": "unmeasured"},
            method=(plan.method if plan is not None else "rtn"),
            budget_bytes=max_ram,
            storage=os.environ.get("FQ_STORAGE", "nvme"))
        rm.approximations = list(plan.approximations) if plan else []
        rm.peak_observed_bytes = stats["peak_rss"]
        rm.runtime_seconds = stats["elapsed_s"]
        rm.bytes_read = stats["bytes_read"]
        rm.bytes_written = stats["bytes_written"]
        rm.output_sha256 = sha256_file(dst)
        rm.save(run_manifest_path)
```

`peak_observed_bytes` from RSS is telemetry only — the enforced ceiling is still the measurement, and `enforcement` stays `cgroup_v2_memory_max` because runs are launched through `run_under_ceiling.sh`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_cli_subcommands.py tests/test_cli.py tests/test_launcher.py -v`
Expected: all pass — new subcommands work and the legacy flat form is untouched.

- [ ] **Step 6: Commit**

```bash
git add featherquant/cli.py featherquant/engine.py tests/unit/test_cli_subcommands.py
git commit -m "feat: plan/run/verify subcommands; engine executes a committed plan"
```

---

### Task 11: Prediction-accuracy harness — the M3 gate

**Files:**
- Create: `scripts/plan_accuracy.py`
- Modify: `docs/memory_model.md`
- Test: `tests/integration/test_plan_accuracy.py`

**Interfaces:**
- Consumes: `index_model`, `plan_job`, `quantize_model`, `run_under_ceiling.sh`.
- Produces: `scripts/plan_accuracy.py MODEL_PATH BUDGET[,BUDGET...] --out accuracy.json` writing `[{"model": str, "budget_bytes": int, "predicted": int, "observed": int, "error_pct": float, "oom_killed": bool}]`; the M3 gate is `max(abs(error_pct)) <= 10` across ten model/budget pairs.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_plan_accuracy.py
import json
import subprocess
import sys

import numpy as np

from tests.conftest import make_gguf


def test_accuracy_harness_reports_error_pct(tmp_path):
    src = tmp_path / "src.gguf"
    rng = np.random.default_rng(3)
    make_gguf(src, {"blk.0.ffn_down.weight":
                    rng.standard_normal((64, 512), dtype=np.float32)})
    out = tmp_path / "accuracy.json"
    r = subprocess.run([sys.executable, "scripts/plan_accuracy.py", str(src),
                        "512MB,1GB", "--out", str(out), "--method", "rtn",
                        "--format", "q8_0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    assert all("error_pct" in row and row["observed"] > 0 for row in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_plan_accuracy.py -v`
Expected: FAIL — `can't open file 'scripts/plan_accuracy.py'`.

- [ ] **Step 3: Write the harness**

```python
#!/usr/bin/env python3
"""scripts/plan_accuracy.py — M3 gate: predicted peak vs observed peak.

Runs the planner and then the job for each budget, recording the relative
error. The gate is 10%. A run that is OOM-killed under its ceiling is
recorded, not retried at a higher budget.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from featherquant.cli import parse_size
from featherquant.engine import quantize_model
from featherquant.indexer import index_model
from featherquant.planner import CalibConfig, InfeasiblePlan, plan_job


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_path")
    p.add_argument("budgets", help="comma-separated, e.g. 512MB,1GB,2GB")
    p.add_argument("--method", default="rtn", choices=["rtn", "gptq"])
    p.add_argument("--format", default="q8_0", choices=["q8_0", "q4_k_m"])
    p.add_argument("--calib-samples", type=int, default=0)
    p.add_argument("--calib-seqlen", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    index = index_model(a.model_path)
    rows = []
    for text in a.budgets.split(","):
        budget = parse_size(text)
        calib = CalibConfig(a.calib_samples, a.calib_seqlen)
        try:
            plan = plan_job(index, budget, a.method, a.format, calib,
                            model_path=a.model_path)
        except InfeasiblePlan as exc:
            rows.append({"model": a.model_path, "budget_bytes": budget,
                         "predicted": None, "observed": None,
                         "error_pct": None, "refused": str(exc).splitlines()[0]})
            continue
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.gguf")
            try:
                stats = quantize_model(a.model_path, out, budget,
                                       fmt=a.format, plan=plan)
            except SystemExit as exc:
                rows.append({"model": a.model_path, "budget_bytes": budget,
                             "predicted": plan.predicted_peak_bytes,
                             "observed": None, "error_pct": None,
                             "refused": str(exc)})
                continue
        predicted, observed = plan.predicted_peak_bytes, stats["peak_rss"]
        rows.append({"model": a.model_path, "budget_bytes": budget,
                     "predicted": predicted, "observed": observed,
                     "error_pct": round(100 * (predicted - observed) / observed, 2),
                     "oom_killed": False})
        print(f"{text}: predicted {predicted >> 20} MiB, "
              f"observed {observed >> 20} MiB, "
              f"error {rows[-1]['error_pct']:+.1f}%")
    try:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
    except OSError as exc:
        sys.exit(f"cannot write {a.out}: {exc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_plan_accuracy.py -v`
Expected: PASS.

- [ ] **Step 5: Run the ten model/budget pairs (the gate)**

```bash
.venv/bin/python scripts/plan_accuracy.py ~/models/qwen3-0.6b-bf16.gguf \
  512MB,1GB,2GB,4GB,8GB --out /tmp/acc_0.6b.json
.venv/bin/python scripts/plan_accuracy.py ~/models/qwen3-14b-bf16.gguf \
  1GB,2GB,4GB,8GB,16GB --out /tmp/acc_14b.json
```

Expected: `max(abs(error_pct)) <= 10` across all ten rows. If a row exceeds 10%, the cost model is wrong — fix `estimate_peak` (usually `runtime_overhead` or `output_buffer`), do not widen the gate.

- [ ] **Step 6: Verify every infeasible case refuses before allocating**

```bash
for B in 1MB 8MB 64MB 128MB; do
  .venv/bin/featherquant plan /tmp/idx_qwen3_14b.json --budget $B \
    --method gptq --format q4_k_m --calib-samples 128 --calib-seqlen 512 \
    -o /tmp/should_not_exist.json; echo "exit=$?"
  test ! -f /tmp/should_not_exist.json && echo "no plan written: correct"
done
```

Expected: every case exits 2, names a binding term, and writes nothing.

- [ ] **Step 7: Write `docs/memory_model.md`**

Document: the budget equation with each term's formula (spec §3.2 verbatim); the worked example (spec §3.3) recomputed by `estimate_peak` with the actual numbers the code produces; the ten accuracy rows in a table with their source JSON paths; the observed minimum feasible budget per model; and a short "if you change the budget equation, update this file in the same commit" note (spec §11.1).

- [ ] **Step 8: Commit**

```bash
git add scripts/plan_accuracy.py tests/integration/test_plan_accuracy.py docs/memory_model.md
git commit -m "feat: plan-accuracy harness; M3 gate met within 10% on ten model/budget pairs"
```

---

## Milestone M4 — Calibrator with in-memory Hessian

**Gate:** matches reference GPTQ perplexity within noise on a small model, with a unit test asserting that the activation cache after a layer differs from the fp32-path result (i.e. §4.4d propagation is genuinely active).

### Task 12: Decoder-layer forward pass

**Files:**
- Create: `featherquant/model_fwd.py`
- Test: `tests/unit/test_model_fwd.py`, `tests/integration/test_forward_matches_llama_cli.py`

**Interfaces:**
- Consumes: nothing from the new modules (pure numpy).
- Produces:
  - `@dataclass(frozen=True) FwdConfig(hidden_size, intermediate_size, n_heads, n_kv_heads, head_dim, rms_eps, rope_theta)`; `FwdConfig.from_index(index: ModelIndex, rms_eps: float, rope_theta: float) -> FwdConfig`.
  - `@dataclass LayerWeights` with fp32 C-contiguous arrays `attn_norm (h,)`, `q (n_heads*head_dim, h)`, `k (n_kv*head_dim, h)`, `v (n_kv*head_dim, h)`, `o (h, n_heads*head_dim)`, `q_norm (head_dim,) | None`, `k_norm (head_dim,) | None`, `ffn_norm (h,)`, `gate (i, h)`, `up (i, h)`, `down (h, i)`.
  - `rms_norm(x, w, eps) -> np.ndarray`; `apply_rope(x, positions, theta) -> np.ndarray` for `x` shaped `(seq, n_heads, head_dim)`; `attention(x, lw, cfg) -> np.ndarray`; `mlp_intermediate(h, lw) -> np.ndarray` (the SwiGLU product, i.e. the `down_proj` input); `forward_layer(x, lw, cfg) -> np.ndarray` for one sample `(seq, hidden)` fp32.
  - `LINEAR_GROUPS: list[tuple[str, tuple[str, ...]]] = [("qkv", ("attn_q", "attn_k", "attn_v")), ("o", ("attn_o",)), ("gate_up", ("ffn_gate", "ffn_up")), ("down", ("ffn_down",))]` — the calibrator's sub-layer processing order (Task 16 depends on it).
  - `layer_input_for(group: str, x: np.ndarray, lw: LayerWeights, cfg: FwdConfig) -> np.ndarray` — the activation that is the *input* of the named linear group, computed on the fly so `down_proj`'s `n × s × i` input is never cached.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/unit/test_model_fwd.py
import numpy as np
import pytest

from featherquant.model_fwd import (
    FwdConfig,
    LayerWeights,
    apply_rope,
    attention,
    forward_layer,
    layer_input_for,
    mlp_intermediate,
    rms_norm,
)


def cfg():
    return FwdConfig(hidden_size=16, intermediate_size=32, n_heads=4,
                     n_kv_heads=2, head_dim=4, rms_eps=1e-6,
                     rope_theta=10000.0)


def weights(c, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    return LayerWeights(
        attn_norm=np.ones(c.hidden_size, np.float32),
        q=r(c.n_heads * c.head_dim, c.hidden_size),
        k=r(c.n_kv_heads * c.head_dim, c.hidden_size),
        v=r(c.n_kv_heads * c.head_dim, c.hidden_size),
        o=r(c.hidden_size, c.n_heads * c.head_dim),
        q_norm=np.ones(c.head_dim, np.float32),
        k_norm=np.ones(c.head_dim, np.float32),
        ffn_norm=np.ones(c.hidden_size, np.float32),
        gate=r(c.intermediate_size, c.hidden_size),
        up=r(c.intermediate_size, c.hidden_size),
        down=r(c.hidden_size, c.intermediate_size))


def test_rms_norm_matches_float64_reference():
    x = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    w = np.array([1.0, 2.0, 0.5, 1.5], np.float32)
    got = rms_norm(x, w, 1e-6)
    x64 = x.astype(np.float64)
    want = x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-6) * w
    assert np.allclose(got, want, atol=1e-5)


def test_rope_preserves_norm_and_is_position_dependent():
    x = np.random.default_rng(1).standard_normal((5, 2, 4)).astype(np.float32)
    y = apply_rope(x, np.arange(5), 10000.0)
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1),
                       atol=1e-5)
    assert np.allclose(y[0], x[0], atol=1e-6)     # position 0 is identity
    assert not np.allclose(y[3], x[3])


def test_attention_is_causal():
    c, lw = cfg(), weights(cfg())
    x = np.random.default_rng(2).standard_normal((6, c.hidden_size)).astype(np.float32)
    full = attention(x, lw, c)
    # Truncating the future must not change earlier positions.
    prefix = attention(x[:3], lw, c)
    assert np.allclose(full[:3], prefix, atol=1e-4)


def test_gqa_repeats_kv_heads():
    c = cfg()
    assert c.n_heads % c.n_kv_heads == 0
    lw = weights(c)
    x = np.zeros((2, c.hidden_size), np.float32)
    assert attention(x, lw, c).shape == (2, c.hidden_size)


def test_forward_layer_is_residual():
    c, lw = cfg(), weights(cfg())
    x = np.random.default_rng(3).standard_normal((4, c.hidden_size)).astype(np.float32)
    zero = LayerWeights(**{**lw.__dict__,
                           "o": np.zeros_like(lw.o),
                           "down": np.zeros_like(lw.down)})
    # With both output projections zeroed, the layer is the identity.
    assert np.allclose(forward_layer(x, zero, c), x, atol=1e-5)


def test_mlp_intermediate_feeds_down_proj():
    c, lw = cfg(), weights(cfg())
    h = np.random.default_rng(4).standard_normal((3, c.hidden_size)).astype(np.float32)
    inter = mlp_intermediate(h, lw)
    assert inter.shape == (3, c.intermediate_size)
    assert np.allclose(layer_input_for("down", h, lw, c), inter, atol=1e-6)


@pytest.mark.parametrize("group,width", [("qkv", 16), ("o", 16), ("gate_up", 16),
                                         ("down", 32)])
def test_layer_input_widths(group, width):
    c, lw = cfg(), weights(cfg())
    x = np.random.default_rng(5).standard_normal((3, c.hidden_size)).astype(np.float32)
    assert layer_input_for(group, x, lw, c).shape[-1] == width


def test_dtype_is_float32_everywhere():
    c, lw = cfg(), weights(cfg())
    x = np.zeros((2, c.hidden_size), np.float32)
    assert forward_layer(x, lw, c).dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_model_fwd.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.model_fwd'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/model_fwd.py
"""Decoder-layer forward pass in numpy, fp32 throughout.

The calibrator must push the activation cache through the *quantized*
layer (spec §4.4d), so FeatherQuant needs its own forward pass — one it
can run on a single layer, one sample at a time, inside a fixed buffer.
Shapes follow the Hugging Face layout: every weight is (out, in) and a
linear is x @ W.T.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

# Sub-layer processing order. Each group shares one input activation, so
# one Hessian at a time is resident (spec §3.3's worked example assumes
# exactly this).  Later groups depend on earlier groups already being
# quantized, which is why they are processed in sequence.
LINEAR_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("qkv", ("attn_q", "attn_k", "attn_v")),
    ("o", ("attn_o",)),
    ("gate_up", ("ffn_gate", "ffn_up")),
    ("down", ("ffn_down",)),
]


@dataclass(frozen=True)
class FwdConfig:
    """Architecture constants, all derived from config.json."""
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float

    @classmethod
    def from_index(cls, index: Any, rms_eps: float,
                   rope_theta: float) -> "FwdConfig":
        """Build from a ModelIndex plus the two config.json scalars."""
        return cls(hidden_size=index.hidden_size,
                   intermediate_size=index.intermediate_size,
                   n_heads=index.head_dims["n_heads"],
                   n_kv_heads=index.head_dims["n_kv_heads"],
                   head_dim=index.head_dims["head_dim"],
                   rms_eps=rms_eps, rope_theta=rope_theta)


@dataclass
class LayerWeights:
    """One decoder layer's fp32 weights. Reused buffers, filled per layer."""
    attn_norm: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    o: np.ndarray
    q_norm: np.ndarray | None
    k_norm: np.ndarray | None
    ffn_norm: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis, computed in fp32."""
    scale = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return ((x / scale) * w).astype(np.float32)


def apply_rope(x: np.ndarray, positions: np.ndarray,
               theta: float) -> np.ndarray:
    """Rotary embedding on (seq, heads, head_dim), half-split convention."""
    seq, heads, dim = x.shape
    if dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {dim}")
    half = dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                          axis=-1).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def attention(x: np.ndarray, lw: LayerWeights, cfg: FwdConfig) -> np.ndarray:
    """Causal GQA attention for one sample, returning (seq, hidden)."""
    seq = x.shape[0]
    xn = rms_norm(x, lw.attn_norm, cfg.rms_eps)
    q = (xn @ lw.q.T).reshape(seq, cfg.n_heads, cfg.head_dim)
    k = (xn @ lw.k.T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
    v = (xn @ lw.v.T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
    # Qwen3 normalises q and k per head before RoPE; other families omit it.
    if lw.q_norm is not None:
        q = rms_norm(q, lw.q_norm, cfg.rms_eps)
    if lw.k_norm is not None:
        k = rms_norm(k, lw.k_norm, cfg.rms_eps)
    pos = np.arange(seq)
    q = apply_rope(q, pos, cfg.rope_theta)
    k = apply_rope(k, pos, cfg.rope_theta)
    # GQA: each kv head serves n_heads // n_kv_heads query heads.
    rep = cfg.n_heads // cfg.n_kv_heads
    k = np.repeat(k, rep, axis=1)
    v = np.repeat(v, rep, axis=1)
    scale = np.float32(1.0 / np.sqrt(cfg.head_dim))
    # (heads, seq, seq); causal mask is upper-triangular -inf.
    scores = np.einsum("qhd,khd->hqk", q, k).astype(np.float32) * scale
    mask = np.triu(np.full((seq, seq), -np.inf, np.float32), 1)
    ctx = np.einsum("hqk,khd->qhd", _softmax(scores + mask), v)
    return (ctx.reshape(seq, cfg.n_heads * cfg.head_dim) @ lw.o.T).astype(np.float32)


def mlp_intermediate(h: np.ndarray, lw: LayerWeights) -> np.ndarray:
    """SwiGLU product — this is exactly the down_proj input."""
    g = h @ lw.gate.T
    # SiLU(g) = g * sigmoid(g), written to avoid overflow on large |g|.
    silu = g / (1.0 + np.exp(-np.clip(g, -60.0, 60.0)))
    return (silu * (h @ lw.up.T)).astype(np.float32)


def forward_layer(x: np.ndarray, lw: LayerWeights,
                  cfg: FwdConfig) -> np.ndarray:
    """One decoder layer: attention residual then MLP residual."""
    h = x + attention(x, lw, cfg)
    hn = rms_norm(h, lw.ffn_norm, cfg.rms_eps)
    return (h + mlp_intermediate(hn, lw) @ lw.down.T).astype(np.float32)


def layer_input_for(group: str, x: np.ndarray, lw: LayerWeights,
                    cfg: FwdConfig) -> np.ndarray:
    """The activation feeding a linear group, recomputed rather than cached.

    down_proj's input is n x s x intermediate_size — 1.6 GB at the spec's
    worked-example shape. Recomputing it per sample batch keeps the peak at
    one batch instead.
    """
    if group == "qkv":
        return rms_norm(x, lw.attn_norm, cfg.rms_eps)
    if group == "o":
        seq = x.shape[0]
        xn = rms_norm(x, lw.attn_norm, cfg.rms_eps)
        q = (xn @ lw.q.T).reshape(seq, cfg.n_heads, cfg.head_dim)
        k = (xn @ lw.k.T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
        v = (xn @ lw.v.T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
        if lw.q_norm is not None:
            q = rms_norm(q, lw.q_norm, cfg.rms_eps)
        if lw.k_norm is not None:
            k = rms_norm(k, lw.k_norm, cfg.rms_eps)
        pos = np.arange(seq)
        q, k = apply_rope(q, pos, cfg.rope_theta), apply_rope(k, pos, cfg.rope_theta)
        rep = cfg.n_heads // cfg.n_kv_heads
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        scale = np.float32(1.0 / np.sqrt(cfg.head_dim))
        scores = np.einsum("qhd,khd->hqk", q, k).astype(np.float32) * scale
        mask = np.triu(np.full((seq, seq), -np.inf, np.float32), 1)
        ctx = np.einsum("hqk,khd->qhd", _softmax(scores + mask), v)
        return ctx.reshape(seq, cfg.n_heads * cfg.head_dim).astype(np.float32)
    if group == "gate_up":
        return rms_norm(x + attention(x, lw, cfg), lw.ffn_norm, cfg.rms_eps)
    if group == "down":
        return mlp_intermediate(rms_norm(x + attention(x, lw, cfg),
                                         lw.ffn_norm, cfg.rms_eps), lw)
    raise ValueError(f"unknown linear group {group!r}: "
                     f"{[g for g, _ in LINEAR_GROUPS]}")
```

Note the duplication between `attention()` and `layer_input_for("o", ...)`: the `o` branch needs the pre-projection context, which `attention()` consumes internally. Keeping them separate avoids threading an out-parameter through the hot path; if a third caller appears, factor out `_attention_context(x, lw, cfg) -> np.ndarray` and have both call it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_model_fwd.py -v`
Expected: 11 passed.

- [ ] **Step 5: Write the end-to-end equivalence test**

```python
# tests/integration/test_forward_matches_llama_cli.py
"""The forward pass is only trustworthy if it agrees with llama.cpp.

Greedy decoding is a hard equality check: 20 argmax steps either match
token-for-token or the implementation is wrong somewhere (RoPE theta,
q_norm, GQA repeat order, mask).
"""
import os
import subprocess

import numpy as np
import pytest

from featherquant.calibrator import greedy_generate       # Task 16
from tests.memory.conftest import LLAMA_BIN

MODEL_DIR = os.path.expanduser("~/models/qwen3-0.6b")
MODEL_GGUF = os.path.expanduser("~/models/qwen3-0.6b-bf16.gguf")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(MODEL_DIR)
                       or not os.path.exists(f"{LLAMA_BIN}/llama-cli"),
                       reason="reference model or llama-cli unavailable"),
]


def test_greedy_decode_matches_llama_cli(tmp_path):
    prompt = "The capital of Singapore is"
    ours = greedy_generate(MODEL_DIR, prompt, n_tokens=20)
    r = subprocess.run(
        [f"{LLAMA_BIN}/llama-cli", "-m", MODEL_GGUF, "-p", prompt, "-n", "20",
         "--temp", "0", "--seed", "0", "-no-cnv"],
        capture_output=True, text=True, check=True)
    theirs = r.stdout
    # Compare the generated continuation, whitespace-normalised.
    assert " ".join(ours.split()) in " ".join(theirs.split()), \
        f"ours={ours!r}\ntheirs={theirs!r}"
```

- [ ] **Step 6: Leave it failing until Task 16, then run it**

Run: `.venv/bin/pytest tests/integration/test_forward_matches_llama_cli.py -v`
Expected now: FAIL on the `greedy_generate` import (Task 16 provides it). Mark this step done only when Task 16's Step 8 turns it green — it is the correctness anchor for the whole calibrator.

- [ ] **Step 7: Commit**

```bash
git add featherquant/model_fwd.py tests/unit/test_model_fwd.py \
        tests/integration/test_forward_matches_llama_cli.py
git commit -m "feat: numpy decoder-layer forward (RMSNorm, RoPE, GQA, SwiGLU) with per-group inputs"
```

---

### Task 13: Activation cache with disk spill

**Files:**
- Create: `featherquant/activations.py`
- Test: `tests/unit/test_activations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class ActivationCache(n_samples: int, seqlen: int, hidden: int, dtype: str = "fp16", spill_path: str | None = None)` with `nbytes: int` property, `write(i: int, x: np.ndarray) -> None` (accepts fp32, stores at the cache dtype), `read(i: int, out: np.ndarray | None = None) -> np.ndarray` (returns fp32; reuses `out` when given), `batches(batch: int = 1) -> Iterator[tuple[int, np.ndarray]]`, `close() -> None`, and context-manager support. Spill mode uses `seek`+`readinto`/`write` on a private file — never `mmap`. Consumed by `calibrator.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_activations.py
import numpy as np
import pytest

from featherquant.activations import ActivationCache


@pytest.mark.parametrize("spill", [False, True])
def test_roundtrip(tmp_path, spill):
    path = str(tmp_path / "cache.bin") if spill else None
    with ActivationCache(4, 8, 16, dtype="fp32", spill_path=path) as c:
        for i in range(4):
            c.write(i, np.full((8, 16), i, np.float32))
        for i in range(4):
            assert np.array_equal(c.read(i), np.full((8, 16), i, np.float32))


def test_fp16_storage_is_lossy_but_close(tmp_path):
    with ActivationCache(1, 4, 8, dtype="fp16") as c:
        x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
        c.write(0, x)
        assert np.allclose(c.read(0), x, atol=1e-3)
        assert c.nbytes == 1 * 4 * 8 * 2


def test_read_into_reused_buffer_does_not_allocate(tmp_path):
    with ActivationCache(2, 4, 8, dtype="fp32") as c:
        c.write(0, np.ones((4, 8), np.float32))
        out = np.empty((4, 8), np.float32)
        got = c.read(0, out=out)
        assert got is out and np.array_equal(out, np.ones((4, 8), np.float32))


def test_batches_cover_every_sample(tmp_path):
    with ActivationCache(5, 2, 4, dtype="fp32",
                         spill_path=str(tmp_path / "c.bin")) as c:
        for i in range(5):
            c.write(i, np.full((2, 4), i, np.float32))
        seen = [(i, x.copy()) for i, x in c.batches(batch=2)]
    assert [i for i, _ in seen] == [0, 2, 4]
    assert seen[0][1].shape == (2, 2, 4) and seen[-1][1].shape == (1, 2, 4)


def test_spill_mode_holds_no_array(tmp_path):
    c = ActivationCache(64, 128, 256, dtype="fp16",
                        spill_path=str(tmp_path / "c.bin"))
    assert c.resident_bytes < 1 << 20        # one batch buffer, not the cache
    assert c.nbytes == 64 * 128 * 256 * 2
    c.close()


def test_out_of_range_index_fails_loudly():
    with ActivationCache(2, 2, 2) as c:
        with pytest.raises(RuntimeError, match="sample index"):
            c.write(5, np.zeros((2, 2), np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_activations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.activations'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/activations.py
"""Activation cache: a ring of fixed buffers with optional disk spill.

Spec §3.3: the activation cache is a first-class budget consumer and the
most tunable term. Spilling is a logged approximation with a measured
runtime cost, never a hidden fallback — the caller records it in the run
manifest.
"""
import os
from typing import Iterator

import numpy as np

# Storage dtype by name; fp32 is exact, fp16 halves the cache.
_DTYPES = {"fp16": np.float16, "fp32": np.float32}


class ActivationCache:
    """n_samples x seqlen x hidden activations, resident or disk-backed."""

    def __init__(self, n_samples: int, seqlen: int, hidden: int,
                 dtype: str = "fp16", spill_path: str | None = None):
        try:
            self.dtype = _DTYPES[dtype]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown activation dtype {dtype!r}: {sorted(_DTYPES)}") from exc
        self.n_samples = n_samples
        self.seqlen = seqlen
        self.hidden = hidden
        self.itemsize = np.dtype(self.dtype).itemsize
        self.sample_bytes = seqlen * hidden * self.itemsize
        self.spill_path = spill_path
        self._f = None
        if spill_path is None:
            # Resident: one allocation up front, reused for the whole run.
            self._buf = np.zeros((n_samples, seqlen, hidden), self.dtype)
            self._scratch = np.empty((seqlen, hidden), np.float32)
        else:
            try:
                self._f = open(spill_path, "w+b")
                # Preallocate so a mid-run ENOSPC cannot corrupt the cache.
                self._f.truncate(n_samples * self.sample_bytes)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot create activation spill file {spill_path}: "
                    f"{exc}") from exc
            self._buf = None
            self._raw = bytearray(self.sample_bytes)
            self._scratch = np.empty((seqlen, hidden), np.float32)

    @property
    def nbytes(self) -> int:
        """Logical size of the whole cache."""
        return self.n_samples * self.sample_bytes

    @property
    def resident_bytes(self) -> int:
        """Bytes actually held in memory (the budget-relevant number)."""
        if self._buf is not None:
            return int(self._buf.nbytes) + int(self._scratch.nbytes)
        return len(self._raw) + int(self._scratch.nbytes)

    def _check(self, i: int) -> None:
        """Bounds check with an actionable message."""
        if not 0 <= i < self.n_samples:
            raise RuntimeError(f"sample index {i} out of range "
                               f"[0, {self.n_samples})")

    def write(self, i: int, x: np.ndarray) -> None:
        """Store one sample's activations (fp32 in, cache dtype out)."""
        self._check(i)
        if x.shape != (self.seqlen, self.hidden):
            raise RuntimeError(f"activation shape {x.shape} != "
                               f"{(self.seqlen, self.hidden)}")
        if self._buf is not None:
            self._buf[i] = x.astype(self.dtype, copy=False)
            return
        assert self._f is not None
        try:
            self._f.seek(i * self.sample_bytes)
            self._f.write(np.ascontiguousarray(x, self.dtype).tobytes())
        except OSError as exc:
            raise RuntimeError(f"spill write failed at sample {i}: {exc}") from exc

    def read(self, i: int, out: np.ndarray | None = None) -> np.ndarray:
        """Return one sample as fp32, into ``out`` when supplied."""
        self._check(i)
        dst = self._scratch if out is None else out
        if self._buf is not None:
            np.copyto(dst, self._buf[i], casting="unsafe")
            return dst
        assert self._f is not None
        try:
            self._f.seek(i * self.sample_bytes)
            got = self._f.readinto(memoryview(self._raw))
        except OSError as exc:
            raise RuntimeError(f"spill read failed at sample {i}: {exc}") from exc
        if got != self.sample_bytes:
            raise RuntimeError(f"short spill read at sample {i}: "
                               f"{got}/{self.sample_bytes} bytes")
        view = np.frombuffer(self._raw, self.dtype).reshape(self.seqlen,
                                                            self.hidden)
        np.copyto(dst, view, casting="unsafe")
        return dst

    def batches(self, batch: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (start_index, fp32 block) covering every sample once."""
        block = np.empty((batch, self.seqlen, self.hidden), np.float32)
        for start in range(0, self.n_samples, batch):
            n = min(batch, self.n_samples - start)
            for j in range(n):
                self.read(start + j, out=block[j])
            yield start, block[:n]

    def close(self) -> None:
        """Release the spill file and unlink it (it is scratch, not output)."""
        if self._f is not None:
            try:
                self._f.close()
                if self.spill_path and os.path.exists(self.spill_path):
                    os.unlink(self.spill_path)
            except OSError:
                pass        # best effort: a leftover scratch file is harmless
            self._f = None

    def __enter__(self) -> "ActivationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_activations.py -v && .venv/bin/mypy featherquant`
Expected: 7 passed, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add featherquant/activations.py tests/unit/test_activations.py
git commit -m "feat: activation cache with fixed buffers and optional disk spill"
```

---

### Task 14: In-memory Hessian

**Files:**
- Create: `featherquant/hessian.py`
- Test: `tests/unit/test_hessian.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class InMemoryHessian(d: int, dtype: str = "fp32")` implementing the Hessian protocol: `d: int`; `accumulate(x: np.ndarray) -> None` where `x` is `(n_rows, d)` fp32 (called once per sample batch); `n_seen: int`; `finalize(damp_percent: float = 0.01) -> None` (adds damping, inverts, Cholesky-factors, exposes the upper-triangular inverse factor); `row_block(j0: int, j1: int) -> np.ndarray` returning `Hinv[j0:j1, :]` shape `(j1 - j0, d)`; `dead_columns() -> np.ndarray` (bool mask of zero-variance inputs); `resident_bytes: int`; `close() -> None`.
  - `class DiagonalHessian(d, dtype="fp32")` — same protocol, `O(d)` memory, `row_block` materializes only the requested rows of a diagonal matrix.
  - `damped_inverse_cholesky(H: np.ndarray, damp_percent: float) -> np.ndarray` — the shared math, reused by `TiledHessian` in M5.
  Consumed by `gptq.py` (Task 15) and `calibrator.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hessian.py
import numpy as np
import pytest

from featherquant.hessian import (
    DiagonalHessian,
    InMemoryHessian,
    damped_inverse_cholesky,
)


def test_accumulate_matches_direct_gram():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 6)).astype(np.float32)
    h = InMemoryHessian(6)
    for start in range(0, 40, 7):          # arbitrary batching must not matter
        h.accumulate(x[start:start + 7])
    want = 2.0 * x.T @ x / 40
    assert np.allclose(h.matrix, want, atol=1e-4)
    assert h.n_seen == 40


def test_inverse_cholesky_is_upper_triangular_and_correct():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 5)).astype(np.float32)
    hmat = (a @ a.T + 5 * np.eye(5)).astype(np.float32)
    u = damped_inverse_cholesky(hmat, 0.01)
    assert np.allclose(u, np.triu(u), atol=1e-6)
    # U is the Cholesky factor of H^-1: U.T @ U ~= inv(H_damped)
    damp = 0.01 * np.mean(np.diag(hmat))
    assert np.allclose(u.T @ u, np.linalg.inv(hmat + damp * np.eye(5)),
                       atol=1e-3)


def test_row_block_matches_full_factor():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 8)).astype(np.float32)
    h = InMemoryHessian(8)
    h.accumulate(x)
    h.finalize()
    full = h.row_block(0, 8)
    assert np.allclose(h.row_block(2, 5), full[2:5], atol=1e-6)
    assert h.row_block(2, 5).shape == (3, 8)


def test_dead_columns_are_detected_and_damped():
    x = np.ones((10, 4), np.float32)
    x[:, 2] = 0.0                    # a channel that never activates
    h = InMemoryHessian(4)
    h.accumulate(x)
    h.finalize()
    assert h.dead_columns().tolist() == [False, False, True, False]
    assert np.isfinite(h.row_block(0, 4)).all()


def test_diagonal_hessian_uses_linear_memory():
    d = 4096
    h = DiagonalHessian(d)
    h.accumulate(np.ones((4, d), np.float32))
    h.finalize()
    assert h.resident_bytes < 64 * 1024      # O(d), not O(d^2)
    block = h.row_block(0, 2)
    assert block.shape == (2, d)
    assert np.count_nonzero(block[0]) == 1   # only the diagonal entry


def test_wrong_width_fails_loudly():
    h = InMemoryHessian(4)
    with pytest.raises(RuntimeError, match="expects 4 columns"):
        h.accumulate(np.zeros((2, 5), np.float32))


def test_row_block_before_finalize_fails_loudly():
    h = InMemoryHessian(4)
    h.accumulate(np.ones((2, 4), np.float32))
    with pytest.raises(RuntimeError, match="finalize"):
        h.row_block(0, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_hessian.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.hessian'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/hessian.py
"""Second-order statistics for GPTQ.

H = 2/n * sum_j x_j x_j^T over calibration activations, damped and
inverse-Cholesky factored. The factor U (upper triangular, U^T U = H^-1)
is what the GPTQ loop consumes; only its rows are ever handed out, so an
out-of-core implementation (M5) can satisfy the same protocol.
"""
import numpy as np

_DTYPES = {"fp32": np.float32, "bf16": np.float32}   # bf16 accum via fp32 math


def damped_inverse_cholesky(h: np.ndarray, damp_percent: float) -> np.ndarray:
    """Upper-triangular U with U^T U = inv(H + damp*I).

    Damping is a percentage of the mean diagonal, the standard GPTQ
    stabiliser: without it a rank-deficient H makes the inverse explode.
    """
    d = h.shape[0]
    diag = np.diag(h).copy()
    # Dead channels (never activated) would make H singular; give them a
    # unit diagonal so their columns quantize as plain RTN.
    dead = diag <= 0
    if dead.any():
        h = h.copy()
        h[dead, dead] = 1.0
        diag = np.diag(h).copy()
    damp = float(damp_percent * diag.mean())
    hd = h + damp * np.eye(d, dtype=h.dtype)
    try:
        inv = np.linalg.inv(hd)
        # np.linalg.cholesky returns lower L with L L^T = inv(H); GPTQ wants
        # the upper factor, so transpose.
        return np.linalg.cholesky(inv).T.astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Hessian factorisation failed (d={d}, damp={damp:.3e}): {exc}. "
            f"Raise --damp-percent rather than falling back to RTN") from exc


class InMemoryHessian:
    """Full d x d Hessian, fp32. The reference rung: no approximation."""

    def __init__(self, d: int, dtype: str = "fp32"):
        try:
            np_dtype = _DTYPES[dtype]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown stat precision {dtype!r}: {sorted(_DTYPES)}") from exc
        self.d = d
        self.stat_dtype = dtype
        self.matrix = np.zeros((d, d), np_dtype)
        self.n_seen = 0
        self._u: np.ndarray | None = None
        self._dead: np.ndarray | None = None

    @property
    def resident_bytes(self) -> int:
        """Bytes held: the matrix, plus the factor once finalized."""
        return int(self.matrix.nbytes) + (0 if self._u is None
                                          else int(self._u.nbytes))

    def accumulate(self, x: np.ndarray) -> None:
        """Fold one batch of rows (n, d) into the running Gram matrix."""
        if x.ndim != 2 or x.shape[1] != self.d:
            raise RuntimeError(f"hessian expects {self.d} columns, "
                               f"got shape {x.shape}")
        # syrk-style update; float32 matmul, accumulated in place.
        self.matrix += 2.0 * (x.T @ x)
        self.n_seen += x.shape[0]

    def finalize(self, damp_percent: float = 0.01) -> None:
        """Normalise by sample count, damp, invert and factor."""
        if self.n_seen == 0:
            raise RuntimeError("finalize() called before any accumulate()")
        self.matrix /= self.n_seen
        self._dead = np.diag(self.matrix) <= 0
        self._u = damped_inverse_cholesky(self.matrix, damp_percent)

    def dead_columns(self) -> np.ndarray:
        """Boolean mask of input channels with no observed activation."""
        if self._dead is None:
            raise RuntimeError("dead_columns() requires finalize() first")
        return self._dead

    def row_block(self, j0: int, j1: int) -> np.ndarray:
        """Rows [j0, j1) of the inverse-Cholesky factor, shape (j1-j0, d)."""
        if self._u is None:
            raise RuntimeError("row_block() requires finalize() first")
        return self._u[j0:j1]

    def close(self) -> None:
        """Release both matrices (the layer is done)."""
        self.matrix = np.zeros((0, 0), np.float32)
        self._u = None


class DiagonalHessian:
    """Diagonal-only rung: O(d) memory, degenerates toward scaled RTN."""

    def __init__(self, d: int, dtype: str = "fp32"):
        self.d = d
        self.stat_dtype = dtype
        self.diag = np.zeros(d, np.float32)
        self.n_seen = 0
        self._inv_sqrt: np.ndarray | None = None

    @property
    def resident_bytes(self) -> int:
        """Two length-d vectors at most."""
        return int(self.diag.nbytes) * (2 if self._inv_sqrt is not None else 1)

    def accumulate(self, x: np.ndarray) -> None:
        """Only the diagonal of x^T x is needed."""
        if x.ndim != 2 or x.shape[1] != self.d:
            raise RuntimeError(f"hessian expects {self.d} columns, "
                               f"got shape {x.shape}")
        self.diag += 2.0 * np.einsum("nd,nd->d", x, x)
        self.n_seen += x.shape[0]

    def finalize(self, damp_percent: float = 0.01) -> None:
        """U for a diagonal H is diag(1/sqrt(h_jj))."""
        if self.n_seen == 0:
            raise RuntimeError("finalize() called before any accumulate()")
        self.diag /= self.n_seen
        d = self.diag.copy()
        d[d <= 0] = 1.0
        damp = float(damp_percent * d.mean())
        self._inv_sqrt = (1.0 / np.sqrt(d + damp)).astype(np.float32)

    def dead_columns(self) -> np.ndarray:
        """Channels with no observed activation."""
        return self.diag <= 0

    def row_block(self, j0: int, j1: int) -> np.ndarray:
        """Materialise only the requested rows of a diagonal factor."""
        if self._inv_sqrt is None:
            raise RuntimeError("row_block() requires finalize() first")
        block = np.zeros((j1 - j0, self.d), np.float32)
        for i, j in enumerate(range(j0, j1)):
            block[i, j] = self._inv_sqrt[j]
        return block

    def close(self) -> None:
        """Release the vectors."""
        self.diag = np.zeros(0, np.float32)
        self._inv_sqrt = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_hessian.py -v`
Expected: 7 passed. `test_accumulate_matches_direct_gram` compares against `2 x^T x / n` — if it fails by exactly a factor, reconcile the normalisation with the GPTQ reference before touching the test.

- [ ] **Step 5: Commit**

```bash
git add featherquant/hessian.py tests/unit/test_hessian.py
git commit -m "feat: in-memory and diagonal Hessians with damped inverse-Cholesky factor"
```

---

### Task 15: GPTQ error compensation over ggml grids

**Files:**
- Create: `featherquant/gptq.py`
- Test: `tests/unit/test_gptq.py`

**Interfaces:**
- Consumes: `InMemoryHessian` / `DiagonalHessian` (Task 14); `quantize_q8_0`, `dequantize_q8_0` (existing `q8_0.py`); `GgmlLib.quantize_rows` (existing `ggml_backend.py`); `gguf.quants.dequantize`.
- Produces:
  - `QuantFn = Callable[[np.ndarray], tuple[bytes, np.ndarray]]` — takes `(rows, width)` fp32, returns `(packed_bytes, dequantized_fp32_of_the_same_shape)`.
  - `make_quant_fn(target_type: str, lib: GgmlLib | None) -> QuantFn`.
  - `gptq_quantize_row_group(w: np.ndarray, hess, quant_fn: QuantFn, group: int = 256) -> tuple[bytes, float]` — `w` is `(rows, d_in)` fp32 (one row group of output rows, upcast inside the fixed buffer, never the whole layer); returns packed bytes for those rows in ggml row-major order plus the mean squared reconstruction error. Rows are independent under GPTQ, which is what makes row-group streaming valid.
  - `rtn_quantize_row_group(w: np.ndarray, quant_fn: QuantFn) -> tuple[bytes, float]` — the no-compensation control used by tests and the RTN method.
  Consumed by `calibrator.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_gptq.py
import numpy as np
import pytest

from featherquant.gptq import gptq_quantize_row_group, make_quant_fn, rtn_quantize_row_group
from featherquant.hessian import InMemoryHessian
from featherquant.q8_0 import dequantize_q8_0


def _setup(d_in=256, rows=4, n=64, seed=0):
    """A linear layer plus activations with strongly anisotropic channels."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, d_in)).astype(np.float32)
    # Channel scales spanning two orders of magnitude: this is what makes
    # second-order information worth anything.
    scales = np.logspace(-1, 1, d_in).astype(np.float32)
    x = (rng.standard_normal((n, d_in)).astype(np.float32) * scales)
    return w, x


def test_gptq_beats_rtn_on_activation_weighted_error():
    w, x = _setup()
    h = InMemoryHessian(w.shape[1])
    h.accumulate(x)
    h.finalize()
    qfn = make_quant_fn("Q8_0", None)
    packed_g, _ = gptq_quantize_row_group(w, h, qfn)
    packed_r, _ = rtn_quantize_row_group(w, qfn)
    wq_g = dequantize_q8_0(packed_g).reshape(w.shape)
    wq_r = dequantize_q8_0(packed_r).reshape(w.shape)
    # The metric GPTQ optimises: ||(W - Wq) X^T||^2, not raw weight error.
    err_g = np.linalg.norm((w - wq_g) @ x.T)
    err_r = np.linalg.norm((w - wq_r) @ x.T)
    assert err_g < err_r, f"gptq {err_g:.4f} not better than rtn {err_r:.4f}"


def test_output_is_packed_size_exact():
    w, x = _setup(d_in=512, rows=3)
    h = InMemoryHessian(512)
    h.accumulate(x)
    h.finalize()
    packed, mse = gptq_quantize_row_group(w, h, make_quant_fn("Q8_0", None))
    assert len(packed) == 3 * (512 // 32) * 34
    assert mse > 0


def test_rows_are_independent():
    """Quantizing rows separately must give the same bytes as together."""
    w, x = _setup(d_in=256, rows=4)
    h = InMemoryHessian(256)
    h.accumulate(x)
    h.finalize()
    qfn = make_quant_fn("Q8_0", None)
    together, _ = gptq_quantize_row_group(w, h, qfn)
    apart = b"".join(gptq_quantize_row_group(w[i:i + 1], h, qfn)[0]
                     for i in range(4))
    assert together == apart


def test_deterministic():
    w, x = _setup()
    h = InMemoryHessian(w.shape[1])
    h.accumulate(x)
    h.finalize()
    qfn = make_quant_fn("Q8_0", None)
    a, _ = gptq_quantize_row_group(w, h, qfn)
    b, _ = gptq_quantize_row_group(w, h, qfn)
    assert a == b


def test_group_must_divide_row_length():
    w, x = _setup(d_in=300)
    h = InMemoryHessian(300)
    h.accumulate(x)
    h.finalize()
    with pytest.raises(RuntimeError, match="not a multiple"):
        gptq_quantize_row_group(w, h, make_quant_fn("Q8_0", None), group=256)


def test_identity_hessian_reduces_to_rtn():
    """With H = I the compensation term vanishes; output must match RTN."""
    w, _ = _setup(d_in=256, rows=2)
    h = InMemoryHessian(256)
    h.accumulate(np.eye(256, dtype=np.float32) * np.sqrt(0.5))
    h.finalize(damp_percent=0.0)
    qfn = make_quant_fn("Q8_0", None)
    g, _ = gptq_quantize_row_group(w, h, qfn, group=256)
    r, _ = rtn_quantize_row_group(w, qfn)
    assert g == r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_gptq.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.gptq'`

- [ ] **Step 3: Write the implementation**

```python
# featherquant/gptq.py
"""GPTQ error compensation on top of ggml quantization grids.

The quantization unit is a ggml block (32 for Q8_0, 256 for K-quants), so
compensation happens between blocks: each block is quantized jointly, its
reconstruction error is measured, and that error is pushed into the
not-yet-quantized columns through the inverse-Cholesky factor.

For K-quant targets this block-joint form is an approximation of textbook
per-column GPTQ (whose scalar grid does not exist inside a K superblock).
It is logged as the `kquant_group_joint` rung — never silently.

Output rows are independent: the update only ever moves mass along the
input dimension, so quantizing a row group at a time is exact, not an
approximation, and is what keeps the buffer fixed.
"""
from typing import Callable

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from .ggml_backend import GgmlLib
from .q8_0 import dequantize_q8_0, quantize_q8_0

# (rows, width) fp32 -> (packed bytes, dequantized fp32 of the same shape)
QuantFn = Callable[[np.ndarray], tuple[bytes, np.ndarray]]


def make_quant_fn(target_type: str, lib: GgmlLib | None) -> QuantFn:
    """Build the quantize+dequantize pair for a ggml type name."""
    try:
        ggml_type = GGMLQuantizationType[target_type]
    except KeyError as exc:
        raise RuntimeError(f"unknown ggml type {target_type!r}") from exc
    blk, _ = GGML_QUANT_SIZES[ggml_type]

    if ggml_type == GGMLQuantizationType.Q8_0:
        def q8(block: np.ndarray) -> tuple[bytes, np.ndarray]:
            """Q8_0 has a numpy kernel already byte-matched to llama.cpp."""
            flat = np.ascontiguousarray(block, np.float32).reshape(-1)
            packed = quantize_q8_0(flat)
            return packed, dequantize_q8_0(packed).reshape(block.shape)
        return q8

    from gguf.quants import dequantize as gguf_dequantize

    def kq(block: np.ndarray) -> tuple[bytes, np.ndarray]:
        """K-quants: ggml kernel for bytes, gguf for the round trip."""
        if lib is None:
            raise RuntimeError(f"{target_type} needs the ggml library; pass "
                               f"--ggml-lib or set $GGML_LIB")
        flat = np.ascontiguousarray(block, np.float32).reshape(-1)
        packed = lib.quantize_rows(flat, ggml_type, block.shape[-1])
        raw = np.frombuffer(packed, np.uint8)
        deq = gguf_dequantize(raw, ggml_type).reshape(block.shape)
        return packed, deq.astype(np.float32)

    if blk == 0:
        raise RuntimeError(f"{target_type} has no block size in GGML_QUANT_SIZES")
    return kq


def rtn_quantize_row_group(w: np.ndarray,
                           quant_fn: QuantFn) -> tuple[bytes, float]:
    """No compensation: the control arm and the RTN method's path."""
    packed, deq = quant_fn(w)
    return packed, float(np.mean((w - deq) ** 2))


def gptq_quantize_row_group(w: np.ndarray, hess: object, quant_fn: QuantFn,
                            group: int = 256) -> tuple[bytes, float]:
    """Quantize (rows, d_in) with GPTQ compensation between blocks.

    ``hess`` is any object exposing ``d``, ``row_block(j0, j1)`` and
    ``dead_columns()`` — InMemoryHessian, DiagonalHessian or TiledHessian.
    """
    if w.ndim != 2:
        raise RuntimeError(f"expected a 2-D row group, got shape {w.shape}")
    rows, d_in = w.shape
    if d_in % group:
        raise RuntimeError(f"row length {d_in} is not a multiple of the "
                           f"quantization group {group}; a misaligned group "
                           f"would emit garbage")
    if getattr(hess, "d", d_in) != d_in:
        raise RuntimeError(f"hessian width {getattr(hess, 'd', None)} does not "
                           f"match row length {d_in}")
    # Work on an owned fp32 copy: compensation mutates the not-yet-quantized
    # columns, and the caller's buffer must stay pristine for telemetry.
    work = np.array(w, np.float32, copy=True)
    dead = hess.dead_columns()          # type: ignore[attr-defined]
    if dead.any():
        work[:, dead] = 0.0             # never spend bits on unused channels
    chunks: list[bytes] = []
    sq_err = 0.0
    for j0 in range(0, d_in, group):
        j1 = j0 + group
        block = work[:, j0:j1]
        packed, deq = quant_fn(block)
        chunks.append(packed)
        err = block - deq               # (rows, group)
        sq_err += float(np.sum(err ** 2))
        if j1 >= d_in:
            break
        # U rows for this block: (group, d_in). The diagonal sub-block
        # rescales the error into the factor's coordinates; the right-hand
        # part carries it into the columns still to be quantized.
        u = hess.row_block(j0, j1)      # type: ignore[attr-defined]
        u_diag = u[:, j0:j1]
        try:
            # solve(u_diag.T, err.T).T == err @ inv(u_diag) for triangular u.
            scaled = np.linalg.solve(u_diag.T, err.T).T.astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"singular Hessian block at columns [{j0}, {j1}): {exc}") from exc
        work[:, j1:] -= scaled @ u[:, j1:]
        # Re-seat the just-quantized block so telemetry reflects reality.
        work[:, j0:j1] = deq
    packed_all = b"".join(chunks)
    if len(chunks) > 1:
        # Blocks were emitted in column order for the whole row group; ggml
        # expects row-major order (all blocks of row 0, then row 1, ...).
        packed_all = _reorder_blocks(chunks, rows)
    return packed_all, sq_err / (rows * d_in)


def _reorder_blocks(chunks: list[bytes], rows: int) -> bytes:
    """Column-major block chunks -> ggml's row-major packed layout."""
    per_row = [len(c) // rows for c in chunks]
    out = bytearray()
    for r in range(rows):
        for c, n in zip(chunks, per_row):
            out += c[r * n:(r + 1) * n]
    return bytes(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_gptq.py -v`
Expected: 6 passed.

`test_rows_are_independent` is the one that catches a `_reorder_blocks` bug — if it fails, the packed layout is wrong and every downstream file would load as garbage. Debug it before moving on. `test_identity_hessian_reduces_to_rtn` pins the degenerate case: no compensation when there is nothing to compensate.

- [ ] **Step 5: Commit**

```bash
git add featherquant/gptq.py tests/unit/test_gptq.py
git commit -m "feat: GPTQ block-wise error compensation over ggml grids, row-group bounded"
```

---

### Task 16: Sequential layer-wise calibrator

**Files:**
- Create: `featherquant/calibrator.py`
- Test: `tests/unit/test_calibrator.py`

**Interfaces:**
- Consumes: `Plan`, `CalibConfig` (Task 9); `ModelIndex` (Task 6); `ActivationCache` (Task 13); `InMemoryHessian`, `DiagonalHessian` (Task 14); `make_quant_fn`, `gptq_quantize_row_group`, `rtn_quantize_row_group` (Task 15); `FwdConfig`, `LayerWeights`, `LINEAR_GROUPS`, `forward_layer`, `layer_input_for`, `rms_norm` (Task 12); `SafetensorsSource` / `TensorSource`, `IncrementalWriter` (existing); `load_ggml` (existing).
- Produces:
  - `HESSIANS: dict[str, type] = {"full": InMemoryHessian, "diagonal": DiagonalHessian}` (M5 adds `"blocked"`, M6 adds `"lowrank"`).
  - `load_layer_weights(source, index, layer: int, cfg: FwdConfig, buffers: dict[str, np.ndarray]) -> LayerWeights` — fills pre-allocated fp32 buffers, allocates nothing per layer.
  - `tokenize_calibration(vocab_gguf: str, text_path: str, samples: int, seqlen: int, llama_bin: str) -> np.ndarray` shape `(samples, seqlen)` int32, via `llama-tokenize` so the tokenizer matches every perplexity number.
  - `run_calibration(plan: Plan, index: ModelIndex, dst: str, vocab_gguf: str, calib_text: str, ggml_lib: str | None = None, llama_bin: str = "", progress: ProgressFn | None = None) -> dict[str, Any]` — the spec §4.4 loop; returns a stats dict including `per_linear_mse: dict[str, float]`, `peak_rss`, `bytes_read`, `bytes_written`, `recompute_passes`.
  - `greedy_generate(model_dir: str, prompt: str, n_tokens: int, vocab_gguf: str | None = None, llama_bin: str = "") -> str` — fp32 forward over every layer, used by the M4 equivalence test.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_calibrator.py
"""The §4.4d test the spec demands, plus the loop's structural invariants."""
import numpy as np
import pytest

from featherquant.activations import ActivationCache
from featherquant.calibrator import HESSIANS, propagate_layer
from featherquant.gptq import make_quant_fn
from featherquant.hessian import InMemoryHessian
from featherquant.model_fwd import FwdConfig, LayerWeights, forward_layer


def cfg():
    return FwdConfig(hidden_size=32, intermediate_size=64, n_heads=4,
                     n_kv_heads=2, head_dim=8, rms_eps=1e-6, rope_theta=10000.0)


def weights(c, seed=0):
    rng = np.random.default_rng(seed)

    def r(*s):
        return rng.standard_normal(s).astype(np.float32) * 0.1

    return LayerWeights(
        attn_norm=np.ones(c.hidden_size, np.float32),
        q=r(c.n_heads * c.head_dim, c.hidden_size),
        k=r(c.n_kv_heads * c.head_dim, c.hidden_size),
        v=r(c.n_kv_heads * c.head_dim, c.hidden_size),
        o=r(c.hidden_size, c.n_heads * c.head_dim),
        q_norm=np.ones(c.head_dim, np.float32),
        k_norm=np.ones(c.head_dim, np.float32),
        ffn_norm=np.ones(c.hidden_size, np.float32),
        gate=r(c.intermediate_size, c.hidden_size),
        up=r(c.intermediate_size, c.hidden_size),
        down=r(c.hidden_size, c.intermediate_size))


def test_quantized_layer_propagation_is_active():
    """Spec §4.4d: the cache must carry quantization error forward.

    If this passes with equality, the calibrator is forwarding the ORIGINAL
    weights — a silent correctness bug that only shows up as mysteriously
    poor output quality.
    """
    c = cfg()
    lw = weights(c)
    rng = np.random.default_rng(1)
    with ActivationCache(2, 8, c.hidden_size, dtype="fp32") as cache:
        for i in range(2):
            cache.write(i, rng.standard_normal((8, c.hidden_size)).astype(np.float32))
        fp32_out = [forward_layer(cache.read(i).copy(), lw, c).copy()
                    for i in range(2)]
        quantized = propagate_layer(cache, lw, c, target_type="Q8_0",
                                    quant_fn=make_quant_fn("Q8_0", None),
                                    hessians=None)
        assert quantized is None or isinstance(quantized, dict)
        for i in range(2):
            got = cache.read(i)
            assert got.shape == fp32_out[i].shape
            assert not np.allclose(got, fp32_out[i], atol=1e-7), \
                "activation cache equals the fp32 path — §4.4d is not active"
            # ...but it must still be close: quantization error, not chaos.
            assert np.allclose(got, fp32_out[i], atol=0.5)


def test_hessian_registry_has_the_reference_rung():
    assert HESSIANS["full"] is InMemoryHessian
    assert "diagonal" in HESSIANS


def test_group_order_is_dependency_order():
    from featherquant.model_fwd import LINEAR_GROUPS
    assert [g for g, _ in LINEAR_GROUPS] == ["qkv", "o", "gate_up", "down"]


def test_hessian_width_matches_group_input():
    """qkv/o/gate_up see hidden_size; down sees intermediate_size."""
    c = cfg()
    lw = weights(c)
    from featherquant.model_fwd import layer_input_for
    x = np.zeros((4, c.hidden_size), np.float32)
    widths = {g: layer_input_for(g, x, lw, c).shape[-1]
              for g, _ in [("qkv", ()), ("o", ()), ("gate_up", ()), ("down", ())]}
    assert widths == {"qkv": 32, "o": 32, "gate_up": 32, "down": 64}


def test_propagate_rejects_shape_mismatch():
    c = cfg()
    lw = weights(c)
    with ActivationCache(1, 4, c.hidden_size + 8, dtype="fp32") as cache:
        with pytest.raises(RuntimeError, match="hidden size"):
            propagate_layer(cache, lw, c, target_type="Q8_0",
                            quant_fn=make_quant_fn("Q8_0", None), hessians=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_calibrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.calibrator'`

- [ ] **Step 3: Write the calibrator**

```python
# featherquant/calibrator.py
"""Sequential layer-wise calibration (spec §4.4) — the core of the project.

    1. Run embeddings over the calibration batch -> activation_cache
    2. For each layer L:
         a. Load L's weights into the fixed layer buffer
         b. Accumulate statistics for L from activation_cache
         c. Quantize L (delegates to gptq/quantizer)
         d. Forward activation_cache through the *quantized* L, in place
         e. Release L
    3. activation_cache now holds inputs for L+1

Step (d) uses the quantized layer. Propagating quantization error forward
is what makes sequential calibration work; using the original weights is a
silent correctness bug (see tests/unit/test_calibrator.py).

Statistics are gathered one linear group at a time (LINEAR_GROUPS), each
group re-running the forward-to-that-point over the cache. That costs four
recompute passes per layer and keeps exactly one Hessian resident, which
is what the budget equation assumes.
"""
import json
import os
import subprocess
from typing import Any

import numpy as np

from .activations import ActivationCache
from .events import Phase, ProgressFn, TensorDone, TensorStart
from .ggml_backend import load_ggml
from .gptq import QuantFn, gptq_quantize_row_group, make_quant_fn, rtn_quantize_row_group
from .hessian import DiagonalHessian, InMemoryHessian
from .indexer import ModelIndex
from .model_fwd import (
    LINEAR_GROUPS,
    FwdConfig,
    LayerWeights,
    forward_layer,
    layer_input_for,
)
from .planner import Plan
from .roles import Role

# Hessian rung name -> implementation. M5 registers "blocked".
HESSIANS: dict[str, type] = {"full": InMemoryHessian,
                             "diagonal": DiagonalHessian}

# Role -> the LayerWeights attribute it fills.
_ROLE_ATTR = {Role.ATTN_Q.value: "q", Role.ATTN_K.value: "k",
              Role.ATTN_V.value: "v", Role.ATTN_O.value: "o",
              Role.FFN_GATE.value: "gate", Role.FFN_UP.value: "up",
              Role.FFN_DOWN.value: "down"}

# Sample batch for statistics and propagation. One sample keeps the
# attention scratch term at heads*s*s*4, exactly what the planner assumed.
BATCH = 1


def tokenize_calibration(vocab_gguf: str, text_path: str, samples: int,
                         seqlen: int, llama_bin: str) -> np.ndarray:
    """Token ids for the calibration set, via llama-tokenize.

    Using llama.cpp's tokenizer means calibration, perplexity and the
    reference baselines all share one tokenizer — spec §6 requires stating
    it, and sharing it is the only way the numbers compare.
    """
    exe = os.path.join(llama_bin, "llama-tokenize")
    try:
        out = subprocess.run([exe, "-m", vocab_gguf, "-f", text_path,
                              "--ids"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"llama-tokenize failed ({exe}): {exc}") from exc
    try:
        ids = np.array(json.loads(out.strip().splitlines()[-1]), np.int32)
    except (ValueError, json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(f"cannot parse llama-tokenize output: {exc}") from exc
    need = samples * seqlen
    if ids.size < need:
        raise RuntimeError(
            f"calibration corpus has {ids.size} tokens, need {need} "
            f"({samples} x {seqlen}) — use a longer corpus rather than "
            f"reducing the sample count silently")
    return ids[:need].reshape(samples, seqlen)


def load_layer_weights(source: Any, index: ModelIndex, layer: int,
                       cfg: FwdConfig,
                       buffers: dict[str, np.ndarray]) -> LayerWeights:
    """Fill the pre-allocated fp32 layer buffers from disk.

    ``buffers`` is allocated once by the caller (one entry per attribute)
    and reused for every layer — no allocation inside the layer loop.
    """
    attrs: dict[str, Any] = {"q_norm": None, "k_norm": None}
    for t in index.layer_tensors(layer):
        if t.role == Role.NORM.value:
            key = ("q_norm" if "q_norm" in t.name else
                   "k_norm" if "k_norm" in t.name else
                   "attn_norm" if "input_layernorm" in t.name
                   or "attn_norm" in t.name else "ffn_norm")
        else:
            key = _ROLE_ATTR[t.role]
        buf = buffers[key]
        rows = t.shape[0] if len(t.shape) >= 2 else 1
        # Read the whole tensor in row groups into its fixed buffer.
        flat = buf.reshape(-1)
        per_row = int(np.prod(t.shape[1:])) if len(t.shape) >= 2 else t.shape[0]
        scratch = bytearray(per_row * 8)
        for r in range(rows):
            x = source.read_rows_f32(_SourceView(t), r, 1, scratch)
            flat[r * per_row:(r + 1) * per_row] = x[:per_row]
        attrs[key] = buf.reshape(t.shape) if len(t.shape) >= 2 else buf[:t.shape[0]]
    missing = {"attn_norm", "q", "k", "v", "o", "ffn_norm", "gate", "up", "down"} - attrs.keys()
    if missing:
        raise RuntimeError(f"layer {layer} is missing tensors for {sorted(missing)}")
    return LayerWeights(**attrs)   # type: ignore[arg-type]


class _SourceView:
    """Adapts a TensorInfo to the reader's (shape in ne-order, tensor_type,
    data_offset) expectations, so the existing pread path is reused."""

    def __init__(self, t: Any) -> None:
        from gguf import GGMLQuantizationType
        self.name = t.name
        self.shape = tuple(reversed(t.shape))
        self.tensor_type = GGMLQuantizationType[t.dtype]
        self.data_offset = t.byte_offset
        self.n_elements = int(np.prod(t.shape))
        self.n_bytes = t.byte_length


def propagate_layer(cache: ActivationCache, lw: LayerWeights, cfg: FwdConfig,
                    target_type: str, quant_fn: QuantFn,
                    hessians: dict[str, Any] | None) -> dict[str, float] | None:
    """Quantize the layer in place and push the cache through it (§4.4 b-d).

    ``hessians`` maps group name -> a finalized Hessian; None means RTN
    (no statistics, no compensation) and is what the §4.4d test uses.
    Returns per-attribute reconstruction MSE, or None for the RTN path.
    """
    if cache.hidden != cfg.hidden_size:
        raise RuntimeError(f"cache hidden size {cache.hidden} != model "
                           f"hidden size {cfg.hidden_size}")
    mse: dict[str, float] = {}
    for group, roles in LINEAR_GROUPS:
        for role in roles:
            attr = _ROLE_ATTR[role]
            w = getattr(lw, attr)
            hess = None if hessians is None else hessians.get(group)
            if hess is None:
                packed, err = rtn_quantize_row_group(w, quant_fn)
            else:
                packed, err = gptq_quantize_row_group(w, hess, quant_fn)
            mse[attr] = err
            # Replace the layer's weights with their quantized values so the
            # forward below sees exactly what the output file contains.
            _, deq = quant_fn(w)
            if hess is not None:
                # Re-derive the compensated reconstruction from the packed
                # bytes so cache and file agree bit for bit.
                deq = _dequantize_packed(packed, target_type, w.shape)
            np.copyto(w, deq)
            _LAST_PACKED[attr] = packed
    for start, block in cache.batches(batch=BATCH):
        for j in range(block.shape[0]):
            cache.write(start + j, forward_layer(block[j], lw, cfg))
    return mse if hessians is not None else None


# Packed bytes of the most recently quantized attribute, handed to the
# writer by run_calibration. Module-level to keep propagate_layer's
# signature honest about what it returns (statistics, not bytes).
_LAST_PACKED: dict[str, bytes] = {}


def _dequantize_packed(packed: bytes, target_type: str,
                       shape: tuple[int, ...]) -> np.ndarray:
    """Round-trip packed bytes back to fp32 in the tensor's shape."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    if target_type == "Q8_0":
        from .q8_0 import dequantize_q8_0
        return dequantize_q8_0(packed).reshape(shape).astype(np.float32)
    raw = np.frombuffer(packed, np.uint8)
    return dequantize(raw, GGMLQuantizationType[target_type]).reshape(
        shape).astype(np.float32)
```

`run_calibration` and `greedy_generate` complete the module:

```python
def run_calibration(plan: Plan, index: ModelIndex, dst: str, vocab_gguf: str,
                    calib_text: str, ggml_lib: str | None = None,
                    llama_bin: str = "", rms_eps: float = 1e-6,
                    rope_theta: float = 1000000.0,
                    progress: ProgressFn | None = None) -> dict[str, Any]:
    """Execute a calibrated plan end to end, writing a GGUF at ``dst``."""
    from .engine import RESERVE, rss_bytes                 # local: cycle-free
    from .gguf_io import IncrementalWriter
    from .st_source import SafetensorsSource

    cfg = FwdConfig.from_index(index, rms_eps, rope_theta)
    source = SafetensorsSource(os.path.dirname(index.tensors[0].shard_path),
                               vocab_gguf)
    lib = load_ggml(ggml_lib) if plan.fmt != "q8_0" else None
    quant_fns: dict[str, QuantFn] = {}
    stats: dict[str, Any] = {"per_linear_mse": {}, "peak_rss": rss_bytes(),
                             "bytes_read": 0, "bytes_written": 0,
                             "recompute_passes": len(LINEAR_GROUPS)}
    tokens = tokenize_calibration(vocab_gguf, calib_text, plan.calib.samples,
                                  plan.calib.seqlen, llama_bin)
    if progress is not None:
        progress(Phase(f"tokenized {tokens.shape[0]}x{tokens.shape[1]} "
                       f"calibration tokens"))

    spill = (os.path.join(os.path.dirname(dst) or ".", "fq_activations.bin")
             if plan.calib.spill else None)
    hess_cls = HESSIANS[plan.hessian_approx]
    writer = IncrementalWriter(dst, source.reader, _file_type(plan.fmt))
    # Declaration order == calibration order, so data streams straight out.
    order = _output_order(index, plan)
    for tp in order:
        writer.add_tensor_info(tp.name, tuple(reversed(tp.shape)),
                               tp.output_bytes, tp.ggml_type)
    writer.begin_data()

    with ActivationCache(plan.calib.samples, plan.calib.seqlen,
                         index.hidden_size, dtype=plan.calib.act_dtype,
                         spill_path=spill) as cache:
        _embed(source, index, tokens, cache, writer, plan, progress)
        buffers = _allocate_layer_buffers(index, cfg)
        for layer in plan.layer_order:
            if progress is not None:
                progress(Phase(f"layer {layer}: statistics"))
            lw = load_layer_weights(source, index, layer, cfg, buffers)
            hessians = {}
            for group, _roles in LINEAR_GROUPS:
                width = layer_input_for(group, np.zeros(
                    (1, index.hidden_size), np.float32), lw, cfg).shape[-1]
                h = hess_cls(width, dtype=plan.calib.act_dtype)
                for start, block in cache.batches(batch=BATCH):
                    for j in range(block.shape[0]):
                        h.accumulate(layer_input_for(group, block[j], lw, cfg))
                h.finalize()
                hessians[group] = h
                stats["peak_rss"] = max(stats["peak_rss"], rss_bytes())
            if progress is not None:
                progress(Phase(f"layer {layer}: quantize + propagate"))
            mse = propagate_layer(cache, lw, cfg, _target_of(plan, layer),
                                  _quant_fn(quant_fns, plan, layer, lib),
                                  hessians)
            for group_h in hessians.values():
                group_h.close()             # release before the next layer
            _write_layer(writer, index, layer, plan, stats, progress)
            stats["per_linear_mse"].update(
                {f"{layer}.{k}": v for k, v in (mse or {}).items()})
            stats["peak_rss"] = max(stats["peak_rss"], rss_bytes())
        _write_tail(source, index, writer, plan, stats)
    writer.close()
    source.close()
    return stats
```

The helpers `_file_type`, `_output_order`, `_target_of`, `_quant_fn`, `_embed`, `_allocate_layer_buffers`, `_write_layer` and `_write_tail` are small and mechanical; implement each against the existing engine's equivalents:

- `_file_type(fmt)` → `FORMATS[fmt].file_type`.
- `_output_order(index, plan)` → embed first, then for each layer the norms followed by `LINEAR_GROUPS` order, then final norm and output; each entry carries `name`, `shape`, `ggml_type` and `output_bytes` from the matching `TensorPlan` (compute `output_bytes` with `engine.packed_nbytes`).
- `_target_of(plan, layer)` / `_quant_fn(cache, plan, layer, lib)` → look up the `TensorPlan.target_type` and memoize `make_quant_fn(target_type, lib)` per type.
- `_embed` → for each sample, read the embedding rows for its token ids through `source.read_rows_f32` into the fixed buffer and `cache.write(i, rows)`; then quantize and write the embedding tensor itself in row groups (it is `largest_tensor_bytes`, so it must be row-grouped, spec §3.3).
- `_allocate_layer_buffers(index, cfg)` → one fp32 `np.empty` per attribute, sized from the layer-0 tensor shapes; allocated once, reused for every layer.
- `_write_layer` → `writer.begin_tensor()` then `writer.write(_LAST_PACKED[attr])` in declaration order, plus verbatim norm copies.
- `_write_tail` → final norm and `output.weight` (row-grouped, RTN — no calibration statistics exist past the last layer).

```python
def greedy_generate(model_dir: str, prompt: str, n_tokens: int,
                    vocab_gguf: str | None = None, llama_bin: str = "") -> str:
    """fp32 forward over every layer, argmax decoding. Correctness anchor."""
    from .indexer import index_model
    from .st_source import SafetensorsSource

    index = index_model(model_dir)
    vocab_gguf = vocab_gguf or os.path.join(model_dir, "vocab.gguf")
    source = SafetensorsSource(model_dir, vocab_gguf)
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read config.json: {exc}") from exc
    cfg = FwdConfig.from_index(index, float(raw.get("rms_norm_eps", 1e-6)),
                               float(raw.get("rope_theta", 1000000.0)))
    ids = list(_tokenize_prompt(vocab_gguf, prompt, llama_bin))
    buffers = _allocate_layer_buffers(index, cfg)
    for _ in range(n_tokens):
        x = _embed_ids(source, index, np.array(ids, np.int32))
        for layer in range(index.n_layers):
            lw = load_layer_weights(source, index, layer, cfg, buffers)
            x = forward_layer(x, lw, cfg)
        logits = _final_logits(source, index, x, cfg)
        ids.append(int(np.argmax(logits[-1])))
    source.close()
    return _detokenize(vocab_gguf, ids[-n_tokens:])
```

`_tokenize_prompt`, `_embed_ids`, `_final_logits` (final RMSNorm then `lm_head`/tied embedding matmul, computed in vocab row groups so the `v × h` matrix is never fully resident) and `_detokenize` (map ids through `tokenizer.ggml.tokens` from the vocab GGUF, translating the byte-level markers `Ġ`→space and `Ċ`→newline) round the module out.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_calibrator.py -v`
Expected: 5 passed. `test_quantized_layer_propagation_is_active` is the spec-mandated one — if it ever starts passing trivially (because `propagate_layer` stopped mutating the cache), the calibrator is broken even though everything else is green.

- [ ] **Step 5: Wire `--method gptq` into `featherquant run`**

In `_cmd_run` (Task 10), branch on `plan.method`:

```python
    if plan.method == "gptq":
        stats = run_calibration(plan, ModelIndex.load(plan.index_path),
                                a.output, a.vocab_gguf, a.calib_text,
                                ggml_lib=a.ggml_lib, llama_bin=a.llama_bin,
                                progress=reporter)
    else:
        stats = quantize_model(...)     # the existing RTN path
```

Add `--calib-text` (required when the plan's method is `gptq`) and `--llama-bin` (default `$LLAMA_BIN`) to the `run` parser, and write the run manifest from `stats` exactly as the RTN path does.

- [ ] **Step 6: Run the forward-equivalence integration test from Task 12**

Run: `.venv/bin/pytest tests/integration/test_forward_matches_llama_cli.py -v -m slow`
Expected: PASS — our greedy decode matches `llama-cli`'s. If it does not, the bug is in `model_fwd.py`, not the calibrator: check `rope_theta` (Qwen3 uses 1e6, not 1e4), the `q_norm`/`k_norm` application order (before RoPE), and the GQA `np.repeat` axis. Go back and fix Task 12 before continuing — everything downstream inherits this error.

- [ ] **Step 7: Full suite and commit**

Run: `.venv/bin/pytest -q -m "not slow" && .venv/bin/ruff check featherquant tests && .venv/bin/mypy featherquant`

```bash
git add featherquant/calibrator.py featherquant/cli.py tests/unit/test_calibrator.py
git commit -m "feat: sequential layer-wise calibrator with quantized-forward propagation"
```

---

### Task 17: M4 gate — GPTQ quality vs the reference

**Files:**
- Create: `bench/harness/eval_quality.sh`
- Modify: `docs/approximation_costs.md` (the `hessian_full` row's source), `docs/baselines.md`

**Interfaces:**
- Consumes: `featherquant plan`/`run` (Task 10, 16); `bench/harness/run_baseline.sh` (Task 2); `bench/manifests/m0_gptq_reference.json` (Task 4).
- Produces: `bench/harness/eval_quality.sh GGUF RUN_MANIFEST` — runs `llama-perplexity` and `llama-cli`, writes `quality.ppl`, `quality.ppl_dataset` and a `coherent_generation` note into the run manifest; fails if generation is empty or degenerate.

- [ ] **Step 1: Write the quality-evaluation harness**

```bash
#!/usr/bin/env bash
# bench/harness/eval_quality.sh — perplexity + coherence into a run manifest.
# Usage: eval_quality.sh MODEL.gguf RUN_MANIFEST.json [CORPUS] [CTX]
# A file that loads is not a file that works (spec §11.4): a run whose
# generation is empty or degenerate fails here and its numbers do not
# enter any table.
set -euo pipefail
GGUF=$1; MANIFEST=$2
CORPUS=${3:-bench/data/wiki.test.raw}; CTX=${4:-512}
LC=${LLAMA_BIN:-$HOME/llama.cpp/build-cpu/bin}

PPL_OUT=$(mktemp)
"$LC/llama-perplexity" -m "$GGUF" -f "$CORPUS" -c "$CTX" 2>&1 | tee "$PPL_OUT"
GEN=$("$LC/llama-cli" -m "$GGUF" -p "The capital of Singapore is" -n 24 \
        --temp 0 -no-cnv 2>/dev/null)

python - "$MANIFEST" "$PPL_OUT" "$CORPUS" "$CTX" "$GEN" <<'PY'
import re
import sys

from featherquant.run_manifest import RunManifest

manifest, ppl_out, corpus, ctx, gen = sys.argv[1:6]
hits = re.findall(r"Final estimate: PPL = ([\d.]+)", open(ppl_out).read())
if not hits:
    sys.exit("no perplexity in llama-perplexity output — refusing to record")
words = gen.split()
if len(words) < 6 or len(set(words)) < 4:
    sys.exit(f"generation is degenerate ({gen!r}) — refusing to record a "
             f"quality number for a model that does not work")
m = RunManifest.load(manifest)
m.quality = {"ppl": float(hits[-1]),
             "ppl_dataset": f"{corpus} c={ctx} tokenizer=qwen3",
             "tasks": {"coherent_generation": gen.strip()[:200]}}
m.save(manifest)
print(f"ppl={hits[-1]} recorded in {manifest}")
PY
rm -f "$PPL_OUT"
```

- [ ] **Step 2: Plan and run calibrated Q8_0 on Qwen3-0.6B**

```bash
.venv/bin/featherquant index ~/models/qwen3-0.6b -o /tmp/idx_0.6b.json
.venv/bin/featherquant plan /tmp/idx_0.6b.json --budget 8GiB --method gptq \
  --format q8_0 --calib-samples 128 --calib-seqlen 512 -o /tmp/plan_0.6b_gptq.json
bash bench/harness/run_baseline.sh \
  "[\".venv/bin/featherquant\",\"run\",\"/tmp/plan_0.6b_gptq.json\",\"-o\",\"/tmp/m4_gptq_q8_0.gguf\",\"--vocab-gguf\",\"$HOME/models/qwen3-0.6b-vocab.gguf\",\"--calib-text\",\"bench/data/wiki.test.raw\",\"--ui\",\"none\"]" \
  m4_gptq_q8_0 gptq_q8_0 Qwen/Qwen3-0.6B /tmp/m4_gptq_q8_0.gguf nvme 8589934592
bash bench/harness/eval_quality.sh /tmp/m4_gptq_q8_0.gguf \
  bench/manifests/m4_gptq_q8_0.json
```

- [ ] **Step 3: Run the RTN control at the same format**

```bash
bash bench/harness/run_baseline.sh \
  "[\".venv/bin/featherquant\",\"--model\",\"$HOME/models/qwen3-0.6b-bf16.gguf\",\"--output\",\"/tmp/m4_rtn_q8_0.gguf\",\"--format\",\"q8_0\",\"--max-ram\",\"8GB\",\"--ui\",\"none\"]" \
  m4_rtn_q8_0 rtn_q8_0 Qwen/Qwen3-0.6B /tmp/m4_rtn_q8_0.gguf nvme 8589934592
bash bench/harness/eval_quality.sh /tmp/m4_rtn_q8_0.gguf bench/manifests/m4_rtn_q8_0.json
```

- [ ] **Step 4: Repeat at Q4_K_M, where calibration should actually pay**

Same two commands with `--format q4_k_m` and run ids `m4_gptq_q4_k_m` / `m4_rtn_q4_k_m`. Q8_0 is nearly lossless, so a GPTQ-vs-RTN gap there would be noise; the 4-bit pair is the informative comparison.

- [ ] **Step 5: Compare against the reference GPTQ manifest**

Read `bench/manifests/m0_gptq_reference.json` and the four M4 manifests. The gate is: FeatherQuant's `gptq_q4_k_m` perplexity is within noise of the reference GPTQ perplexity, and below `rtn_q4_k_m`. Estimate "noise" by running `m4_gptq_q4_k_m` twice with different calibration slices and taking the spread — do not assume a number.

- [ ] **Step 6: Record the result honestly**

Write the comparison into `docs/baselines.md` under an "M4 gate" heading, with all five run ids. **If GPTQ does not beat RTN, or does not reach the reference, write that down and stop.** Spec §11.5: when a result contradicts the thesis, report the result; do not adjust the experiment until it agrees. Then investigate as a bug, with `tests/unit/test_gptq.py::test_gptq_beats_rtn_on_activation_weighted_error` and the §4.4d test as the first two things to re-check.

- [ ] **Step 7: Fill the `hessian_full` row's source**

In `docs/approximation_costs.md`, set the `hessian_full` row's `source` to `m4_gptq_q4_k_m.json`. It stays the 0/0/0 reference row — every other rung is measured as a delta against it.

- [ ] **Step 8: Commit**

```bash
git add bench/harness/eval_quality.sh bench/manifests docs/baselines.md \
        docs/approximation_costs.md
git commit -m "feat: quality-eval harness; M4 gate measured against reference GPTQ"
```

---

## Milestone M5 — Out-of-core Hessian

**Gate:** same quality as M4, inside a budget where M4 is OOM-killed. **This is the project's central claim.**

### Task 18: Blocked disk-backed Hessian with panel Cholesky

**Files:**
- Modify: `featherquant/hessian.py`, `featherquant/calibrator.py` (register the rung), `featherquant/planner.py` (`HESSIAN_FRACTION["blocked"]` already present — verify it against the measured panel size)
- Test: `tests/unit/test_tiled_hessian.py`, `tests/memory/test_ooc_hessian_under_ceiling.py`

**The math this task implements** (state it in the module docstring; the equality test against `InMemoryHessian` is the arbiter):

GPTQ consumes `U`, upper triangular with `Uᵀ U = H⁻¹`. Equivalently `H = V Vᵀ` with `V = U⁻¹` upper — the *reverse* (UL) Cholesky of `H`. With `J` the exchange matrix (index reversal), `V = J · chol_lower(J H J) · J`. So:

1. Accumulate `H` into disk tiles, one panel of rows at a time.
2. Blocked lower Cholesky of the reversed matrix `J H J`, right-looking, one column panel resident → gives `V` implicitly.
3. `row_block(j0, j1)` returns rows of `U = V⁻¹` by blocked back-substitution against `V`, holding one panel.

Peak resident memory is `panel × d × 4` bytes, not `d² × 4`.

**Interfaces:**
- Consumes: `damped_inverse_cholesky` (Task 14, used by the small-matrix test path).
- Produces: `class TiledHessian(d: int, panel: int, work_dir: str, dtype: str = "fp32")` satisfying the same protocol as `InMemoryHessian` — `accumulate(x)`, `finalize(damp_percent)`, `row_block(j0, j1)`, `dead_columns()`, `resident_bytes`, `close()` (deletes its scratch files) — plus `panel_bytes: int` and `disk_bytes: int` for the manifest. Registered as `HESSIANS["blocked"]`.

Accumulation note: `accumulate(x)` cannot hold `H` in memory, so it buffers each batch's contribution panel by panel — the calibrator calls it once per sample batch per group, and `TiledHessian` writes `H[p0:p1, :] += 2 · x[:, p0:p1]ᵀ x` with one panel buffer resident, reading and writing that panel's tile each time. That is `d / panel` read-modify-write passes over the tile file per batch; the runtime cost is real and must be recorded, not hidden.

- [ ] **Step 1: Write the failing equality test**

```python
# tests/unit/test_tiled_hessian.py
"""The tiled Hessian must agree with the in-memory one. Full stop.

M5's claim is "same quality, less memory". If row_block() diverges from
the reference beyond float tolerance, the claim is false regardless of
what the perplexity table says.
"""
import numpy as np
import pytest

from featherquant.hessian import InMemoryHessian, TiledHessian


@pytest.mark.parametrize("d,panel", [(64, 16), (64, 64), (96, 32), (100, 32)])
def test_row_block_matches_in_memory(tmp_path, d, panel):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, d)).astype(np.float32)
    ref = InMemoryHessian(d)
    tiled = TiledHessian(d, panel=panel, work_dir=str(tmp_path))
    for start in range(0, 256, 32):
        batch = x[start:start + 32]
        ref.accumulate(batch)
        tiled.accumulate(batch)
    ref.finalize()
    tiled.finalize()
    for j0 in range(0, d, panel):
        j1 = min(j0 + panel, d)
        assert np.allclose(tiled.row_block(j0, j1), ref.row_block(j0, j1),
                           atol=1e-3, rtol=1e-3), f"rows [{j0},{j1}) diverge"
    tiled.close()


def test_resident_memory_is_panel_sized(tmp_path):
    d, panel = 2048, 64
    h = TiledHessian(d, panel=panel, work_dir=str(tmp_path))
    h.accumulate(np.ones((4, d), np.float32))
    h.finalize()
    assert h.resident_bytes < 4 * panel * d * 4
    assert h.resident_bytes < d * d * 4 // 4      # decisively sub-quadratic
    assert h.disk_bytes >= d * d * 4
    h.close()


def test_close_removes_scratch_files(tmp_path):
    h = TiledHessian(32, panel=8, work_dir=str(tmp_path))
    h.accumulate(np.ones((2, 32), np.float32))
    h.finalize()
    h.close()
    assert list(tmp_path.iterdir()) == []


def test_gptq_output_matches_between_hessian_backends(tmp_path):
    """End-to-end: same packed bytes, whichever Hessian produced the factor."""
    from featherquant.gptq import gptq_quantize_row_group, make_quant_fn
    rng = np.random.default_rng(1)
    d = 256
    x = rng.standard_normal((128, d)).astype(np.float32)
    w = rng.standard_normal((4, d)).astype(np.float32)
    ref, tiled = InMemoryHessian(d), TiledHessian(d, 64, str(tmp_path))
    for h in (ref, tiled):
        h.accumulate(x)
        h.finalize()
    qfn = make_quant_fn("Q8_0", None)
    a, _ = gptq_quantize_row_group(w, ref, qfn)
    b, _ = gptq_quantize_row_group(w, tiled, qfn)
    # Quantization is a step function of the compensated weights, so tiny
    # float differences may flip a handful of codes; require near-identity.
    diff = sum(1 for p, q in zip(a, b) if p != q)
    assert diff <= len(a) // 1000, f"{diff}/{len(a)} bytes differ"
    tiled.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_tiled_hessian.py -v`
Expected: FAIL — `ImportError: cannot import name 'TiledHessian'`

- [ ] **Step 3: Implement `TiledHessian`**

Add to `featherquant/hessian.py`. Structure (all file access via `seek` + `readinto`/`write` into one preallocated `bytearray` — never `mmap`, never an allocation inside the panel loop):

```python
class TiledHessian:
    """Disk-backed Hessian: peak resident memory is one panel, not d^2.

    Files under work_dir:
      h.bin     — the d x d Gram matrix, fp32, row-major
      chol.bin  — the reverse-Cholesky factor V (same layout)
    Both are deleted by close(); they are scratch, not output.
    """

    def __init__(self, d: int, panel: int, work_dir: str,
                 dtype: str = "fp32"):
        # Preallocate both files and the single panel buffer here.
        ...

    def accumulate(self, x: np.ndarray) -> None:
        """H[p0:p1, :] += 2 * x[:, p0:p1].T @ x, one panel at a time."""
        ...

    def finalize(self, damp_percent: float = 0.01) -> None:
        """Damp the diagonal, then right-looking blocked Cholesky of J H J.

        For each panel k:
          1. read the diagonal block, factor it in memory (np.linalg.cholesky)
          2. solve the panel below it against that factor (scipy-free:
             np.linalg.solve on the triangular block is adequate at panel size)
          3. rank-k update the trailing tiles, streamed panel by panel
        """
        ...

    def row_block(self, j0: int, j1: int) -> np.ndarray:
        """Rows [j0, j1) of U = V^-1 by blocked back-substitution."""
        ...
```

Implementation guidance: keep `panel` a parameter, default it from the planner's budget (`panel = max(64, budget_share // (d * 4))`), and round it to a multiple of 64 so the panels align with the GPTQ group boundaries the caller asks for. Do not attempt to be clever with BLAS threading; correctness against the in-memory reference is the whole deliverable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_tiled_hessian.py -v`
Expected: 7 passed. `test_row_block_matches_in_memory` is the one that decides whether M5 is real. Do not loosen its tolerance to make it pass; a real divergence means the reversal (`J`) or the back-substitution is wrong.

- [ ] **Step 5: Register the rung**

In `featherquant/calibrator.py`:

```python
HESSIANS: dict[str, type] = {"full": InMemoryHessian,
                             "blocked": TiledHessian,
                             "diagonal": DiagonalHessian}
```

`TiledHessian` needs `panel` and `work_dir`, so construct it through a small factory in `run_calibration`:

```python
def _make_hessian(plan: Plan, width: int, work_dir: str) -> Any:
    """Build the Hessian for one linear group at the plan's rung."""
    cls = HESSIANS[plan.hessian_approx]
    if cls is TiledHessian:
        panel = max(64, (plan.budget_bytes // 16) // (width * 4) // 64 * 64)
        return TiledHessian(width, panel=panel, work_dir=work_dir)
    return cls(width, dtype=plan.calib.act_dtype)
```

and verify the planner's `HESSIAN_FRACTION["blocked"] = 0.10` against the panel this formula actually picks; if it is off, fix the fraction (and `docs/memory_model.md`), not the test.

- [ ] **Step 6: Write the central-claim memory test**

```python
# tests/memory/test_ooc_hessian_under_ceiling.py
"""M5's claim, as a test: blocked succeeds where full is OOM-killed."""
import json
import os
import subprocess
import sys

import pytest

from tests.memory.conftest import needs_cgroup

pytestmark = [pytest.mark.memory, pytest.mark.slow, needs_cgroup,
              pytest.mark.skipif(
                  not os.path.exists(os.path.expanduser("~/models/qwen3-0.6b")),
                  reason="calibration model unavailable")]

MODEL = os.path.expanduser("~/models/qwen3-0.6b")
VOCAB = os.path.expanduser("~/models/qwen3-0.6b-vocab.gguf")


def _run(plan_path, out, ceiling):
    return subprocess.run(
        ["bash", "bench/harness/run_under_ceiling.sh", ceiling,
         sys.executable, "-m", "featherquant.cli", "run", plan_path,
         "-o", out, "--vocab-gguf", VOCAB, "--calib-text",
         "bench/data/wiki.test.raw", "--ui", "none"],
        capture_output=True, text=True)


def _plan(tmp_path, approx, budget):
    idx = str(tmp_path / "idx.json")
    subprocess.run([sys.executable, "-m", "featherquant.cli", "index", MODEL,
                    "-o", idx], check=True, capture_output=True)
    plan = str(tmp_path / f"plan_{approx}.json")
    r = subprocess.run([sys.executable, "-m", "featherquant.cli", "plan", idx,
                        "--budget", budget, "--method", "gptq", "--format",
                        "q4_k_m", "--calib-samples", "32", "--calib-seqlen",
                        "512", "--hessian-approx", approx, "-o", plan],
                       capture_output=True, text=True)
    return plan, r


def test_blocked_fits_where_full_does_not(tmp_path):
    ceiling = os.environ.get("FQ_M5_CEILING", "700M")
    full_plan, full_r = _plan(tmp_path, "full", ceiling.replace("M", "MB"))
    blocked_plan, blocked_r = _plan(tmp_path, "blocked", ceiling.replace("M", "MB"))
    # The planner must already refuse 'full' at this ceiling, naming the
    # hessian — refusing before work is itself part of the claim.
    assert full_r.returncode == 2 and "hessian" in full_r.stderr
    assert blocked_r.returncode == 0, blocked_r.stderr
    out = str(tmp_path / "m5.gguf")
    run = _run(blocked_plan, out, ceiling)
    assert run.returncode == 0, (
        f"blocked run exit {run.returncode} "
        f"(137 = OOM-killed at {ceiling})\n{run.stderr[-2000:]}")
    assert os.path.getsize(out) > 0
```

`FQ_M5_CEILING` exists because the exact crossover depends on the machine; find it by bisection in Step 7 and set the default to the measured value.

- [ ] **Step 7: Measure the crossover and run the gate**

```bash
for C in 2G 1500M 1G 800M 700M 600M; do
  echo "== $C"; FQ_M5_CEILING=$C .venv/bin/pytest \
    tests/memory/test_ooc_hessian_under_ceiling.py -q -m memory
done
```

Record the largest ceiling at which `full` is refused/OOM-killed and `blocked` completes. Then produce the paired quality numbers at that ceiling:

```bash
bash bench/harness/run_baseline.sh "[...blocked run at the crossover ceiling...]" \
  m5_blocked_q4_k_m gptq_q4_k_m_blocked Qwen/Qwen3-0.6B /tmp/m5_blocked.gguf nvme <ceiling_bytes>
bash bench/harness/eval_quality.sh /tmp/m5_blocked.gguf bench/manifests/m5_blocked_q4_k_m.json
```

The gate is `ppl(m5_blocked_q4_k_m) ≈ ppl(m4_gptq_q4_k_m)` within the noise band measured in Task 17, at a ceiling where `m4_gptq_q4_k_m` cannot run.

- [ ] **Step 8: Write it up — including a negative result**

Add an "M5 — central claim" section to `docs/baselines.md` with: the crossover ceiling, both manifests' perplexities, the runtime ratio (blocked is slower — report memory and runtime together, spec §6), and the panel size used. If blocked does **not** match M4's quality, or does not run where M4 cannot, say so plainly and keep the numbers. A negative result here is still publishable (spec §5).

- [ ] **Step 9: Commit**

```bash
git add featherquant/hessian.py featherquant/calibrator.py featherquant/planner.py \
        tests/unit/test_tiled_hessian.py tests/memory/test_ooc_hessian_under_ceiling.py \
        bench/manifests docs/baselines.md docs/memory_model.md
git commit -m "feat: blocked out-of-core Hessian with panel Cholesky; M5 crossover measured"
```

---

## Milestone M6 — Approximation ladder

**Gate:** `docs/approximation_costs.md` fully populated, no `UNMEASURED` rows.

### Task 19: Low-rank rung and the remaining knobs

**Files:**
- Modify: `featherquant/hessian.py` (add `LowRankHessian`), `featherquant/calibrator.py`, `featherquant/planner.py`
- Test: `tests/unit/test_lowrank_hessian.py`

**Interfaces:**
- Consumes: `damped_inverse_cholesky`, `DiagonalHessian` (Task 14).
- Produces: `class LowRankHessian(d: int, rank: int, dtype: str = "fp32")` — same protocol; keeps a `d × rank` sketch plus the exact diagonal, reconstructing `H ≈ diag + S Sᵀ` at `finalize()`. Registered as `HESSIANS["lowrank"]`. Planner: `HESSIAN_FRACTION["lowrank"]` is replaced by an exact formula `rank * d * 4 + d * 4`, and `plan_job` gains `hessian_rank: int = 32` recorded in the plan and the run manifest's `approximations`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_lowrank_hessian.py
import numpy as np
import pytest

from featherquant.hessian import InMemoryHessian, LowRankHessian


def test_rank_r_memory_is_linear_in_d():
    d, r = 4096, 32
    h = LowRankHessian(d, rank=r)
    h.accumulate(np.ones((4, d), np.float32))
    h.finalize()
    assert h.resident_bytes < 4 * (r + 4) * d * 4
    assert h.resident_bytes < d * d * 4 // 100


@pytest.mark.parametrize("rank", [4, 16, 64])
def test_higher_rank_is_closer_to_full(rank):
    """The ladder must be monotone: more rank, less error."""
    rng = np.random.default_rng(0)
    d = 128
    x = rng.standard_normal((512, d)).astype(np.float32)
    ref = InMemoryHessian(d)
    ref.accumulate(x)
    ref.finalize()
    lr = LowRankHessian(d, rank=rank)
    lr.accumulate(x)
    lr.finalize()
    err = np.linalg.norm(lr.row_block(0, d) - ref.row_block(0, d))
    assert np.isfinite(err)
    pytest.approx  # error magnitude is asserted in the monotonicity test below


def test_error_decreases_with_rank():
    rng = np.random.default_rng(1)
    d = 128
    x = rng.standard_normal((512, d)).astype(np.float32)
    ref = InMemoryHessian(d)
    ref.accumulate(x)
    ref.finalize()
    errs = []
    for rank in (4, 16, 64):
        lr = LowRankHessian(d, rank=rank)
        lr.accumulate(x)
        lr.finalize()
        errs.append(np.linalg.norm(lr.row_block(0, d) - ref.row_block(0, d)))
    assert errs[0] > errs[1] > errs[2], errs
```

- [ ] **Step 2: Run it, implement, re-run**

Run: `.venv/bin/pytest tests/unit/test_lowrank_hessian.py -v` (FAIL: no `LowRankHessian`), implement the class with a streaming randomized sketch (`S += 2 · xᵀ (x Ω)` with a fixed seeded `Ω` of shape `d × rank`, exact diagonal accumulated alongside), then re-run to green.

Seed `Ω` from a constant (`np.random.default_rng(0)`) so runs stay bit-exact — invariant 4 applies to the ladder too.

- [ ] **Step 3: Register and plan-wire the rung**

`HESSIANS["lowrank"] = LowRankHessian`; `_make_hessian` passes `rank=plan.hessian_rank`; `plan_job` records `{"rung": "hessian_lowrank", "rank": r}` in `approximations`. Add `--hessian-rank` to `featherquant plan`.

- [ ] **Step 4: Commit**

```bash
git add featherquant/hessian.py featherquant/calibrator.py featherquant/planner.py \
        featherquant/cli.py tests/unit/test_lowrank_hessian.py
git commit -m "feat: low-rank + diagonal Hessian rungs wired into the planner and CLI"
```

---

### Task 20: Ladder sweep and table population

**Files:**
- Create: `bench/sweeps/ladder.yaml`, `featherquant/bench.py`
- Modify: `docs/approximation_costs.md`, `pyproject.toml` (add `pyyaml>=6`)
- Test: `tests/unit/test_bench.py`

**Interfaces:**
- Consumes: `Plan`, `plan_job`, `run_calibration`, `RunManifest`, `bench/harness/*`.
- Produces:
  - `load_sweep(path: str) -> list[dict[str, Any]]` — expands a YAML matrix into one config dict per cell (`model`, `budget`, `method`, `format`, `hessian_approx`, `calib_samples`, `calib_seqlen`, `spill`, `stat_precision`, `storage`).
  - `run_sweep(configs, out_dir, dry_run: bool = False) -> list[str]` — returns the run-manifest paths it produced (or would produce), skipping cells whose manifest already exists so a sweep is resumable.
  - `frontier_table(manifest_paths: list[str]) -> str` — a markdown table of `rung | peak Δ | runtime Δ | PPL Δ | task Δ | source`, deltas computed against the `hessian_full` reference manifest, ready to paste into `docs/approximation_costs.md`.
  - CLI: `featherquant bench --sweep bench/sweeps/ladder.yaml [--out bench/manifests] [--dry-run]`.

- [ ] **Step 1: Write the sweep file**

```yaml
# bench/sweeps/ladder.yaml — spec §5, one cell per rung.
model: ~/models/qwen3-0.6b
vocab_gguf: ~/models/qwen3-0.6b-vocab.gguf
calib_text: bench/data/wiki.test.raw
format: q4_k_m
method: gptq
storage: nvme
budget: 8GiB          # generous: the ladder measures quality, not feasibility
defaults:
  calib_samples: 128
  calib_seqlen: 512
  hessian_approx: full
  stat_precision: fp32
  spill: false
cells:
  - {run_id: m6_hessian_full,     hessian_approx: full}
  - {run_id: m6_hessian_blocked,  hessian_approx: blocked}
  - {run_id: m6_hessian_lowrank,  hessian_approx: lowrank, hessian_rank: 32}
  - {run_id: m6_hessian_diagonal, hessian_approx: diagonal}
  - {run_id: m6_calib_samples_64, calib_samples: 64}
  - {run_id: m6_calib_samples_32, calib_samples: 32}
  - {run_id: m6_calib_seqlen_256, calib_seqlen: 256}
  - {run_id: m6_calib_spill,      spill: true}
  - {run_id: m6_stats_bf16,       stat_precision: bf16}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_bench.py
import json

from featherquant.bench import frontier_table, load_sweep


def test_sweep_expands_defaults_into_cells(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "model: /m\nformat: q8_0\nmethod: gptq\nbudget: 1GiB\nstorage: nvme\n"
        "defaults: {calib_samples: 128, hessian_approx: full}\n"
        "cells:\n  - {run_id: a}\n  - {run_id: b, calib_samples: 64}\n")
    cells = load_sweep(str(p))
    assert [c["run_id"] for c in cells] == ["a", "b"]
    assert cells[0]["calib_samples"] == 128
    assert cells[1]["calib_samples"] == 64
    assert all(c["model"] == "/m" for c in cells)


def test_frontier_table_computes_deltas_against_the_reference(tmp_path):
    def manifest(run_id, rung, peak, runtime, ppl):
        d = {"run_id": run_id, "date": "04/08/2026",
             "model": {"id": "m", "revision": "r", "sha256": "s"},
             "method": "gptq", "approximations": [{"rung": rung}],
             "budget_bytes": 1 << 30, "enforcement": "cgroup_v2_memory_max",
             "peak_observed_bytes": peak, "oom_killed": False,
             "runtime_seconds": runtime, "bytes_read": 1, "bytes_written": 1,
             "storage": "nvme", "output_sha256": "x",
             "quality": {"ppl": ppl, "ppl_dataset": "w c=512", "tasks": {}},
             "host": {"cpu": "c", "ram_gb": 1, "kernel": "k"}}
        path = tmp_path / f"{run_id}.json"
        path.write_text(json.dumps(d))
        return str(path)

    paths = [manifest("ref", "hessian_full", 1000, 100.0, 10.0),
             manifest("diag", "hessian_diagonal", 400, 90.0, 10.31)]
    table = frontier_table(paths)
    assert "hessian_diagonal" in table
    assert "-600" in table or "-0.6" in table      # peak delta
    assert "+0.31" in table                        # ppl delta
    assert "UNMEASURED" not in table
```

- [ ] **Step 3: Run it, implement `featherquant/bench.py`, re-run**

Run: `.venv/bin/pytest tests/unit/test_bench.py -v` (FAIL: no module), then implement:

- `load_sweep` — `yaml.safe_load`, `os.path.expanduser` every path, merge `defaults` under each cell, carry the top-level keys down, and fail loudly on a cell without `run_id` or on an unknown key (a typo'd knob must not silently use the default).
- `run_sweep` — for each cell: index (cached per model), `plan_job`, then execute through `bench/harness/run_baseline.sh` wrapped in `run_under_ceiling.sh` so `enforcement` stays honest, then `eval_quality.sh`. Skip a cell whose manifest exists. On `InfeasiblePlan`, record a manifest with `oom_killed: false` and the refusal text in `approximations` — a refusal is data.
- `frontier_table` — load every manifest, find the one whose `approximations` contains `hessian_full` (or `run_id == "m6_hessian_full"`) as the reference, and emit deltas. A missing `quality.ppl` renders as `UNMEASURED`, never as a blank or a zero.
- CLI: register `_cmd_bench` in `_dispatch`.

- [ ] **Step 4: Add the dependency**

In `pyproject.toml`: `dependencies = ["numpy>=1.26", "gguf>=0.16,<0.18", "rich>=13", "pyyaml>=6"]`, then `uv pip install -e '.[dev]'`.

- [ ] **Step 5: Run the ladder sweep**

```bash
.venv/bin/featherquant bench --sweep bench/sweeps/ladder.yaml --dry-run   # inspect first
.venv/bin/featherquant bench --sweep bench/sweeps/ladder.yaml
```

Expected: nine manifests in `bench/manifests/`, each with a `quality.ppl` and a coherent-generation note (`eval_quality.sh` refuses otherwise).

- [ ] **Step 6: Populate the table — the M6 gate**

```bash
.venv/bin/python -c "
from featherquant.bench import frontier_table
import glob
print(frontier_table(sorted(glob.glob('bench/manifests/m6_*.json'))))
" > /tmp/ladder_table.md
```

Paste the generated rows into `docs/approximation_costs.md`, replacing every `UNMEASURED` row and filling each `source` cell with its manifest filename. Then re-run `.venv/bin/pytest tests/unit/test_approx_costs.py` — `test_unmeasured_option_line_says_so` will now fail, because nothing is unmeasured. Update that test to assert the opposite (measured numbers appear in the planner's refusal options) and add a guard test:

```python
def test_no_unmeasured_rows_remain():
    """M6 gate, enforced: every ladder rung has a committed measurement."""
    costs = load_costs()
    unmeasured = [name for name, c in costs.items() if not c.measured]
    assert unmeasured == [], f"still UNMEASURED: {unmeasured}"
```

- [ ] **Step 7: Verify the planner now quotes real numbers**

```bash
.venv/bin/featherquant plan /tmp/idx_qwen3_14b.json --budget 1GiB --method gptq \
  --format q4_k_m --calib-samples 128 --calib-seqlen 512 -o /tmp/x.json
```

Expected: the refusal's `options:` block shows measured PPL costs, matching the spec §4.2 example shape.

- [ ] **Step 8: Commit**

```bash
git add featherquant/bench.py featherquant/cli.py bench/sweeps/ladder.yaml \
        bench/manifests docs/approximation_costs.md pyproject.toml \
        tests/unit/test_bench.py tests/unit/test_approx_costs.py
git commit -m "feat: ladder sweep runner; approximation_costs.md fully measured (M6 gate)"
```

---

## Milestone M7 — Checkpoint / resume

**Gate:** `SIGKILL` at 100 random points; every resume produces output bit-identical to the uninterrupted run.

### Task 21: Calibration-state checkpointing

**Files:**
- Modify: `featherquant/manifest.py` (extend the checkpoint schema), `featherquant/calibrator.py`, `featherquant/planner.py` (budget the checkpoint), `featherquant/cli.py`
- Create: `scripts/kill_resume_calibration.sh`
- Test: `tests/unit/test_calibration_checkpoint.py`

**Interfaces:**
- Consumes: `Manifest`, `TensorEntry`, `sha256_file_region` (existing); `ActivationCache` (Task 13); `Plan` (Task 9).
- Produces:
  - `MANIFEST_VERSION = 2` with two new optional fields on `Manifest`: `calibration: dict[str, Any] | None` (`{"layer": int, "cache_path": str, "cache_sha256": str, "samples": int, "seqlen": int, "hidden": int, "dtype": str}`) and `plan_sha256: str | None`. `load()` still refuses any other version — a v1 manifest from an interrupted RTN run must fail loudly, not resume into a different schema.
  - `ActivationCache.dump(path: str) -> str` (writes the cache to `path`, returns its sha256) and `ActivationCache.restore(path: str, expected_sha256: str) -> None` (verifies before trusting).
  - `run_calibration(..., resume: bool = False)` — on resume, verifies the plan hash, the committed tensors, and the cache checksum, then continues at the recorded layer.
  - `estimate_peak(..., checkpoint: bool = False)` — when checkpointing is enabled the activation-cache term is counted twice (live copy + the copy being written), because that is what the process actually holds during a dump.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_calibration_checkpoint.py
import numpy as np
import pytest

from featherquant.activations import ActivationCache
from featherquant.manifest import MANIFEST_VERSION, Manifest, TensorEntry


def test_manifest_version_is_2_with_calibration_field(tmp_path):
    assert MANIFEST_VERSION == 2
    m = Manifest(source_path="/s", source_size=1, source_mtime_ns=2,
                 config={"fmt": "q4_k_m"}, header_end=32, header_sha256="h",
                 tensors=[TensorEntry("t", 8, 32, 34, None)],
                 status="in_progress",
                 calibration={"layer": 7, "cache_path": "/tmp/c.bin",
                              "cache_sha256": "a" * 64, "samples": 4,
                              "seqlen": 8, "hidden": 16, "dtype": "fp16"},
                 plan_sha256="b" * 64)
    p = tmp_path / "m.json"
    m.save(str(p))
    again = Manifest.load(str(p))
    assert again.calibration is not None
    assert again.calibration["layer"] == 7
    assert again.plan_sha256 == "b" * 64


def test_v1_manifest_is_refused(tmp_path):
    p = tmp_path / "old.json"
    p.write_text('{"version": 1, "source_path": "/s", "source_size": 1, '
                 '"source_mtime_ns": 2, "config": {}, "header_end": 0, '
                 '"header_sha256": "", "tensors": [], "status": "in_progress"}')
    with pytest.raises(RuntimeError, match="version 1"):
        Manifest.load(str(p))


def test_cache_dump_and_restore_roundtrip(tmp_path):
    path = str(tmp_path / "dump.bin")
    with ActivationCache(3, 4, 8, dtype="fp32") as c:
        for i in range(3):
            c.write(i, np.full((4, 8), i, np.float32))
        digest = c.dump(path)
    with ActivationCache(3, 4, 8, dtype="fp32") as c2:
        c2.restore(path, digest)
        for i in range(3):
            assert np.array_equal(c2.read(i), np.full((4, 8), i, np.float32))


def test_restore_rejects_a_corrupt_cache(tmp_path):
    path = str(tmp_path / "dump.bin")
    with ActivationCache(1, 2, 4, dtype="fp32") as c:
        c.write(0, np.ones((2, 4), np.float32))
        digest = c.dump(path)
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(b"\xff\xff\xff\xff")
    with ActivationCache(1, 2, 4, dtype="fp32") as c2:
        with pytest.raises(RuntimeError, match="checksum"):
            c2.restore(path, digest)


def test_checkpoint_doubles_the_activation_term_in_the_budget():
    from featherquant.indexer import ModelIndex, TensorInfo
    from featherquant.planner import CalibConfig, estimate_peak
    idx = ModelIndex("qwen3", 1, 64, 128, 256,
                     {"n_heads": 4, "n_kv_heads": 2, "head_dim": 16},
                     [TensorInfo("model.layers.0.mlp.down_proj.weight",
                                 (64, 128), "BF16", "s", 0, 64 * 128 * 2,
                                 True, 0, "ffn_down")],
                     64 * 128 * 2, 64 * 128 * 2)
    calib = CalibConfig(8, 16)
    plain = estimate_peak(idx, 1 << 30, "gptq", calib, "full", 64, 1 << 20)
    ckpt = estimate_peak(idx, 1 << 30, "gptq", calib, "full", 64, 1 << 20,
                         checkpoint=True)
    assert ckpt.activation_cache == 2 * plain.activation_cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_calibration_checkpoint.py -v`
Expected: FAIL — `assert 1 == 2` on `MANIFEST_VERSION`.

- [ ] **Step 3: Implement**

- `featherquant/manifest.py`: bump `MANIFEST_VERSION` to `2`; add `calibration: dict[str, Any] | None = None` and `plan_sha256: str | None = None` fields (defaults keep the RTN call sites unchanged).
- `featherquant/activations.py`: add `dump()` (stream the cache to `path` in sample-sized chunks through the existing buffer, `fsync`, return the sha256 computed as it writes) and `restore()` (hash first, compare, then read back sample by sample; `RuntimeError` mentioning "checksum" on mismatch).
- `featherquant/calibrator.py`: after each layer, write the packed tensors, `flush()`, dump the cache to `<dst>.calib.bin`, then save the manifest with `calibration.layer = layer + 1`. On `resume=True`, load the manifest, compare `plan_sha256` against the current plan's hash (refuse on mismatch, exactly as the RTN path refuses on a changed source), verify committed tensors, restore the cache, and start at the recorded layer.
- `featherquant/planner.py`: `estimate_peak(..., checkpoint: bool = False)` doubles the activation term; `plan_job` gains `checkpoint: bool = False`, records `{"rung": "checkpoint_calibration"}` in `approximations` when enabled (it is a budget consumer, not a quality approximation — note that in the entry's `reason`), and `featherquant plan` gains `--checkpoint`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_calibration_checkpoint.py tests/test_resume.py -v`
Expected: all pass — the existing RTN resume tests must still pass against schema v2.

- [ ] **Step 5: Write the SIGKILL torture script**

```bash
#!/usr/bin/env bash
# scripts/kill_resume_calibration.sh — M7 gate for the calibrated path.
# SIGKILL at random moments, resume, repeat, then require byte-identity
# with an uninterrupted reference run.
# Usage: kill_resume_calibration.sh PLAN.json OUT.gguf VOCAB.gguf CALIB.txt [KILLS]
set -uo pipefail
PLAN=$1; OUT=$2; VOCAB=$3; CALIB=$4; KILLS=${5:-100}
PY=$(command -v python)
rm -f "$OUT" "$OUT.manifest.json" "$OUT.calib.bin" "$OUT.ref" "$OUT.ref.manifest.json"

"$PY" -m featherquant.cli run "$PLAN" -o "$OUT.ref" --vocab-gguf "$VOCAB" \
  --calib-text "$CALIB" --ui none >/dev/null || { echo "reference run failed"; exit 1; }

tries=0
while [ "$tries" -lt "$KILLS" ]; do
  tries=$((tries+1))
  "$PY" -m featherquant.cli run "$PLAN" -o "$OUT" --vocab-gguf "$VOCAB" \
    --calib-text "$CALIB" --resume --ui none >/dev/null 2>&1 &
  pid=$!
  sleep "$((RANDOM % 30)).$((RANDOM % 9))"
  kill -9 "$pid" 2>/dev/null
  wait "$pid"; code=$?
  if [ "$code" -eq 0 ]; then
    if cmp -s "$OUT" "$OUT.ref"; then
      echo "PASS: byte-identical after $tries interrupted runs"
      rm -f "$OUT" "$OUT.manifest.json" "$OUT.calib.bin"
      continue                      # start the next kill cycle from scratch
    fi
    echo "FAIL: output differs from the uninterrupted reference"; exit 1
  fi
done
echo "PASS: $KILLS kill/resume cycles, every completion byte-identical"
```

- [ ] **Step 6: Run the gate**

```bash
bash scripts/kill_resume_calibration.sh /tmp/plan_0.6b_gptq.json /tmp/m7.gguf \
  ~/models/qwen3-0.6b-vocab.gguf bench/data/wiki.test.raw 100
```

Expected: `PASS: 100 kill/resume cycles, every completion byte-identical`. Any single mismatch is a correctness bug — likely the cache dump racing the manifest save; the manifest must be written *after* the cache dump `fsync`s, never before.

- [ ] **Step 7: Record and commit**

Add an "M7 gate" line to `docs/baselines.md` with the kill count and the date (DD/MM/YYYY).

```bash
git add featherquant/manifest.py featherquant/activations.py featherquant/calibrator.py \
        featherquant/planner.py featherquant/cli.py scripts/kill_resume_calibration.sh \
        tests/unit/test_calibration_checkpoint.py docs/baselines.md
git commit -m "feat: calibration-state checkpointing; 100-kill resume gate byte-identical"
```

---

## Milestone M8 — Scale

**Gate:** frontier measured across budget × model-size ratios × storage tiers.

### Task 22: Frontier sweep

**Files:**
- Create: `bench/sweeps/budgets.yaml`, `docs/frontier.md`
- Modify: `featherquant/bench.py`
- Test: `tests/unit/test_frontier.py`

**Interfaces:**
- Consumes: `load_sweep`, `run_sweep`, `frontier_table` (Task 20).
- Produces: `frontier_rows(manifest_paths) -> list[dict[str, Any]]` with `model`, `model_bytes`, `budget_bytes`, `ratio` (`model_bytes / budget_bytes`), `storage`, `method`, `hessian_approx`, `ppl`, `runtime_seconds`, `oom_killed`, `refused`; and `frontier_markdown(rows) -> str` grouping by storage tier. `featherquant bench --sweep bench/sweeps/budgets.yaml --frontier docs/frontier.md`.

- [ ] **Step 1: Write the sweep matrix**

```yaml
# bench/sweeps/budgets.yaml — spec §8 frontier: budget x ratio x storage.
# Every cell answers the primary research question (spec §1.3): given a
# ceiling B, what is the best achievable quality?
calib_text: bench/data/wiki.test.raw
format: q4_k_m
method: gptq
defaults:
  calib_samples: 128
  calib_seqlen: 512
  hessian_approx: blocked
models:
  - {id: Qwen/Qwen3-0.6B, path: ~/models/qwen3-0.6b, vocab_gguf: ~/models/qwen3-0.6b-vocab.gguf}
  - {id: Qwen/Qwen3-14B,  path: ~/models/qwen3-14b,  vocab_gguf: ~/models/qwen3-14b-vocab.gguf}
budgets: [1GiB, 2GiB, 4GiB, 8GiB]
storage: [nvme]           # add sata_ssd / hdd by re-running with FQ_TEMP_DIR
cells:
  - {run_id_prefix: m8, hessian_approx: blocked}
  - {run_id_prefix: m8_full, hessian_approx: full}     # refuses at low budgets: that is data
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_frontier.py
import json

from featherquant.bench import frontier_markdown, frontier_rows


def _m(tmp_path, run_id, budget, ppl, storage, oom=False, refused=None):
    d = {"run_id": run_id, "date": "04/08/2026",
         "model": {"id": "Qwen/Qwen3-0.6B", "revision": "r", "sha256": "s"},
         "method": "gptq", "approximations": [{"rung": "hessian_blocked"}],
         "budget_bytes": budget, "enforcement": "cgroup_v2_memory_max",
         "peak_observed_bytes": budget - 1, "oom_killed": oom,
         "runtime_seconds": 42.0, "bytes_read": 10, "bytes_written": 5,
         "storage": storage, "output_sha256": "x",
         "quality": {"ppl": ppl, "ppl_dataset": "w c=512", "tasks": {}},
         "host": {"cpu": "c", "ram_gb": 64, "kernel": "k"}}
    if refused:
        d["approximations"].append({"refused": refused})
    p = tmp_path / f"{run_id}.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_rows_carry_ratio_and_storage(tmp_path):
    paths = [_m(tmp_path, "a", 1 << 30, 12.0, "nvme"),
             _m(tmp_path, "b", 4 << 30, 11.4, "nvme")]
    rows = frontier_rows(paths, model_bytes={"Qwen/Qwen3-0.6B": 2 << 30})
    assert rows[0]["ratio"] == 2.0 and rows[1]["ratio"] == 0.5
    assert {r["storage"] for r in rows} == {"nvme"}


def test_oom_and_refusal_rows_survive_into_the_table(tmp_path):
    paths = [_m(tmp_path, "ok", 4 << 30, 11.4, "nvme"),
             _m(tmp_path, "dead", 1 << 30, None, "nvme", oom=True)]
    md = frontier_markdown(frontier_rows(paths,
                                         model_bytes={"Qwen/Qwen3-0.6B": 2 << 30}))
    assert "OOM" in md
    assert "11.4" in md
```

- [ ] **Step 3: Run it, implement, re-run**

Run: `.venv/bin/pytest tests/unit/test_frontier.py -v` (FAIL), implement `frontier_rows` / `frontier_markdown` in `bench.py`, re-run to green.

`frontier_markdown` must render `oom_killed` rows as `OOM` and refused rows as `REFUSED (<binding term>)` — the failures are half the frontier. Never drop them.

- [ ] **Step 4: Run the sweep**

```bash
.venv/bin/featherquant bench --sweep bench/sweeps/budgets.yaml --dry-run
.venv/bin/featherquant bench --sweep bench/sweeps/budgets.yaml \
  --frontier docs/frontier.md
```

Then re-run with `FQ_TEMP_DIR` pointed at a SATA SSD and, if one is available, an HDD, changing `storage:` accordingly. Between runs, drop caches (`FQ_DROP_CACHES=1`) and say so in the doc — cold vs warm page cache changes I/O numbers substantially (spec §6).

- [ ] **Step 5: Write `docs/frontier.md`**

Structure: one table per storage tier, rows sorted by ratio then budget, columns `model | model size | budget (GiB) | ratio | method | rung | ppl | runtime (s) | result`. Above the tables, three paragraphs:

1. **What this measures** — the primary research question (spec §1.3), verbatim.
2. **Where the frontier sits** — the observed answer, including every OOM and refusal.
3. **What it does not show** — the RTN baseline succeeds under these ceilings too (spec §1.1); the contribution is the calibrated frontier, not the fact that a model was quantized in 2 GiB.

- [ ] **Step 6: Commit**

```bash
git add bench/sweeps/budgets.yaml featherquant/bench.py tests/unit/test_frontier.py \
        bench/manifests docs/frontier.md
git commit -m "feat: frontier sweep across budget x ratio x storage; docs/frontier.md (M8 gate)"
```

---

### Task 23: README and claim hygiene

**Files:**
- Modify: `README.md`, `docs/baselines.md`
- Test: `tests/unit/test_claim_hygiene.py`

**Interfaces:**
- Consumes: `load_costs` (Task 8); `RunManifest.load` (Task 1).
- Produces: a test that fails the build when a claim in the README or docs is not backed by a committed run manifest.

Spec §1.1 is explicit: *"Any claim of the form 'we quantized a 20 GB model in 2 GB of RAM' that refers to RTN K-quants is not a result... Do not let it become the headline in the README."* Today's README leads with exactly that framing. This task fixes it.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_claim_hygiene.py
"""Numbers in the docs must trace to a committed run manifest (spec §11.2)."""
import glob
import json
import pathlib
import re

DOCS = ["README.md", "docs/baselines.md", "docs/frontier.md",
        "docs/approximation_costs.md", "docs/memory_model.md"]


def _manifest_ids():
    ids = set()
    for p in glob.glob("bench/manifests/*.json"):
        ids.add(json.loads(pathlib.Path(p).read_text())["run_id"])
        ids.add(pathlib.Path(p).name)
    return ids


def test_every_ppl_number_names_its_source():
    """A perplexity figure must appear on a line that also names a manifest."""
    ids = _manifest_ids()
    bad = []
    for doc in DOCS:
        path = pathlib.Path(doc)
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bppl\b|\bperplexity\b", line, re.I) and \
                    re.search(r"\d+\.\d+", line):
                if not any(mid in line for mid in ids) and \
                        "UNMEASURED" not in line and not line.lstrip().startswith(">"):
                    bad.append(f"{doc}:{i}: {line.strip()}")
    assert not bad, "perplexity figures without a manifest reference:\n" + "\n".join(bad)


def test_readme_does_not_headline_the_rtn_baseline():
    """Spec §1.1: streaming RTN is the baseline, not the contribution."""
    head = pathlib.Path("README.md").read_text().split("## ", 1)[0].lower()
    assert "calibration" in head or "calibrated" in head, \
        "the README opening must frame the contribution as calibration-aware"
    banned = re.search(r"quantiz\w+ a \d+\s*gb model in \d+\s*gb", head)
    assert banned is None, f"RTN-baseline headline claim: {banned[0]!r}"
```

- [ ] **Step 2: Run it to see what currently fails**

Run: `.venv/bin/pytest tests/unit/test_claim_hygiene.py -v`
Expected: FAIL on `test_readme_does_not_headline_the_rtn_baseline` (the current opening leads with "converts an F16/BF16 GGUF... while keeping peak process memory under a user-configured budget").

- [ ] **Step 3: Rewrite the README opening**

Replace the opening paragraph with the honest framing, keeping the existing "How it works" and "Quick start" sections intact:

- One sentence on what FeatherQuant is: calibration-aware quantization under a hard, externally enforced memory ceiling.
- A short **"What is not the contribution"** paragraph stating plainly that streaming RTN K-quantization already works today under low ceilings (`llama-quantize` + `convert_hf_to_gguf.py`), with a pointer to `docs/baselines.md` for the measured ceilings at which each baseline breaks.
- A **"What is the contribution"** paragraph: running imatrix/GPTQ-class methods — the ones that need statistics across the calibration set — inside a declared budget, and measuring the quality cost of every approximation taken to get there. Link `docs/approximation_costs.md` and `docs/frontier.md`.
- Keep the memory/determinism/crash-safety bullet list; add one bullet for the calibrated path with a link to the frontier.

- [ ] **Step 4: Re-run and fix any unsourced numbers**

Run: `.venv/bin/pytest tests/unit/test_claim_hygiene.py -v`
Expected: both pass. If `test_every_ppl_number_names_its_source` flags a line, add the manifest filename to that line rather than deleting the number.

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/pytest -q -m "not slow" && .venv/bin/ruff check featherquant tests scripts && .venv/bin/mypy featherquant`

```bash
git add README.md docs tests/unit/test_claim_hygiene.py
git commit -m "docs: frame the contribution per spec §1; enforce manifest-backed numbers"
```

---

## Self-review notes

Checked against `spec.md` after drafting:

- **§2 invariants** — cgroup enforcement (Tasks 2, 7, 18), no peak-RSS-only claims (run manifests carry `enforcement`), no `mmap` (Task 7 guard test), determinism (Task 7), pre-allocated buffers (Tasks 13, 16), fail-loudly refusal (Task 9), approximations as first-class manifest fields (Tasks 9, 19, 20).
- **§3 memory model** — budget equation and the `d_in = intermediate_size` trap are both pinned by tests in Task 9; the embedding floor is in the refusal message and in `_embed`'s row-grouping.
- **§4 architecture** — indexer (6), planner (9), reader (existing, guarded in 7), calibrator (16), quantizer (existing + 15), writer (existing), checkpoint (21), validator (7).
- **§5 ladder** — full/blocked/lowrank/diagonal (14, 18, 19), calibration knobs and spill (13, 20), stat precision (20).
- **§6 measurement** — run manifest (1), ppl provenance (3, 17), runtime always reported alongside memory (2, 22), cold-cache note (2, 22).
- **§7 gates** — M0 (2–4), M1 (6), M2 (7), M3 (11), M4 (16, 17), M5 (18), M6 (20), M7 (21), M8 (22).
- **§8 CLI** — all five subcommands (6, 10, 20), legacy flat form preserved for `featherquant.sh`.
- **§9 layout** — `tests/{unit,integration,determinism,memory}` and `bench/{harness,sweeps,manifests}` created; module files stay flat per your decision.
- **§10 forbidden patterns** — each has a test or an explicit step; `try/except` around memory errors is avoided throughout (only I/O and parsing are wrapped).
- **§11 working rules** — `docs/memory_model.md` updated in the same commits that touch the budget equation (11, 18, 21); numbers require manifests (23); format constants verified against `GGML_QUANT_SIZES` at runtime (15); coherence checked before any table entry (17).

Two things worth flagging before execution:

1. **Task 12 is the schedule risk.** Everything from M4 on rests on the numpy forward pass agreeing with llama.cpp. If `test_greedy_decode_matches_llama_cli` resists, the likely culprits are `rope_theta` (Qwen3 uses 1e6), `q_norm`/`k_norm` placement (before RoPE, per head), and the GQA repeat axis. Budget real time for it; do not proceed to Task 16 on a forward pass that only "looks close".
2. **Task 18's math is stated but unproven in this document.** The reverse-Cholesky identity (`V = J·chol_lower(JHJ)·J`) is the design; `test_row_block_matches_in_memory` is the arbiter. If the identity does not hold as written, the fallback is a blocked explicit inverse followed by an in-panel factorization — slower, more disk, same interface, and the tests do not change.

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-04-featherquant-spec-m0-m8.md`.**
