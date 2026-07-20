# Checkpoint/Resume Implementation Plan (Phase 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A killed featherquant run resumes with `--resume` and finishes with output byte-identical to an uninterrupted run.

**Architecture:** A sidecar manifest (`<output>.manifest.json`) is written atomically (tmp + `os.replace`) after every committed tensor. It records source identity, full config, the deterministic tensor plan with output offsets, and a sha256 per committed tensor. Resume validates source + config + every committed tensor's bytes, then reopens the output `r+b`, seeks to the first uncommitted tensor's offset, and continues. Granularity is per-tensor (largest tensor ≈ 1–2 GB packed → bounded redo work); block-level is a later refinement.

**Tech Stack:** stdlib only (`json`, `hashlib`, `os.replace`).

## Global Constraints

- All MVP global constraints apply. New: never treat a partially written model as complete — the manifest's `status` field is the only source of truth, and the final rename of status to `"complete"` happens only after validation.
- Determinism is load-bearing: resume assumes the header/KV/tensor-info bytes of a re-planned run are identical to the original. Guarded by an explicit header-hash check, not by hope.
- This plan and the adaptive-blocks plan both modify `engine.py`; execute this one first, rebase the other.
- Gates (pytest, ruff, mypy strict) green per task; commits by majipa007.

## File Structure

```
featherquant/manifest.py   — Manifest dataclass, atomic save/load, verify (new)
featherquant/engine.py     — commit-per-tensor, resume path (modify)
featherquant/cli.py        — --resume flag (modify)
tests/test_manifest.py     — unit tests (new)
tests/test_resume.py       — kill/resume integration tests (new)
scripts/kill_resume_test.sh — random-kill loop on a real model (new)
```

---

### Task 1: Manifest module

**Files:**
- Create: `featherquant/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing from featherquant.
- Produces:
  - `sha256_file_region(path: str, offset: int, nbytes: int) -> str` (streamed, 8 MiB chunks — must work on files bigger than RAM).
  - `class Manifest` with fields: `version: int` (1), `source_path: str`, `source_size: int`, `source_mtime_ns: int`, `config: dict[str, Any]` (fmt, max_ram is EXCLUDED — budget may change between runs), `header_end: int` (file offset where tensor data begins), `header_sha256: str`, `tensors: list[TensorEntry]`, `status: str` ("in_progress" | "complete").
  - `TensorEntry`: `name, ggml_type: int, offset: int` (absolute file offset), `nbytes: int`, `sha256: str | None` (None = not committed).
  - `Manifest.save(path)` — atomic: write `path + ".tmp"`, flush+fsync, `os.replace`.
  - `Manifest.load(path) -> Manifest`, raises `RuntimeError` on version/shape mismatch.

- [ ] **Step 1: Write the failing tests**

```python
"""Manifest atomic save/load/verify unit tests."""
import json

import pytest

from featherquant.manifest import Manifest, TensorEntry, sha256_file_region


def _mk(tmp_path):
    return Manifest(source_path="/x/src.gguf", source_size=100, source_mtime_ns=5,
                    config={"fmt": "q8_0"}, header_end=64, header_sha256="h" * 64,
                    tensors=[TensorEntry("a", 8, 64, 34, None)], status="in_progress")


def test_roundtrip_atomic(tmp_path):
    m = _mk(tmp_path)
    p = tmp_path / "out.gguf.manifest.json"
    m.save(str(p))
    assert not (tmp_path / "out.gguf.manifest.json.tmp").exists()  # replaced, not left over
    m2 = Manifest.load(str(p))
    assert m2 == m


def test_load_rejects_wrong_version(tmp_path):
    p = tmp_path / "m.json"
    m = _mk(tmp_path)
    m.save(str(p))
    d = json.loads(p.read_text())
    d["version"] = 999
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError):
        Manifest.load(str(p))


def test_sha256_file_region(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"A" * 10 + b"B" * 20 + b"C" * 5)
    assert sha256_file_region(str(p), 10, 20) == hashlib.sha256(b"B" * 20).hexdigest()
```

- [ ] **Step 2: Verify failure** — `ModuleNotFoundError`.
- [ ] **Step 3: Implement** — `@dataclass` with `asdict`/`from dict` (validate `version == 1`, reconstruct `TensorEntry` objects), `save` using `tempfile` in the same directory + `os.fsync` + `os.replace`, streamed region hashing:

```python
def sha256_file_region(path: str, offset: int, nbytes: int) -> str:
    """Hash a byte range without materializing it (8 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = f.read(min(8 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"short read hashing {path} at {offset}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Verify pass, gates, commit** — `feat: atomic sidecar manifest for checkpointing`.

### Task 2: Engine writes checkpoints

**Files:**
- Modify: `featherquant/engine.py`
- Test: extend `tests/test_resume.py` (new file)

**Interfaces:**
- Consumes: Task 1.
- Produces: `quantize_model(...)` gains `manifest_path: str | None` (default `dst + ".manifest.json"`). Behavior: after `begin_data()`, record `header_end` (current file offset, aligned) + `header_sha256` (hash of bytes `[0, header_end)`) + full tensor plan with absolute offsets; save. After each tensor: fill its `sha256` (hash the just-written output region via `sha256_file_region`), save. After validation-free completion: `status = "complete"`, save. `IncrementalWriter` exposes `data_offset_of(i)`: absolute offset of the i-th declared tensor (`header_end + gguf-relative offset` from `self.w.ti_data_offset`... implement by capturing `f.tell()` at each `begin_tensor()` — simpler and provably correct).

- [ ] **Step 1: Failing test** — run `quantize_model` on the 3-tensor synthetic model; assert manifest exists, `status == "complete"`, every entry has a sha256 that equals `sha256_file_region(out, e.offset, e.nbytes)`, and offsets are 32-aligned relative to `header_end`.
- [ ] **Step 2: Watch fail** (`TypeError` on kwarg).
- [ ] **Step 3: Implement** — capture per-tensor absolute offset at `begin_tensor()` time; manifest save after each tensor. `source_mtime_ns` from `os.stat`.
- [ ] **Step 4: Suite green (all existing tests untouched: manifest is additive).**
- [ ] **Step 5: Commit** — `feat: per-tensor checkpoint manifest during quantization`.

### Task 3: Resume path

**Files:**
- Modify: `featherquant/engine.py`, `featherquant/cli.py` (`--resume` flag)
- Test: extend `tests/test_resume.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `quantize_model(..., resume: bool = False)`. Resume flow:
  1. Load manifest; reject if `status == "complete"` (nothing to do) or source path/size/mtime mismatch (`RuntimeError` naming the difference).
  2. Re-run planning; re-serialize header/KV/TI to a scratch buffer (`GGUFWriter` to a temp file in `report`-adjacent scratch); its sha256 must equal `header_sha256` — proves determinism assumption holds for this source+config.
  3. Verify every committed tensor: `sha256_file_region(dst, e.offset, e.nbytes) == e.sha256`; first mismatch → that tensor is re-marked uncommitted (crash mid-write) and becomes the resume point.
  4. Open `dst` as `r+b`, seek to resume offset, continue the normal loop from the first uncommitted tensor (writer wrapper for resume mode writes directly to the reopened handle — `IncrementalWriter` gains a classmethod `reopen(path, offset)` returning a writer whose `begin_data()` is a no-op and whose file is positioned at `offset`).

- [ ] **Step 1: Failing tests**

```python
def test_resume_completes_identically(tmp_path):
    # Run once fully -> reference bytes. Run again but force a crash after
    # tensor 1 (monkeypatch _stream_copy/_stream_quantize to raise on 2nd
    # tensor), then resume; final bytes must equal the reference.

def test_resume_rejects_changed_source(tmp_path):
    # touch/rewrite source between crash and resume -> RuntimeError

def test_resume_detects_corrupt_committed_tensor(tmp_path):
    # flip one byte inside a committed tensor's region -> that tensor re-runs,
    # final output still byte-identical to reference
```

(Write these as real code during execution — the crash hook is a `_fail_after: int | None` test kwarg on `quantize_model`, symmetrical with `_force_chunk_rows`.)

- [ ] **Step 2: Watch them fail. Step 3: Implement. Step 4: Suite green. Step 5: Commit** — `feat: verified resume from sidecar manifest`.

### Task 4: Kill -9 torture script + real model

**Files:**
- Create: `scripts/kill_resume_test.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Torture test: repeatedly SIGKILL featherquant at random moments, resume,
# until it completes; then verify against an uninterrupted reference run.
# Usage: scripts/kill_resume_test.sh SRC.gguf OUT.gguf [BUDGET]
set -uo pipefail
SRC=$1; OUT=$2; BUDGET=${3:-1GB}
PY=$(command -v python)
rm -f "$OUT" "$OUT.manifest.json"
"$PY" -m featherquant.cli --model "$SRC" --output "$OUT.ref" --max-ram "$BUDGET" >/dev/null
tries=0
while true; do
  tries=$((tries+1))
  "$PY" -m featherquant.cli --model "$SRC" --output "$OUT" --max-ram "$BUDGET" --resume &
  pid=$!
  sleep "0.$((RANDOM % 9))$((RANDOM % 9))"; kill -9 $pid 2>/dev/null
  wait $pid; code=$?
  if [ $code -eq 0 ]; then break; fi
  [ $tries -gt 200 ] && { echo "FAIL: no completion in 200 kills"; exit 1; }
done
cmp "$OUT" "$OUT.ref" && echo "PASS: identical after $tries interrupted runs" || exit 1
```

- [ ] **Step 2: Run on Qwen3-0.6B** — `scripts/kill_resume_test.sh ~/models/qwen3-0.6b-bf16.gguf /tmp/kr.gguf 1GB`. Expected: `PASS: identical after N interrupted runs`. Also run one pass with `--resume` on a fresh path (no manifest) — must behave like a normal run, not crash.
- [ ] **Step 3: Record in baseline notes; commit script + docs** — `feat: kill-resume torture harness`.
