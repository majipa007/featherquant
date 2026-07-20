# FeatherQuant Minimum Research Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize an F16/BF16 GGUF model to a valid Q8_0 GGUF while peak process RAM stays under a user-configured budget, even when the model is larger than that budget.

**Architecture:** A single-process streaming pipeline: GGUF metadata is read without materializing tensors, output tensor sizes/offsets are precomputed (Q8_0 size is deterministic from shape), then each tensor is read in row-aligned chunks sized from the budget, quantized with a numpy Q8_0 kernel that byte-matches llama.cpp's reference, and written sequentially to its precomputed position. Non-quantizable tensors are byte-copied in chunks. Validation compares against `llama-quantize` output and runs under an OS-enforced cgroup ceiling.

**Tech Stack:** Python ≥3.10, numpy, the `gguf` pip package (from llama.cpp) for metadata/header handling, pytest, systemd-run (cgroup v2) for external memory enforcement, llama.cpp binaries for baseline/smoke tests.

## Global Constraints

- Linux only, CPU only, single process, single worker (spec §10).
- Python `>=3.10`; dependencies exactly: `numpy>=1.26`, `gguf>=0.16,<0.18`, `pytest>=8` (dev). No other runtime deps.
- Scope: input = little-endian GGUF with F32/F16/BF16 tensors; output = GGUF with Q8_0 quantized 2-D tensors; everything else copied verbatim. No Safetensors, no K-quants, no calibration, no resume, no adaptive block resizing (those are later plans).
- Memory contract for this prototype: measured peak RSS of the featherquant process stays ≤ `--max-ram`, AND the job completes under an external `systemd-run -p MemoryMax=` ceiling. Never materialize a full tensor's data; all tensor data moves through caller-owned reusable buffers.
- Quantization rule (mirrors `llama-quantize` Q8_0 defaults; verified against reference in Task 7): tensors with ≥2 dims, source type F32/F16/BF16, and row length divisible by 32 → Q8_0; all other tensors keep their source type and bytes.
- Determinism: same input + same config → byte-identical output file.
- The GGUF file is the working dir's only pre-existing content; project is not yet a git repo — Task 1 creates it.
- All commands below run from the repo root with the venv active: `source .venv/bin/activate`.

## File Structure

```
featherquant/
  __init__.py     — empty package marker
  q8_0.py         — Q8_0 block quantize/dequantize + bf16→f32 (pure numpy, no I/O)
  gguf_io.py      — TensorSource (sliced reads) + IncrementalWriter (streamed output) + metadata copy
  engine.py       — planner (chunk sizing from budget), streaming loop, RSS telemetry, report
  cli.py          — argparse entry point, size parsing
tests/
  conftest.py     — tiny GGUF fixture builder
  test_q8_0.py
  test_gguf_io.py
  test_engine.py
  test_cli.py
scripts/
  baseline.sh           — Phase 0: pinned llama-quantize reference run + peak RSS
  memlimit_run.sh       — run featherquant under systemd cgroup MemoryMax ceiling
  compare_reference.py  — tensor-by-tensor GGUF diff (types + bytes)
pyproject.toml
.gitignore
```

---

### Task 1: Project scaffold + Q8_0 kernel

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `featherquant/__init__.py`, `featherquant/q8_0.py`
- Test: `tests/test_q8_0.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `featherquant.q8_0` exporting `BLOCK: int = 32`, `TYPE_SIZE: int = 34`, `quantize_q8_0(x: np.ndarray) -> bytes` (1-D float32, len % 32 == 0), `dequantize_q8_0(raw: bytes) -> np.ndarray` (float32), `bf16_to_f32(raw: np.ndarray) -> np.ndarray` (uint16 in, float32 out). Later tasks import all of these.

- [ ] **Step 1: Initialize repo and environment**

```bash
cd "/mnt/c/Users/SulavKumarShresta/OneDrive - In.Corp Global Pte. Ltd/Documents/personal_projects/featherQuant"
git init
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Write `pyproject.toml` and `.gitignore`**

`pyproject.toml`:

```toml
[project]
name = "featherquant"
version = "0.0.1"
description = "Memory-bounded out-of-core GGUF quantization"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "gguf>=0.16,<0.18"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
featherquant = "featherquant.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["featherquant"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
*.gguf
*.report.json
baseline_*.txt
```

- [ ] **Step 3: Install**

Run: `pip install -e '.[dev]'`
Expected: installs numpy, gguf, pytest without error. (`featherquant/__init__.py` must exist first — create it empty now, along with an empty `featherquant/q8_0.py` so the package imports.)

- [ ] **Step 4: Write the failing tests**

`tests/test_q8_0.py`:

```python
import numpy as np
import pytest

from featherquant.q8_0 import BLOCK, TYPE_SIZE, quantize_q8_0, dequantize_q8_0, bf16_to_f32


def test_constants():
    assert BLOCK == 32 and TYPE_SIZE == 34


def test_known_block():
    # amax = 127 -> d = 1.0 exactly; -63.5 must round AWAY from zero (llama.cpp roundf)
    x = np.zeros(32, np.float32)
    x[0], x[1] = 127.0, -63.5
    raw = quantize_q8_0(x)
    assert len(raw) == TYPE_SIZE
    d = np.frombuffer(raw[:2], np.float16)[0]
    q = np.frombuffer(raw[2:], np.int8)
    assert d == np.float16(1.0)
    assert q[0] == 127 and q[1] == -64
    assert not q[2:].any()


def test_zero_block():
    raw = quantize_q8_0(np.zeros(32, np.float32))
    assert raw == b"\x00" * TYPE_SIZE


def test_roundtrip_error_bound():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024).astype(np.float32)
    y = dequantize_q8_0(quantize_q8_0(x))
    # per-block error <= ~0.5*d plus fp16 scale rounding (~127*d*2^-11)
    d = np.abs(x).reshape(-1, BLOCK).max(axis=1) / 127.0
    assert np.all(np.abs(x - y).reshape(-1, BLOCK) <= 0.6 * d[:, None] + 1e-7)


def test_chunked_equals_full():
    # blocks are independent: quantizing in arbitrary 32-multiple chunks
    # must be byte-identical to one-shot quantization
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32 * 100).astype(np.float32)
    full = quantize_q8_0(x)
    step = 32 * 7
    parts = b"".join(quantize_q8_0(x[i : i + step]) for i in range(0, x.size, step))
    assert parts == full


def test_rejects_bad_input():
    with pytest.raises(AssertionError):
        quantize_q8_0(np.zeros(33, np.float32))
    with pytest.raises(AssertionError):
        quantize_q8_0(np.zeros(32, np.float64))


def test_bf16_to_f32():
    f = np.array([1.0, -2.5, 0.0, 15.75], np.float32)  # all exactly representable in bf16
    raw = (f.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(bf16_to_f32(raw), f)
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_q8_0.py -v`
Expected: FAIL — `ImportError: cannot import name 'BLOCK'` (module is empty).

- [ ] **Step 6: Implement `featherquant/q8_0.py`**

```python
"""Q8_0 block quantization, byte-compatible with llama.cpp quantize_row_q8_0_ref.

Layout per block of 32 elements: fp16 scale d, then 32 int8 values.
d is computed in float32 (amax/127), stored as fp16, but 1/d uses the
float32 value — matching the C reference. Rounding is half-away-from-zero
(C roundf), NOT numpy's default half-to-even.
"""
import numpy as np

BLOCK = 32
TYPE_SIZE = 34  # 2-byte fp16 scale + 32 int8


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def quantize_q8_0(x: np.ndarray) -> bytes:
    assert x.dtype == np.float32 and x.ndim == 1 and x.size % BLOCK == 0
    blocks = x.reshape(-1, BLOCK)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d > 0)
    v = blocks * inv[:, None]
    q = np.trunc(v + np.copysign(np.float32(0.5), v)).astype(np.int8)
    out = np.empty((blocks.shape[0], TYPE_SIZE), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, TYPE_SIZE)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)  # shape (n, 1)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_q8_0.py -v`
Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore featherquant/ tests/
git commit -m "feat: project scaffold and Q8_0 reference-compatible kernel"
```

---

### Task 2: Sliced GGUF tensor reader

**Files:**
- Create: `featherquant/gguf_io.py`, `tests/conftest.py`
- Test: `tests/test_gguf_io.py`

**Interfaces:**
- Consumes: `featherquant.q8_0.bf16_to_f32`.
- Produces in `featherquant.gguf_io`:
  - `ALIGN: int = 32`, `ITEMSIZE: dict[GGMLQuantizationType, int]` mapping `{F32: 4, F16: 2, BF16: 2}`.
  - `class TensorSource`: `__init__(path: str)`, attribute `reader: gguf.GGUFReader`, attribute `tensors: list` (GGUFReader `ReaderTensor` objects, file order; each has `.name`, `.shape` (ggml order, `shape[0]` = contiguous row length), `.tensor_type`, `.n_elements`, `.n_bytes`, `.data_offset`), `read_rows_f32(tensor, row_start: int, n_rows: int, buf: bytearray) -> np.ndarray` (float32, length `n_rows*shape[0]`), `read_raw(tensor, byte_start: int, nbytes: int, buf: bytearray) -> memoryview`, `close()`.
  - Test helper (conftest): `make_gguf(path, tensors: dict[str, np.ndarray], arch='llama')`.

- [ ] **Step 1: Write fixture builder**

`tests/conftest.py`:

```python
import gguf
import numpy as np


def make_gguf(path, tensors, arch="llama"):
    """Write a tiny GGUF for tests. float16/float32 arrays get F16/F32 types
    automatically; in-memory buffering is fine at this size."""
    w = gguf.GGUFWriter(str(path), arch)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_gguf_io.py`:

```python
import gguf
import numpy as np

from featherquant.gguf_io import ITEMSIZE, TensorSource
from tests.conftest import make_gguf


def test_read_rows_f16(tmp_path):
    a = (np.arange(8 * 64, dtype=np.float32).reshape(8, 64) / 16).astype(np.float16)
    p = tmp_path / "m.gguf"
    make_gguf(p, {"w": a})
    src = TensorSource(str(p))
    t = src.tensors[0]
    assert t.name == "w"
    assert int(t.shape[0]) == 64  # ne0 = contiguous row length
    isz = ITEMSIZE[t.tensor_type]
    buf = bytearray(3 * 64 * isz)
    x = src.read_rows_f32(t, 2, 3, buf)
    assert x.dtype == np.float32
    assert np.array_equal(x, a[2:5].astype(np.float32).ravel())
    src.close()


def test_read_rows_f32_and_raw(tmp_path):
    a = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    p = tmp_path / "m.gguf"
    make_gguf(p, {"w": a})
    src = TensorSource(str(p))
    t = src.tensors[0]
    buf = bytearray(4 * 32 * 4)
    assert np.array_equal(src.read_rows_f32(t, 0, 4, buf), a.ravel())
    raw = src.read_raw(t, 32 * 4, 32 * 4, buf)  # second row's bytes
    assert bytes(raw) == a[1].tobytes()
    src.close()


def test_read_rows_bf16(tmp_path):
    f = (np.arange(64, dtype=np.float32) / 4).reshape(2, 32)  # exact in bf16
    u16 = (f.view(np.uint32) >> 16).astype(np.uint16)
    p = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(p), "llama")
    w.add_tensor("w", u16.view(np.uint8).reshape(2, 64),
                 raw_shape=(2, 32), raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    src = TensorSource(str(p))
    buf = bytearray(2 * 32 * 2)
    x = src.read_rows_f32(src.tensors[0], 0, 2, buf)
    assert np.array_equal(x, f.ravel())
    src.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_gguf_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.gguf_io'`.

- [ ] **Step 4: Implement the reader half of `featherquant/gguf_io.py`**

```python
"""GGUF I/O: sliced tensor reads and incremental streamed output.

Reads use an explicit file handle with seek+readinto into caller-owned
buffers — never the GGUFReader memmap — so resident memory is exactly
the buffer the caller sized, not whatever pages mmap happened to touch.
GGUFReader is used for metadata only.
"""
import numpy as np
from gguf import (GGML_QUANT_VERSION, GGMLQuantizationType, GGUFReader,
                  GGUFValueType, GGUFWriter)

from .q8_0 import bf16_to_f32

ALIGN = 32  # default GGUF tensor-data alignment

ITEMSIZE = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.BF16: 2,
}


class TensorSource:
    def __init__(self, path: str):
        self.reader = GGUFReader(path)
        self.tensors = list(self.reader.tensors)
        self.f = open(path, "rb")

    def close(self):
        self.f.close()

    def read_rows_f32(self, tensor, row_start: int, n_rows: int, buf: bytearray) -> np.ndarray:
        ne0 = int(tensor.shape[0])
        isz = ITEMSIZE[tensor.tensor_type]
        nb = n_rows * ne0 * isz
        self.f.seek(int(tensor.data_offset) + row_start * ne0 * isz)
        got = self.f.readinto(memoryview(buf)[:nb])
        assert got == nb, f"short read on {tensor.name}: {got}/{nb}"
        n = n_rows * ne0
        tt = tensor.tensor_type
        if tt == GGMLQuantizationType.F32:
            return np.frombuffer(buf, np.float32, count=n)
        if tt == GGMLQuantizationType.F16:
            return np.frombuffer(buf, np.float16, count=n).astype(np.float32)
        return bf16_to_f32(np.frombuffer(buf, np.uint16, count=n))

    def read_raw(self, tensor, byte_start: int, nbytes: int, buf: bytearray) -> memoryview:
        self.f.seek(int(tensor.data_offset) + byte_start)
        got = self.f.readinto(memoryview(buf)[:nbytes])
        assert got == nbytes, f"short read on {tensor.name}: {got}/{nbytes}"
        return memoryview(buf)[:nbytes]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gguf_io.py -v`
Expected: 3 passed. If `test_read_rows_f16` fails on the `shape[0] == 64` assert, GGUFReader's shape convention differs from expected — fix `read_rows_f32` to use the contiguous dimension GGUFReader actually reports (check `t.data.shape` against the fixture), do not weaken the test's `array_equal` checks.

- [ ] **Step 6: Commit**

```bash
git add featherquant/gguf_io.py tests/conftest.py tests/test_gguf_io.py
git commit -m "feat: sliced GGUF tensor reader with fixed reusable buffers"
```

---

### Task 3: Incremental GGUF writer + metadata copy

**Files:**
- Modify: `featherquant/gguf_io.py` (append)
- Test: `tests/test_gguf_io.py` (append)

**Interfaces:**
- Consumes: `TensorSource`/`GGUFReader` from Task 2, `quantize_q8_0` from Task 1 (tests only).
- Produces in `featherquant.gguf_io`:
  - `copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None`.
  - `class IncrementalWriter`: `__init__(path: str, reader: GGUFReader, file_type: gguf.LlamaFileType)`, `add_tensor_info(name: str, ggml_shape, nbytes: int, ggml_type: GGMLQuantizationType)` (ggml_shape as reader reports it, ne-order), `begin_data()`, `begin_tensor()`, `write(data: bytes | memoryview)`, `close()`. Call order: all `add_tensor_info` first, then `begin_data()`, then per tensor in the same order: `begin_tensor()` + one or more `write()`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_gguf_io.py`)**

```python
from gguf import GGUFReader, LlamaFileType

from featherquant.gguf_io import IncrementalWriter
from featherquant.q8_0 import quantize_q8_0


def test_incremental_writer_roundtrip(tmp_path):
    src_arr = (np.arange(4 * 64, dtype=np.float32).reshape(4, 64) / 8).astype(np.float16)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"w": src_arr})
    reader = GGUFReader(str(sp))
    out = tmp_path / "out.gguf"
    iw = IncrementalWriter(str(out), reader, LlamaFileType.MOSTLY_Q8_0)
    t = reader.tensors[0]
    payload = quantize_q8_0(src_arr.astype(np.float32).ravel())
    iw.add_tensor_info(t.name, t.shape, len(payload), gguf.GGMLQuantizationType.Q8_0)
    iw.begin_data()
    iw.begin_tensor()
    iw.write(payload[:170])   # two chunks proves streamed writes land correctly
    iw.write(payload[170:])
    iw.close()

    r2 = GGUFReader(str(out))
    t2 = r2.tensors[0]
    assert t2.name == "w"
    assert t2.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert [int(d) for d in t2.shape] == [64, 4]
    assert t2.data.tobytes() == payload
    assert int(r2.fields["general.file_type"].contents()) == int(LlamaFileType.MOSTLY_Q8_0)
    assert r2.fields["general.architecture"].contents() == "llama"


def test_two_tensor_alignment_and_copy(tmp_path):
    # Q8_0 payloads are 34-byte multiples (not 32-aligned): second tensor
    # exercises inter-tensor padding. Second tensor is a verbatim F32 copy.
    a = np.ones((1, 32), np.float16)
    b = np.arange(32, dtype=np.float32)
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"a": a, "b": b})
    reader = GGUFReader(str(sp))
    ta, tb = reader.tensors[0], reader.tensors[1]
    out = tmp_path / "out.gguf"
    iw = IncrementalWriter(str(out), reader, LlamaFileType.MOSTLY_Q8_0)
    pa = quantize_q8_0(a.astype(np.float32).ravel())
    iw.add_tensor_info(ta.name, ta.shape, len(pa), gguf.GGMLQuantizationType.Q8_0)
    iw.add_tensor_info(tb.name, tb.shape, int(tb.n_bytes), tb.tensor_type)
    iw.begin_data()
    iw.begin_tensor()
    iw.write(pa)
    iw.begin_tensor()
    iw.write(b.tobytes())
    iw.close()

    r2 = GGUFReader(str(out))
    by_name = {t.name: t for t in r2.tensors}
    assert by_name["a"].data.tobytes() == pa
    assert by_name["b"].tensor_type == gguf.GGMLQuantizationType.F32
    assert np.array_equal(by_name["b"].data.reshape(-1), b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gguf_io.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'IncrementalWriter'`.

- [ ] **Step 3: Implement writer half (append to `featherquant/gguf_io.py`)**

```python
# KV keys the writer sets itself; copying them would duplicate/conflict.
_SKIP_KEYS = {"general.architecture", "general.file_type", "general.quantization_version"}


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    # ponytail: contents() materializes tokenizer lists (~tens of MB for big
    # vocabs); stream the KV copy if it ever threatens the budget.
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name in _SKIP_KEYS:
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_key_value(field.name, field.contents(), vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(field.name, field.contents(), vtype)


class IncrementalWriter:
    """GGUFWriter handles header/KV/tensor-info (and thus precomputes every
    tensor's offset with 32-byte alignment); we stream the tensor data
    ourselves in declaration order with the same alignment rule, so data
    lands exactly at the precomputed offsets."""

    def __init__(self, path: str, reader: GGUFReader, file_type):
        arch = reader.fields["general.architecture"].contents()
        self.w = GGUFWriter(path, arch)
        copy_metadata(reader, self.w)
        self.w.add_file_type(file_type)
        self.w.add_quantization_version(GGML_QUANT_VERSION)
        self.f = None

    def add_tensor_info(self, name: str, ggml_shape, nbytes: int, ggml_type) -> None:
        # GGUFReader reports shape in ggml ne-order; add_tensor_info expects
        # numpy order and reverses it back when writing.
        np_shape = [int(d) for d in reversed(list(ggml_shape))]
        self.w.add_tensor_info(name, np_shape, np.float32, int(nbytes), raw_dtype=ggml_type)

    def begin_data(self) -> None:
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_ti_data_to_file()
        self.f = self.w.fout[0]

    def begin_tensor(self) -> None:
        pad = (-self.f.tell()) % ALIGN
        if pad:
            self.f.write(b"\x00" * pad)

    def write(self, data) -> None:
        self.f.write(data)

    def close(self) -> None:
        self.f.flush()
        self.w.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gguf_io.py -v`
Expected: 5 passed. Known version-sensitive spots if it fails instead:
- `self.w.fout[0]` — on some gguf versions `fout` is a single file object, not a list; then use `self.w.fout`.
- `add_key_value(..., sub_type=...)` — if the installed gguf predates `sub_type`, the pin `gguf>=0.16,<0.18` in pyproject is not being honored; fix the environment, not the code.

- [ ] **Step 5: Commit**

```bash
git add featherquant/gguf_io.py tests/test_gguf_io.py
git commit -m "feat: incremental GGUF writer with streamed aligned tensor data"
```

---

### Task 4: Budget planner + streaming engine

**Files:**
- Create: `featherquant/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `TensorSource`, `IncrementalWriter`, `ITEMSIZE` (Task 2/3); `BLOCK`, `TYPE_SIZE`, `quantize_q8_0` (Task 1).
- Produces in `featherquant.engine`:
  - `RESERVE: int` (64 MiB safety reserve), `rss_bytes() -> int`.
  - `target_type(tensor) -> GGMLQuantizationType`, `q8_0_nbytes(n_elements: int) -> int`, `per_row_cost(ne0: int, itemsize: int) -> int`.
  - `quantize_model(src: str, dst: str, max_ram: int, report: str | None = None, _force_chunk_rows: int | None = None) -> dict` — returns stats dict with keys `max_ram, reserve, working_budget, bytes_read, bytes_written, peak_rss, budget_violations, chunks, elapsed_s`. Exits via `SystemExit` with a message naming the minimum feasible budget when the budget cannot fit one row. Task 5's CLI calls this.

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader

from featherquant.engine import (RESERVE, per_row_cost, q8_0_nbytes,
                                 quantize_model, rss_bytes, target_type)
from featherquant.q8_0 import quantize_q8_0
from tests.conftest import make_gguf


def test_q8_0_nbytes():
    assert q8_0_nbytes(64) == 2 * 34


def test_per_row_cost_positive():
    assert per_row_cost(4096, 2) > 4096 * 2  # read buf plus working temps


def _make_model(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((10, 64)).astype(np.float16)
    norm = rng.standard_normal(64).astype(np.float32)  # 1-D: must be copied, not quantized
    odd = rng.standard_normal((4, 30)).astype(np.float16)  # row % 32 != 0: copied
    sp = tmp_path / "src.gguf"
    make_gguf(sp, {"blk.0.attn_q.weight": w,
                   "blk.0.attn_norm.weight": norm,
                   "blk.0.odd.weight": odd})
    return sp, w, norm, odd


def test_engine_streams_and_matches_in_memory_reference(tmp_path):
    sp, w, norm, odd = _make_model(tmp_path)
    big = rss_bytes() + (512 << 20)
    o1, o2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    s1 = quantize_model(str(sp), str(o1), big, _force_chunk_rows=3)  # forces 4 chunks
    quantize_model(str(sp), str(o2), big)  # one chunk
    assert o1.read_bytes() == o2.read_bytes()  # chunking must not change output
    assert s1["chunks"] >= 4 and s1["peak_rss"] <= big

    r = GGUFReader(str(o1))
    by_name = {t.name: t for t in r.tensors}
    tq = by_name["blk.0.attn_q.weight"]
    assert tq.tensor_type == GGMLQuantizationType.Q8_0
    assert tq.data.tobytes() == quantize_q8_0(w.astype(np.float32).ravel())
    tn = by_name["blk.0.attn_norm.weight"]
    assert tn.tensor_type == GGMLQuantizationType.F32
    assert np.array_equal(tn.data.reshape(-1), norm)
    to = by_name["blk.0.odd.weight"]
    assert to.tensor_type == GGMLQuantizationType.F16
    assert np.array_equal(to.data.reshape(-1), odd.ravel())


def test_deterministic_across_runs(tmp_path):
    sp, *_ = _make_model(tmp_path)
    big = rss_bytes() + (512 << 20)
    o1, o2 = tmp_path / "a.gguf", tmp_path / "b.gguf"
    quantize_model(str(sp), str(o1), big)
    quantize_model(str(sp), str(o2), big)
    assert o1.read_bytes() == o2.read_bytes()


def test_impossible_budget_exits_with_minimum(tmp_path):
    sp, *_ = _make_model(tmp_path)
    with pytest.raises(SystemExit):
        quantize_model(str(sp), str(tmp_path / "o.gguf"), max_ram=RESERVE)


def test_report_written(tmp_path):
    import json
    sp, *_ = _make_model(tmp_path)
    rp = tmp_path / "r.json"
    quantize_model(str(sp), str(tmp_path / "o.gguf"),
                   rss_bytes() + (512 << 20), report=str(rp))
    stats = json.loads(rp.read_text())
    assert stats["bytes_read"] > 0 and stats["peak_rss"] > 0 and "elapsed_s" in stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.engine'`.

- [ ] **Step 3: Implement `featherquant/engine.py`**

```python
"""Budget-planned streaming quantization loop.

Working set per chunk = source read buffer + numpy temporaries in
quantize_q8_0 + packed output. The planner sizes the largest row chunk
whose estimated cost fits (max_ram - rss_at_start - RESERVE), and RSS is
sampled after every tensor for the report.
"""
import json
import os
import sys
import time

from gguf import GGMLQuantizationType, LlamaFileType

from .gguf_io import ITEMSIZE, IncrementalWriter, TensorSource
from .q8_0 import BLOCK, TYPE_SIZE, quantize_q8_0

RESERVE = 64 << 20  # ponytail: fixed 64 MiB reserve; adaptive governor is Phase 3
COPY_CHUNK = 8 << 20


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def target_type(t):
    if len(t.shape) >= 2 and t.tensor_type in ITEMSIZE and int(t.shape[0]) % BLOCK == 0:
        return GGMLQuantizationType.Q8_0
    return t.tensor_type


def q8_0_nbytes(n_elements: int) -> int:
    return n_elements // BLOCK * TYPE_SIZE


def per_row_cost(ne0: int, itemsize: int) -> int:
    # ponytail: crude static model — read buf + ~5 float32-sized numpy
    # temporaries inside quantize_q8_0 + packed row; replaced by measured
    # feedback in the Phase 3 adaptive controller.
    return ne0 * itemsize + 20 * ne0 + (ne0 // BLOCK) * TYPE_SIZE


def quantize_model(src, dst, max_ram, report=None, _force_chunk_rows=None):
    t0 = time.monotonic()
    source = TensorSource(src)
    working = max_ram - rss_bytes() - RESERVE
    if working <= 0:
        sys.exit(f"budget {max_ram} B too small: runtime already at "
                 f"{rss_bytes()} B RSS + {RESERVE} B reserve")
    stats = {"max_ram": max_ram, "reserve": RESERVE, "working_budget": working,
             "bytes_read": 0, "bytes_written": 0, "peak_rss": rss_bytes(),
             "budget_violations": 0, "chunks": 0}

    plan = []
    for t in source.tensors:
        tt = target_type(t)
        nbytes = q8_0_nbytes(int(t.n_elements)) if tt != t.tensor_type else int(t.n_bytes)
        plan.append((t, tt, nbytes))

    iw = IncrementalWriter(dst, source.reader, LlamaFileType.MOSTLY_Q8_0)
    for t, tt, nbytes in plan:
        iw.add_tensor_info(t.name, t.shape, nbytes, tt)
    iw.begin_data()

    for t, tt, _ in plan:
        iw.begin_tensor()
        if tt != t.tensor_type:
            _stream_quantize(source, iw, t, working, stats, _force_chunk_rows)
        else:
            _stream_copy(source, iw, t, stats)
        r = rss_bytes()
        stats["peak_rss"] = max(stats["peak_rss"], r)
        if r > max_ram:
            stats["budget_violations"] += 1

    iw.close()
    source.close()
    stats["elapsed_s"] = round(time.monotonic() - t0, 3)
    if report:
        with open(report, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def _stream_quantize(source, iw, t, working, stats, force_rows):
    ne0 = int(t.shape[0])
    rows = int(t.n_elements) // ne0
    isz = ITEMSIZE[t.tensor_type]
    chunk = force_rows or min(rows, working // per_row_cost(ne0, isz))
    if chunk < 1:
        sys.exit(f"budget too small for tensor {t.name}: one row needs about "
                 f"{per_row_cost(ne0, isz)} B working set on top of runtime + reserve")
    buf = bytearray(chunk * ne0 * isz)
    for r0 in range(0, rows, chunk):
        n = min(chunk, rows - r0)
        packed = quantize_q8_0(source.read_rows_f32(t, r0, n, buf))
        iw.write(packed)
        stats["bytes_read"] += n * ne0 * isz
        stats["bytes_written"] += len(packed)
        stats["chunks"] += 1


def _stream_copy(source, iw, t, stats):
    remaining = int(t.n_bytes)
    pos = 0
    buf = bytearray(min(remaining, COPY_CHUNK))
    while remaining:
        n = min(COPY_CHUNK, remaining)
        iw.write(source.read_raw(t, pos, n, buf))
        pos += n
        remaining -= n
        stats["bytes_read"] += n
        stats["bytes_written"] += n
        stats["chunks"] += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v`
Expected: 6 passed. If `blk.0.odd.weight` comes back Q8_0, `target_type` is checking the wrong shape dimension — the row (contiguous) dim must be `shape[0]` in ggml order.

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass (18 total).

- [ ] **Step 6: Commit**

```bash
git add featherquant/engine.py tests/test_engine.py
git commit -m "feat: budget-planned streaming quantization engine with RSS telemetry"
```

---

### Task 5: CLI

**Files:**
- Create: `featherquant/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `featherquant.engine.quantize_model`.
- Produces: `featherquant.cli.parse_size(s: str) -> int`; `featherquant.cli.main()` (argparse; flags `--model`, `--output`, `--format` [only `q8_0`], `--max-ram`, `--report`); console script `featherquant` and `python -m featherquant.cli` both work.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
import json
import sys

import numpy as np
import pytest

from featherquant import cli
from featherquant.engine import rss_bytes
from tests.conftest import make_gguf


def test_parse_size():
    assert cli.parse_size("2GB") == 2 << 30
    assert cli.parse_size("1.5GiB") == int(1.5 * (1 << 30))
    assert cli.parse_size("512M") == 512 << 20
    assert cli.parse_size("64KB") == 64 << 10
    assert cli.parse_size("1024") == 1024
    with pytest.raises(Exception):
        cli.parse_size("lots")


def test_cli_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src.gguf"
    make_gguf(src, {"w": np.ones((4, 64), np.float16)})
    out = tmp_path / "out.gguf"
    rp = tmp_path / "out.report.json"
    budget = str(rss_bytes() + (512 << 20))
    monkeypatch.setattr(sys, "argv",
                        ["featherquant", "--model", str(src), "--output", str(out),
                         "--format", "q8_0", "--max-ram", budget, "--report", str(rp)])
    cli.main()
    assert out.exists()
    stats = json.loads(rp.read_text())
    assert stats["peak_rss"] <= int(budget)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'featherquant.cli'` (or missing attribute).

- [ ] **Step 3: Implement `featherquant/cli.py`**

```python
import argparse
import json
import re

from .engine import quantize_model


def parse_size(s: str) -> int:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT])?I?B?", s.strip(), re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s!r} (try 2GB, 512M, 1024)")
    exp = "KMGT".find(m[2].upper()) + 1 if m[2] else 0
    return int(float(m[1]) * 1024 ** exp)


def main():
    p = argparse.ArgumentParser(prog="featherquant",
                                description="Memory-bounded GGUF quantization")
    p.add_argument("--model", required=True, help="source F16/BF16 GGUF")
    p.add_argument("--output", required=True, help="output GGUF path")
    p.add_argument("--format", default="q8_0", choices=["q8_0"])
    p.add_argument("--max-ram", required=True, type=parse_size,
                   help="peak RSS budget, e.g. 2GB")
    p.add_argument("--report", help="write JSON stats here")
    a = p.parse_args()
    stats = quantize_model(a.model, a.output, a.max_ram, report=a.report)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify the console entry point**

Run: `pip install -e . -q && featherquant --help`
Expected: usage text listing `--model --output --format --max-ram --report`.

- [ ] **Step 6: Commit**

```bash
git add featherquant/cli.py tests/test_cli.py
git commit -m "feat: featherquant CLI with human-readable RAM budget"
```

---

### Task 6: External memory-ceiling harness

**Files:**
- Create: `scripts/memlimit_run.sh`

**Interfaces:**
- Consumes: the `featherquant.cli` module (`python -m featherquant.cli`).
- Produces: `scripts/memlimit_run.sh <model.gguf> <out.gguf> [MemoryMax] [budget]` — exits 0 iff the whole job completed inside an OS-enforced cgroup v2 ceiling. Task 7 uses it for the real-model proof.

- [ ] **Step 1: Write the script**

`scripts/memlimit_run.sh`:

```bash
#!/usr/bin/env bash
# Run featherquant inside an OS-enforced memory ceiling (cgroup v2 via
# systemd-run --user). Proves the "externally enforced bound" claim:
# if RSS ever exceeds MemoryMax the kernel OOM-kills the job and we exit
# non-zero. Swap is disabled inside the scope so the ceiling is honest.
#
# Usage: scripts/memlimit_run.sh MODEL OUT [MEMORY_MAX] [BUDGET]
#   MEMORY_MAX  systemd size (default 1G) — the hard external ceiling
#   BUDGET      featherquant --max-ram   (default same as MEMORY_MAX)
set -uo pipefail
MODEL=$1; OUT=$2; LIMIT=${3:-1G}; BUDGET=${4:-$LIMIT}
PY=$(command -v python)

if systemd-run --user --scope --wait --collect --same-dir \
    -p MemoryMax="$LIMIT" -p MemorySwapMax=0 \
    "$PY" -m featherquant.cli --model "$MODEL" --output "$OUT" \
    --format q8_0 --max-ram "$BUDGET" --report "${OUT%.gguf}.report.json"; then
  echo "PASS: completed inside $LIMIT external ceiling"
else
  echo "FAIL: killed or errored under $LIMIT ceiling (OOM if exit=137)"
  exit 1
fi
```

Run: `chmod +x scripts/memlimit_run.sh`

- [ ] **Step 2: Verify the ceiling actually kills over-budget processes**

Run:
```bash
systemd-run --user --scope --wait -p MemoryMax=100M -p MemorySwapMax=0 \
  python -c "x = bytearray(500 << 20)"; echo "exit=$?"
```
Expected: non-zero exit (OOM-killed) — proves enforcement works on this machine. If `systemd-run --user` is unavailable under WSL2, enable systemd in `/etc/wsl.conf` (`[boot] systemd=true`) and restart WSL; that is an environment fix, not a code change.

- [ ] **Step 3: Verify a passing run under the ceiling**

```bash
python - <<'EOF'
import numpy as np
from tests.conftest import make_gguf
rng = np.random.default_rng(0)
make_gguf("smoke_src.gguf", {"w": rng.standard_normal((256, 1024)).astype(np.float16)})
EOF
scripts/memlimit_run.sh smoke_src.gguf smoke_out.gguf 1G
```
Expected: JSON stats printed, then `PASS: completed inside 1G external ceiling`. Clean up: `rm smoke_src.gguf smoke_out.gguf smoke_out.report.json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/memlimit_run.sh
git commit -m "feat: external cgroup memory-ceiling harness"
```

---

### Task 7: Real-model baseline, reference equivalence, and smoke test

**Files:**
- Create: `scripts/baseline.sh`, `scripts/compare_reference.py`

**Interfaces:**
- Consumes: everything above; plus the user's existing llama.cpp checkout (env var `LLAMA_CPP_DIR`, binaries in `$LLAMA_CPP_DIR/build/bin/`) and the existing Qwen3-0.6B BF16 GGUF (env var `SRC_GGUF`).
- Produces: recorded Phase 0 baseline (`baseline_commit.txt`, `baseline_time.txt`), `scripts/compare_reference.py <a.gguf> <b.gguf>` exiting 0 iff tensor sets, types, and bytes all match, and the success-criteria checklist below.

- [ ] **Step 1: Write `scripts/baseline.sh` (Phase 0)**

```bash
#!/usr/bin/env bash
# Phase 0 baseline: pinned llama-quantize Q8_0 run with peak-RSS capture.
# Usage: LLAMA_CPP_DIR=~/llama.cpp scripts/baseline.sh SRC.gguf REF_OUT.gguf
set -euo pipefail
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout with built tools}"
SRC=$1; REF_OUT=$2
git -C "$LLAMA_CPP_DIR" rev-parse HEAD | tee baseline_commit.txt
/usr/bin/time -v "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$SRC" "$REF_OUT" Q8_0 \
  2> baseline_time.txt
grep -E 'Maximum resident|Elapsed' baseline_time.txt
sha256sum "$REF_OUT"
```

Run: `chmod +x scripts/baseline.sh`

- [ ] **Step 2: Write `scripts/compare_reference.py`**

```python
#!/usr/bin/env python3
"""Tensor-by-tensor GGUF comparison: names, types, bytes.

Usage: python scripts/compare_reference.py reference.gguf featherquant.gguf
Exit 0 iff everything matches. Byte comparison is chunked so this also
works on models bigger than RAM (memmap slices, 64 MiB at a time).
"""
import sys

import numpy as np
from gguf import GGUFReader


def chunks_equal(a, b):
    x, y = a.reshape(-1), b.reshape(-1)
    if x.dtype != y.dtype or x.size != y.size:
        return False
    step = max(1, (64 << 20) // x.itemsize)
    return all(np.array_equal(x[i:i + step], y[i:i + step])
               for i in range(0, x.size, step))


def main():
    ra, rb = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    bad = 0
    if ta.keys() != tb.keys():
        print("tensor name sets differ:", sorted(ta.keys() ^ tb.keys()))
        bad += 1
    for name in sorted(ta.keys() & tb.keys()):
        a, b = ta[name], tb[name]
        if a.tensor_type != b.tensor_type:
            print(f"{name}: type {a.tensor_type.name} != {b.tensor_type.name}")
            bad += 1
        elif not chunks_equal(a.data, b.data):
            print(f"{name}: byte mismatch")
            bad += 1
    print(f"{len(ta)} tensors, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the Phase 0 baseline on the real model**

```bash
export LLAMA_CPP_DIR=<path to llama.cpp checkout>
export SRC_GGUF=<path to Qwen3-0.6B BF16 gguf>
scripts/baseline.sh "$SRC_GGUF" ref_q8_0.gguf
```
Expected: prints commit hash, `Maximum resident set size`, elapsed time, and the reference output's sha256. Record all three in `docs/superpowers/plans/2026-07-20-baseline-notes.md` (freeform).

- [ ] **Step 4: Run featherquant on the same model under the external ceiling**

The BF16 source is ~1.5 GB; a 1G ceiling makes source > ceiling, satisfying success criterion 1 at small scale.

```bash
scripts/memlimit_run.sh "$SRC_GGUF" fq_q8_0.gguf 1G
```
Expected: `PASS: completed inside 1G external ceiling`, and `fq_q8_0.report.json` shows `peak_rss` ≤ budget with `budget_violations: 0`.

- [ ] **Step 5: Compare against the reference**

Run: `python scripts/compare_reference.py ref_q8_0.gguf fq_q8_0.gguf`
Expected: `N tensors, 0 mismatches`, exit 0.

If types mismatch: llama-quantize applied a per-tensor rule this prototype's `target_type` doesn't replicate (e.g. keeping `output.weight` or `token_embd.weight` at a different type). Fix by extending `target_type` in `featherquant/engine.py` with a name-based rule matching the observed reference types, add a unit test for the rule in `tests/test_engine.py`, and re-run. Do not weaken the comparison.

If bytes mismatch on Q8_0 tensors only: rounding parity bug in `quantize_q8_0`. Extract one mismatching 32-element block (via GGUFReader slicing on both files plus the source), add it as a regression test in `tests/test_q8_0.py`, fix the kernel, re-run.

- [ ] **Step 6: Inference smoke test**

```bash
"$LLAMA_CPP_DIR/build/bin/llama-cli" -m fq_q8_0.gguf -p "The capital of France is" -n 8 --seed 1
```
Expected: model loads and generates coherent tokens, no GGUF validation errors.

- [ ] **Step 7: Record results and commit**

Append to `docs/superpowers/plans/2026-07-20-baseline-notes.md`: featherquant peak RSS vs baseline peak RSS, elapsed times, ceiling used, compare result, smoke output. Then:

```bash
git add scripts/baseline.sh scripts/compare_reference.py docs/
git commit -m "feat: baseline capture and reference-equivalence validation"
```

---

## Success criteria mapping (spec §14, prototype scale)

1. Source (~1.5 GB) > enforced RAM (1 G ceiling) — Task 7 Step 4.
2. Completes inside external ceiling — Tasks 6–7 (`memlimit_run.sh`).
3. Tensor larger than working budget — chunked row streaming, proven by `_force_chunk_rows` equivalence test (Task 4) and real run chunk counts.
4. Valid loadable model — Task 7 Step 6 smoke test.
5. Matches reference within tolerance — Task 7 Step 5 (byte-exact target).
6. Recovery after interruption — **out of scope**, Phase 4 plan.
7. RAM/runtime trade-off characterization — report JSON provides the metrics; full experiment matrix is a later plan.
8. Reproducible manifest — partial (report JSON + deterministic output); full manifest is Phase 4.
