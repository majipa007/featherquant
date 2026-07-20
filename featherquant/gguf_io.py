"""GGUF I/O: sliced tensor reads and incremental streamed output.

Reads use an explicit file handle with seek+readinto into caller-owned
buffers — never the GGUFReader memmap — so resident memory is exactly the
buffer the caller sized, not whatever pages mmap happened to touch.
GGUFReader is used for metadata only.
"""
from typing import Any, BinaryIO

import numpy as np
from gguf import (
    GGML_QUANT_VERSION,
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
    LlamaFileType,
)
from gguf.gguf_reader import ReaderTensor

from .q8_0 import bf16_to_f32

# Default GGUF tensor-data alignment (bytes).
ALIGN = 32

# Byte size per element for the source types we support.
ITEMSIZE = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.BF16: 2,
}


class TensorSource:
    """Read-only view of a GGUF file with row-sliced tensor access."""

    def __init__(self, path: str):
        try:
            # GGUFReader parses metadata via memmap; we never touch its
            # tensor data pages.
            self.reader = GGUFReader(path)
        except Exception as exc:
            raise RuntimeError(f"failed to parse GGUF metadata from {path}: {exc}") from exc
        self.tensors = list(self.reader.tensors)
        try:
            # Separate handle for explicit, budget-controlled data reads.
            self.f = open(path, "rb")
        except OSError as exc:
            raise RuntimeError(f"failed to open {path} for reading: {exc}") from exc

    def close(self) -> None:
        """Release the data file handle."""
        try:
            self.f.close()
        except OSError:
            pass  # closing is best-effort; nothing actionable on failure

    def release_metadata(self) -> None:
        """Free GGUFReader's KV parse objects (~0.5 GiB on 150k-token vocabs).

        GGUFReader materializes hundreds of thousands of numpy view objects
        for large tokenizer arrays and keeps them alive via ``fields``. Call
        this once all metadata reads are done — tensor slicing needs only
        the ReaderTensor shape/type/offset scalars, which stay valid.
        """
        self.reader.fields.clear()

    def read_rows_f32(self, tensor: "ReaderTensor | Any", row_start: int,
                      n_rows: int, buf: bytearray) -> np.ndarray:
        """Read ``n_rows`` rows starting at ``row_start`` into ``buf``.

        Returns a float32 array of length ``n_rows * ne0``.  The returned
        array may alias ``buf`` (F32 source) or be a fresh converted copy
        (F16/BF16 source) — callers must consume it before the next read.
        """
        ne0 = int(tensor.shape[0])  # ggml order: shape[0] = contiguous row length
        isz = ITEMSIZE[tensor.tensor_type]
        nb = n_rows * ne0 * isz
        try:
            self.f.seek(int(tensor.data_offset) + row_start * ne0 * isz)
            got = self.f.readinto(memoryview(buf)[:nb])
        except OSError as exc:
            raise RuntimeError(f"read error on tensor {tensor.name}: {exc}") from exc
        if got != nb:
            raise RuntimeError(f"short read on {tensor.name}: {got}/{nb} bytes")
        n = n_rows * ne0
        tt = tensor.tensor_type
        if tt == GGMLQuantizationType.F32:
            return np.frombuffer(buf, np.float32, count=n)
        if tt == GGMLQuantizationType.F16:
            return np.frombuffer(buf, np.float16, count=n).astype(np.float32)
        # Remaining supported type is BF16: reinterpret bits then widen.
        return bf16_to_f32(np.frombuffer(buf, np.uint16, count=n))

    def read_raw(self, tensor: "ReaderTensor | Any", byte_start: int,
                 nbytes: int, buf: bytearray) -> memoryview:
        """Read raw tensor bytes (for verbatim copies of unquantized tensors)."""
        try:
            self.f.seek(int(tensor.data_offset) + byte_start)
            got = self.f.readinto(memoryview(buf)[:nbytes])
        except OSError as exc:
            raise RuntimeError(f"read error on tensor {tensor.name}: {exc}") from exc
        if got != nbytes:
            raise RuntimeError(f"short read on {tensor.name}: {got}/{nbytes} bytes")
        return memoryview(buf)[:nbytes]


# KV keys the writer sets itself; copying them would duplicate/conflict.
_SKIP_KEYS = {"general.architecture", "general.file_type",
              "general.quantization_version"}


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy every KV field from a source GGUF into a GGUFWriter.

    Skips the virtual GGUF.* fields and keys the writer owns.
    """
    # ponytail: contents() materializes tokenizer lists (~tens of MB for big
    # vocabs); stream the KV copy if it ever threatens the budget.
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name in _SKIP_KEYS:
            continue
        try:
            vtype = field.types[0]
            if vtype == GGUFValueType.ARRAY:
                writer.add_key_value(field.name, field.contents(), vtype,
                                     sub_type=field.types[-1])
            else:
                writer.add_key_value(field.name, field.contents(), vtype)
        except Exception as exc:
            raise RuntimeError(f"failed to copy metadata key {field.name}: {exc}") from exc


class IncrementalWriter:
    """Streamed GGUF output with precomputed offsets.

    GGUFWriter handles header/KV/tensor-info (and thus precomputes every
    tensor's offset with 32-byte alignment); we stream the tensor data
    ourselves in declaration order with the same alignment rule, so data
    lands exactly at the precomputed offsets.

    Call order: all ``add_tensor_info`` first, then ``begin_data()``, then
    per tensor in the same order: ``begin_tensor()`` + one or more
    ``write()``, and finally ``close()``.
    """

    def __init__(self, path: str, reader: GGUFReader,
                 file_type: LlamaFileType):
        try:
            arch = reader.fields["general.architecture"].contents()
        except KeyError as exc:
            raise RuntimeError("source GGUF has no general.architecture key") from exc
        self.w = GGUFWriter(path, arch)
        copy_metadata(reader, self.w)
        # Mark the output's overall quantization scheme.
        self.w.add_file_type(file_type)
        self.w.add_quantization_version(GGML_QUANT_VERSION)
        # Underlying data file; only available after begin_data().
        self.f: BinaryIO | None = None

    def add_tensor_info(self, name: str, ggml_shape: Any, nbytes: int,
                        ggml_type: GGMLQuantizationType) -> None:
        """Declare one output tensor (shape as GGUFReader reports, ne-order)."""
        # GGUFReader reports shape in ggml ne-order; add_tensor_info expects
        # numpy order and reverses it back when writing.
        np_shape = [int(d) for d in reversed(list(ggml_shape))]
        self.w.add_tensor_info(name, np_shape, np.dtype(np.float32), int(nbytes),
                               raw_dtype=ggml_type)

    def begin_data(self) -> None:
        """Write header, KV data and tensor infos; open the data section."""
        try:
            self.w.write_header_to_file()
            self.w.write_kv_data_to_file()
            self.w.write_ti_data_to_file()
        except Exception as exc:
            raise RuntimeError(f"failed to write GGUF header/metadata: {exc}") from exc
        # Grab the underlying file handle to stream tensor data directly.
        assert self.w.fout is not None  # populated by write_header_to_file()
        self.f = self.w.fout[0]

    def _data_file(self) -> BinaryIO:
        """Return the open data file, or fail loudly if begin_data() never ran."""
        if self.f is None:
            raise RuntimeError("begin_data() must be called before writing tensors")
        return self.f

    def begin_tensor(self) -> None:
        """Pad to the 32-byte alignment boundary the offsets were computed with."""
        f = self._data_file()
        pad = (-f.tell()) % ALIGN
        if pad:
            f.write(b"\x00" * pad)

    def write(self, data: bytes | memoryview) -> None:
        """Append one chunk of the current tensor's packed data."""
        try:
            self._data_file().write(data)
        except OSError as exc:
            raise RuntimeError(f"write error (disk full?): {exc}") from exc

    def flush(self) -> None:
        """Flush buffered tensor data to disk (checkpoint boundary)."""
        self._data_file().flush()

    def tell(self) -> int:
        """Current absolute position in the output file."""
        return self._data_file().tell()

    def close(self) -> None:
        """Flush and close the output file. Safe to call at any lifecycle stage."""
        try:
            if self.f is not None:
                self.f.flush()
        finally:
            self.w.close()


class ResumeWriter:
    """Writer over an existing partially-written GGUF (resume path).

    Duck-types the subset of ``IncrementalWriter`` the streaming loop uses
    (``begin_tensor``/``write``/``flush``/``tell``/``close``). Header, KV and
    tensor infos already exist in the file from the interrupted run — this
    just repositions and continues the data section.
    """

    def __init__(self, path: str, position: int):
        try:
            self.f = open(path, "r+b")
            self.f.seek(position)
        except OSError as exc:
            raise RuntimeError(f"cannot reopen {path} for resume: {exc}") from exc

    def begin_tensor(self) -> None:
        """Pad to the alignment boundary (no-op when already aligned)."""
        pad = (-self.f.tell()) % ALIGN
        if pad:
            self.f.write(b"\x00" * pad)

    def write(self, data: bytes | memoryview) -> None:
        """Append one chunk of the current tensor's packed data."""
        try:
            self.f.write(data)
        except OSError as exc:
            raise RuntimeError(f"write error (disk full?): {exc}") from exc

    def flush(self) -> None:
        """Flush buffered tensor data to disk (checkpoint boundary)."""
        self.f.flush()

    def tell(self) -> int:
        """Current absolute position in the output file."""
        return self.f.tell()

    def close(self) -> None:
        """Flush and close the reopened file."""
        try:
            self.f.flush()
        finally:
            self.f.close()
