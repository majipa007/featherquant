"""Safetensors input: minimal shard parser and row-sliced reads.

The safetensors format is trivial — an 8-byte little-endian header length,
a JSON header of ``{name: {dtype, shape, data_offsets}}``, then raw
row-major tensor data. Parsing it directly (no ``safetensors`` dependency)
gives exact control over reads: tensor bytes move through caller-owned
buffers just like the GGUF path, never through mmap or full-tensor loads.
"""
import io
import json
import os
import struct
from dataclasses import dataclass

import numpy as np
from gguf import MODEL_ARCH_NAMES, GGMLQuantizationType, GGUFReader, get_tensor_name_map

from .q8_0 import bf16_to_f32

# Byte size per element for supported safetensors dtypes.
ST_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True)
class StTensor:
    """One tensor as declared in a shard header (offsets are data-relative)."""
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


def parse_shard_header(path: str) -> tuple[dict[str, StTensor], int]:
    """Parse one shard's JSON header.

    Returns ({name: StTensor}, data_base) where data_base is the absolute
    file offset at which tensor data begins.
    """
    try:
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse safetensors header {path}: {exc}") from exc
    tensors: dict[str, StTensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = info["dtype"]
            if dtype not in ST_ITEMSIZE:
                raise RuntimeError(
                    f"unsupported safetensors dtype {dtype!r} for {name} "
                    f"in {path} (supported: {sorted(ST_ITEMSIZE)})")
            start, end = info["data_offsets"]
            tensors[name] = StTensor(name, dtype, tuple(info["shape"]),
                                     int(start), int(end))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed entry {name!r} in {path}: {exc}") from exc
    return tensors, 8 + hlen


def read_st_rows(f: io.BufferedReader, t: StTensor, data_base: int,
                 row_start: int, n_rows: int, buf: bytearray) -> np.ndarray:
    """Read ``n_rows`` rows (last-dim-major) into ``buf``; return float32.

    Rows run along the LAST dim (row-major layout), which matches GGUF's
    contiguous ne0 dimension after shape reversal.
    """
    row_len = t.shape[-1] if t.shape else 1
    isz = ST_ITEMSIZE[t.dtype]
    nb = n_rows * row_len * isz
    try:
        f.seek(data_base + t.start + row_start * row_len * isz)
        got = f.readinto(memoryview(buf)[:nb])
    except OSError as exc:
        raise RuntimeError(f"read error on tensor {t.name}: {exc}") from exc
    if got != nb:
        raise RuntimeError(f"short read on {t.name}: {got}/{nb} bytes")
    n = n_rows * row_len
    if t.dtype == "F32":
        return np.frombuffer(buf, np.float32, count=n)
    if t.dtype == "F16":
        return np.frombuffer(buf, np.float16, count=n).astype(np.float32)
    return bf16_to_f32(np.frombuffer(buf, np.uint16, count=n))


# Safetensors dtype -> ggml source type (the types the engine can stream).
_ST_TO_GGML = {
    "F32": GGMLQuantizationType.F32,
    "F16": GGMLQuantizationType.F16,
    "BF16": GGMLQuantizationType.BF16,
}

# Architectures whose HF checkpoints stream over 1:1 (no weight permutes).
# llama-family permutes attn_q/attn_k on conversion — out of scope for now.
_SUPPORTED_ARCHS = {"qwen3"}


@dataclass(frozen=True)
class StReaderTensor:
    """Duck-types the ReaderTensor fields the engine and formats use."""
    name: str                          # GGUF tensor name
    shape: tuple[int, ...]             # ggml ne-order (reversed HF shape)
    tensor_type: GGMLQuantizationType
    n_elements: int
    n_bytes: int
    st: StTensor                       # underlying safetensors entry
    shard: str                         # shard filename (routing key)


class SafetensorsSource:
    """Engine-compatible tensor source over a sharded HF checkpoint.

    Tensor DATA comes from the safetensors shards; model METADATA
    (architecture KVs + tokenizer) comes from a small ``--vocab-only`` GGUF
    (see ``scripts/make_vocab_gguf.sh``), exposed as ``self.reader`` so the
    IncrementalWriter copies it verbatim. Duck-types ``TensorSource``:
    ``reader``, ``tensors``, ``read_rows_f32``, ``read_raw``, ``close``,
    ``release_metadata``.
    """

    def __init__(self, model_dir: str, vocab_gguf: str):
        try:
            self.reader = GGUFReader(vocab_gguf)
        except Exception as exc:
            raise RuntimeError(
                f"cannot read vocab GGUF {vocab_gguf}: {exc}") from exc
        arch = str(self.reader.fields["general.architecture"].contents())
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(
                f"architecture {arch!r} is not supported for safetensors "
                f"input (supported: {sorted(_SUPPORTED_ARCHS)}); llama-family "
                "needs attn_q/attn_k permutation on load")
        bc = self.reader.fields.get(f"{arch}.block_count")
        n_blocks = int(bc.contents()) if bc is not None else 0
        arch_enum = next(k for k, v in MODEL_ARCH_NAMES.items() if v == arch)
        name_map = get_tensor_name_map(arch_enum, n_blocks)

        # Shard list: index.json when sharded, single file otherwise.
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            try:
                with open(index) as f:
                    weight_map = json.load(f)["weight_map"]
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(f"cannot read {index}: {exc}") from exc
            shard_names = sorted(set(weight_map.values()))
        else:
            shard_names = ["model.safetensors"]

        self._files: dict[str, io.BufferedReader] = {}
        self._data_base: dict[str, int] = {}
        self.tensors: list[StReaderTensor] = []
        unmapped: list[str] = []
        for shard in shard_names:
            path = os.path.join(model_dir, shard)
            entries, data_base = parse_shard_header(path)
            try:
                self._files[shard] = open(path, "rb")
            except OSError as exc:
                raise RuntimeError(f"cannot open shard {path}: {exc}") from exc
            self._data_base[shard] = data_base
            for hf_name, st in entries.items():  # header order = deterministic
                # get_name returns the mapped name WITH the matched suffix.
                gguf_name = name_map.get_name(hf_name,
                                              try_suffixes=(".weight", ".bias"))
                if gguf_name is None:
                    unmapped.append(hf_name)
                    continue
                n_elems = 1
                for d in st.shape:
                    n_elems *= d
                self.tensors.append(StReaderTensor(
                    name=gguf_name,
                    shape=tuple(reversed(st.shape)),
                    tensor_type=_ST_TO_GGML[st.dtype],
                    n_elements=n_elems,
                    n_bytes=st.end - st.start,
                    st=st, shard=shard))
        if unmapped:
            raise RuntimeError(
                f"no GGUF mapping for {len(unmapped)} tensor(s) in "
                f"{model_dir}: {unmapped[:5]} — refusing to emit a partial "
                "model")

    def read_rows_f32(self, tensor: StReaderTensor, row_start: int,
                      n_rows: int, buf: bytearray) -> np.ndarray:
        """Read rows via the owning shard (same contract as TensorSource)."""
        return read_st_rows(self._files[tensor.shard], tensor.st,
                            self._data_base[tensor.shard],
                            row_start, n_rows, buf)

    def read_raw(self, tensor: StReaderTensor, byte_start: int, nbytes: int,
                 buf: bytearray) -> memoryview:
        """Raw byte range of a tensor (verbatim copies)."""
        f = self._files[tensor.shard]
        try:
            f.seek(self._data_base[tensor.shard] + tensor.st.start + byte_start)
            got = f.readinto(memoryview(buf)[:nbytes])
        except OSError as exc:
            raise RuntimeError(f"read error on {tensor.name}: {exc}") from exc
        if got != nbytes:
            raise RuntimeError(f"short read on {tensor.name}: {got}/{nbytes}")
        return memoryview(buf)[:nbytes]

    def release_metadata(self) -> None:
        """Free the vocab GGUF's KV parse objects (same as TensorSource)."""
        self.reader.fields.clear()

    def close(self) -> None:
        """Close all shard handles."""
        for f in self._files.values():
            try:
                f.close()
            except OSError:
                pass  # best-effort
