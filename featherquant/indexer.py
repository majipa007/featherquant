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
            # 2-D+ float tensors are the only quantizable ones (norms stay
            # in their native precision regardless of shape); whether a
            # specific block format's row-alignment feasibility holds is a
            # planner call (task 9), not decided here.
            eligible = (len(st.shape) >= 2 and role not in _NEVER_QUANT
                        and st.dtype in ST_ITEMSIZE)
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
    from gguf import GGUFReader  # local import: metadata path only

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
        # GQA models (Qwen3 among them) size heads independently of
        # hidden_size/n_heads; prefer the explicit KV and only fall back
        # to the even split when an architecture omits it.
        key_length = reader.fields.get(f"{arch}.attention.key_length")
        head_dim = int(key_length.contents()) if key_length is not None \
            else (hidden // n_heads if n_heads else 0)
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
            quant_eligible=(len(shape) >= 2 and role not in _NEVER_QUANT),
            layer_index=layer, role=role.value))
    reader.fields.clear()   # drop the KV object graph immediately
    return ModelIndex(
        model_arch=arch, n_layers=n_layers, hidden_size=hidden,
        intermediate_size=inter, vocab_size=vocab,
        head_dims={"n_heads": n_heads, "n_kv_heads": n_kv,
                   "head_dim": head_dim},
        tensors=tensors,
        largest_tensor_bytes=max(t.byte_length for t in tensors),
        total_bytes=sum(t.byte_length for t in tensors))


def index_model(model_path: str) -> ModelIndex:
    """Index an HF checkpoint directory or a GGUF file."""
    if os.path.isdir(model_path):
        return index_safetensors(model_path)
    return index_gguf(model_path)
