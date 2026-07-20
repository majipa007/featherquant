"""Output format registry: model-level preset -> per-tensor ggml type.

The Q4_K_M rules are encoded from empirical ground truth
(``tests/data/qwen3-0.6b-q4_k_m-types.json``, dumped from a reference
``llama-quantize Q4_K_M`` run) and match llama.cpp's ``use_more_bits``
layer schedule. ``test_q4_k_m_matches_reference_dump`` is the arbiter.
"""
from dataclasses import dataclass
from typing import Any, Callable

from gguf import GGMLQuantizationType, LlamaFileType

from .gguf_io import ITEMSIZE
from .q8_0 import BLOCK

# K-quant super-block length: rows must divide this to be K-quantizable.
QK_K = 256


def _layer_index(name: str) -> int | None:
    """Layer number from a 'blk.N.' tensor name, else None."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _quantizable(t: Any, blk: int) -> bool:
    """2-D+ float tensor whose contiguous row length divides the block."""
    return (len(t.shape) >= 2 and t.tensor_type in ITEMSIZE
            and int(t.shape[0]) % blk == 0)


def _use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama.cpp's Q6_K layer schedule: first/last eighth + every 3rd between."""
    return (i_layer < n_layers // 8 or i_layer >= 7 * n_layers // 8
            or (i_layer - n_layers // 8) % 3 == 2)


def _q8_0_rule(t: Any, n_layers: int) -> GGMLQuantizationType:
    """Everything quantizable becomes Q8_0; the rest is copied."""
    return (GGMLQuantizationType.Q8_0 if _quantizable(t, BLOCK)
            else GGMLQuantizationType(t.tensor_type))


def _q4_k_m_rule(t: Any, n_layers: int) -> GGMLQuantizationType:
    """Q4_K_M mix: Q4_K base, Q6_K for output/attn_v/ffn_down per schedule."""
    if not _quantizable(t, QK_K):
        # Rows not divisible by 256 cannot hold a K-quant super-block; keep
        # the source type (no such tensor appears in the reference dump).
        return GGMLQuantizationType(t.tensor_type)
    if t.name == "output.weight":
        return GGMLQuantizationType.Q6_K
    layer = _layer_index(t.name)
    if (layer is not None and n_layers > 0
            and (t.name.endswith("attn_v.weight")
                 or t.name.endswith("ffn_down.weight"))
            and _use_more_bits(layer, n_layers)):
        return GGMLQuantizationType.Q6_K
    return GGMLQuantizationType.Q4_K


@dataclass(frozen=True)
class FormatSpec:
    """One output preset: GGUF file type + per-tensor type rule."""
    file_type: LlamaFileType
    tensor_type: Callable[[Any, int], GGMLQuantizationType]
    needs_ggml: bool  # True when kernels come from the ctypes ggml backend


FORMATS: dict[str, FormatSpec] = {
    "q8_0": FormatSpec(LlamaFileType.MOSTLY_Q8_0, _q8_0_rule, needs_ggml=False),
    "q4_k_m": FormatSpec(LlamaFileType.MOSTLY_Q4_K_M, _q4_k_m_rule, needs_ggml=True),
}
