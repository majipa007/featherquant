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


def _classify(
    name: str,
    rules: list[tuple[re.Pattern[str], Role]],
    layer_re: re.Pattern[str],
) -> tuple[Role, int | None]:
    """Apply an ordered rule table; fail loudly when nothing matches."""
    m = layer_re.search(name)
    layer = int(m[1]) if m else None
    for pattern, role in rules:
        if pattern.search(name):
            return role, layer
    raise RuntimeError(
        f"unrecognised tensor name {name!r}: add a rule to "
        f"featherquant/roles.py rather than guessing a role"
    )


def classify_hf(name: str) -> tuple[Role, int | None]:
    """Role and layer index for a Hugging Face parameter name."""
    return _classify(name, _HF_RULES, _HF_LAYER)


def classify_gguf(name: str) -> tuple[Role, int | None]:
    """Role and layer index for a GGUF tensor name."""
    return _classify(name, _GGUF_RULES, _GGUF_LAYER)
