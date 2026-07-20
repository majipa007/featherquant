"""Type-map rules must reproduce the reference llama-quantize dump exactly."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from gguf import GGMLQuantizationType, GGUFReader

from featherquant.formats import FORMATS

DUMP_PATH = Path(__file__).parent / "data" / "qwen3-0.6b-q4_k_m-types.json"
SRC_GGUF = "/home/sukuna/models/qwen3-0.6b-bf16.gguf"


@pytest.mark.skipif(not os.path.exists(SRC_GGUF), reason="real model not present")
def test_q4_k_m_matches_reference_dump():
    dump = json.loads(DUMP_PATH.read_text())
    r = GGUFReader(SRC_GGUF)
    n_layers = int(r.fields["qwen3.block_count"].contents())
    spec = FORMATS["q4_k_m"]
    bad = {t.name: (spec.tensor_type(t, n_layers).name, dump[t.name])
           for t in r.tensors
           if spec.tensor_type(t, n_layers).name != dump[t.name]}
    assert not bad, f"rule mismatches (got, want): {bad}"


def test_q8_0_rule_unchanged():
    t = SimpleNamespace(name="blk.0.attn_q.weight", shape=[64, 10],
                        tensor_type=GGMLQuantizationType.F16)
    assert FORMATS["q8_0"].tensor_type(t, 28) == GGMLQuantizationType.Q8_0
    norm = SimpleNamespace(name="blk.0.attn_norm.weight", shape=[64],
                           tensor_type=GGMLQuantizationType.F32)
    assert FORMATS["q8_0"].tensor_type(norm, 28) == GGMLQuantizationType.F32


def test_q4_k_m_synthetic_rules():
    # 28-layer model: layers 0,1,2 (first n/8), 24..27 (last n/8) and every
    # 3rd in between (5,8,...,23) get Q6_K for attn_v/ffn_down.
    spec = FORMATS["q4_k_m"]

    def t(name):
        return SimpleNamespace(name=name, shape=[256, 8],
                               tensor_type=GGMLQuantizationType.BF16)

    assert spec.tensor_type(t("blk.3.attn_v.weight"), 28) == GGMLQuantizationType.Q4_K
    assert spec.tensor_type(t("blk.5.attn_v.weight"), 28) == GGMLQuantizationType.Q6_K
    assert spec.tensor_type(t("blk.27.ffn_down.weight"), 28) == GGMLQuantizationType.Q6_K
    assert spec.tensor_type(t("output.weight"), 28) == GGMLQuantizationType.Q6_K
    assert spec.tensor_type(t("token_embd.weight"), 28) == GGMLQuantizationType.Q4_K
    # rows not divisible by 256: keep source type
    narrow = SimpleNamespace(name="blk.0.attn_q.weight", shape=[64, 8],
                             tensor_type=GGMLQuantizationType.F16)
    assert spec.tensor_type(narrow, 28) == GGMLQuantizationType.F16
