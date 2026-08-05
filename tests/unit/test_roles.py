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
