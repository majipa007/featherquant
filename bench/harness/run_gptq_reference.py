#!/usr/bin/env python3
"""bench/harness/run_gptq_reference.py — unconstrained reference GPTQ.

Runs GPTQ on the GPU with no memory ceiling, records perplexity and
per-linear reconstruction error. This is the quality target M4's calibrator
must match within noise (Task 17 compares against it directly).

Requires (throwaway venv, NOT part of pyproject.toml): uv pip install
gptqmodel torch datasets

Usage: run_gptq_reference.py HF_MODEL_DIR OUT_DIR RUN_ID

Reads calibration texts from bench/data/wiki.test.raw, which MUST be the
corpus pinned by bench/harness/fetch_calibration_corpus.sh (sha256
173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08) — that pin
is what makes this reference comparable to every later FeatherQuant number.
Run fetch_calibration_corpus.sh first if that file is missing. The corpus is
chunked into CALIBRATION_SAMPLES pieces of approximately
CALIBRATION_SEQLEN_TOKENS tokens each (see _chunk_calibration_corpus) —
matching Task 16's calibrator shape, spec §3.3 — and the run fails loudly if
the corpus is too short to produce that many full chunks.

Produces two artifacts:
  bench/manifests/<RUN_ID>.json               spec §6 run manifest. Its
      output_sha256 is a whole-directory digest (see _hash_directory)
      because gptqmodel's model.save() writes a directory, not one file.
  bench/manifests/<RUN_ID>_layer_errors.json  per-linear reconstruction
      error, "<layer_index>.<role>" -> mean-squared-error float, for the
      seven linear roles (attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up,
      ffn_down). Computed by snapshotting each nn.Linear weight before
      model.quantize() overwrites it in place, then looking each snapshotted
      module up by name in the quantized model and comparing its effective
      (dequantized) weight. Any module that cannot be compared (not found
      by name, no accessible weight, shape mismatch) is recorded as that
      module's own "unavailable: <reason>" string rather than dropped or
      failing the whole file — a partial artifact naming exactly which
      modules could not be read is more useful than an all-or-nothing
      marker. The whole-file {"unavailable": "<reason>"} form is reserved
      for the case where module-by-module comparison could not even be
      attempted (the snapshot itself is empty, or the quantized model
      isn't walkable by name at all) — never a fabricated or silently
      partial map.

This script is NOT part of the shipped featherquant package: it lives in
bench/harness/ and produces numbers, not library code. It is never imported
by featherquant/.
"""
import re
import sys

# Argument validation happens here, at module scope, before any of this
# script's heavy (torch/gptqmodel) imports — those only happen inside
# main(), below. That way a bare invocation (e.g. from a CI/test venv that
# has neither dependency installed) fails fast with a usage message instead
# of an ImportError or a raw traceback.
_USAGE = "usage: run_gptq_reference.py HF_MODEL_DIR OUT_DIR RUN_ID"

# spec §3.3 calibration shape: 128 samples x 512 tokens (the same shape
# Task 16's calibrator uses, tokenizing this same pinned corpus). This
# script never loads a tokenizer of its own (gptqmodel tokenizes internally
# from raw text), so sample length is approximated by character count using
# a documented, roughly-4-characters-per-token ratio for this corpus. That
# is an approximation, not a real token count — it exists only to produce
# comparably-sized calibration chunks without pulling in a tokenizer here.
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQLEN_TOKENS = 512
CALIBRATION_CHARS_PER_TOKEN = 4

# HF/Qwen3-style dotted module name suffix -> spec §4.1 role. This is a
# deliberately small, local lookup for this one reference script — it is
# NOT the general model-family indexer (that is Milestone M1's job); it
# only needs to key the per-linear error map consistently with the
# project's role names for the one model family this baseline runs against.
_ROLE_BY_SUFFIX = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_o",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}

_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def _role_from_module_name(name: str) -> str | None:
    """Map a dotted module name (e.g. 'model.layers.3.self_attn.q_proj') to
    one of the seven linear roles this file records, or None if it is not
    one of those (e.g. embeddings, norms, lm_head)."""
    for suffix, role in _ROLE_BY_SUFFIX.items():
        if name.endswith(suffix):
            return role
    return None


def _layer_index_from_module_name(name: str) -> str | None:
    """Extract the decoder-layer index from a dotted module name, e.g.
    'model.layers.12.self_attn.q_proj' -> '12'. None if no layer index is
    present in the name (e.g. lm_head)."""
    m = _LAYER_INDEX_RE.search(name)
    return m.group(1) if m else None


def _chunk_calibration_corpus(
    text: str, samples: int, seqlen_tokens: int, chars_per_token: int
) -> list[str]:
    """Split `text` into contiguous, non-overlapping chunks of approximately
    `seqlen_tokens` tokens each (approximated as `chars_per_token`
    characters per token — see the CALIBRATION_* constants' comment above),
    and return the first `samples` full chunks.

    wikitext-2-raw's wiki.test.raw has essentially no blank-line/paragraph
    structure (a naive "\\n\\n" split returns the whole file as one string),
    so this chunks by raw character count instead of trying to find natural
    boundaries.

    A short trailing chunk (the corpus length isn't an exact multiple of the
    chunk size) is dropped rather than kept undersized. If fewer than
    `samples` full chunks are available, this raises loudly — a run must
    never silently calibrate on less data than it declared.
    """
    chunk_chars = seqlen_tokens * chars_per_token
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    chunks = [c for c in chunks if len(c) == chunk_chars]  # drop short tail

    print(f"calibration corpus: requested {samples} chunks of "
          f"~{seqlen_tokens} tokens (~{chunk_chars} chars) each; obtained "
          f"{len(chunks)} full chunks from {len(text)} chars total",
          file=sys.stderr)

    if len(chunks) < samples:
        raise RuntimeError(
            f"calibration corpus too short: requested {samples} chunks of "
            f"~{seqlen_tokens} tokens each, only {len(chunks)} full chunks "
            f"available ({len(text)} chars total). Refusing to calibrate on "
            "less data than declared."
        )
    return chunks[:samples]


def _snapshot_linear_weights(model: object) -> dict[str, object]:
    """Best-effort snapshot of every named nn.Linear weight, keyed by dotted
    module name, taken BEFORE model.quantize() overwrites them in place.

    Different gptqmodel versions wrap the underlying HF torch.nn.Module
    differently (some directly, some behind a `.model` attribute) — try the
    common case, and return an empty dict (never raise) if neither shape is
    walkable. An empty dict is the caller's signal to mark the layer-errors
    file "unavailable" rather than compare against nothing.
    """
    import torch.nn as nn

    torch_model = getattr(model, "model", None)
    if torch_model is None or not hasattr(torch_model, "named_modules"):
        torch_model = model
    if not hasattr(torch_model, "named_modules"):
        return {}
    snapshot = {}
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Linear):
            snapshot[name] = module.weight.detach().clone().float().cpu()
    return snapshot


def _effective_weight(module: object) -> object | None:
    """Best-effort extraction of a module's effective (dequantized) weight
    as a plain float CPU tensor, comparable to the pre-quantization
    snapshot. gptqmodel commonly replaces nn.Linear with a packed quantized
    layer class after quantize(), so `.weight` alone is not guaranteed to
    still be a full-precision tensor.

    Tries, in order: a still-plain nn.Linear's `.weight`; a couple of
    dequantize-style accessor names used across gptqmodel/AutoGPTQ
    backends; finally a raw `.weight` attribute if one of those methods
    didn't exist. None of these are guaranteed to exist for a given
    installed version — this returns None (never raises, never guesses) if
    none apply, so the caller records that specific module as unavailable
    instead of fabricating a number.
    """
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return module.weight.detach().float().cpu()

    for attr in ("dequantize_weight", "dequantize"):
        fn = getattr(module, attr, None)
        if callable(fn):
            try:
                return fn().detach().float().cpu()
            except (RuntimeError, TypeError, AttributeError):
                continue  # this accessor exists but didn't work; try the next

    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "detach"):
        return weight.detach().float().cpu()
    return None


def _compute_layer_errors(
    model: object, originals: dict[str, object]
) -> dict[str, float | str]:
    """Per-linear mean-squared reconstruction error, "<layer>.<role>" ->
    MSE, driven from the pre-quantization snapshot (not a post-quantization
    nn.Linear scan — gptqmodel commonly replaces Linear with a packed
    quantized class, so scanning for nn.Linear after quantization would find
    nothing and silently produce an all-"unavailable" result on every
    successful run).

    For each snapshotted module name that maps to one of the seven roles,
    this looks the module up by name in the quantized model and records
    either its MSE against the snapshot, or an "unavailable: <reason>"
    string for that module specifically — never a fabricated number, and
    never dropping the entry silently. Only when module-by-module
    comparison cannot even be attempted (no snapshot, or the quantized
    model isn't walkable by name) does this return the whole-file
    {"unavailable": "<reason>"} form instead.
    """
    import torch

    if not originals:
        return {"unavailable": "could not snapshot original linear weights "
                                "before quantization for this gptqmodel "
                                "version (model was not a walkable "
                                "torch.nn.Module via `.model` or itself)"}

    torch_model = getattr(model, "model", None)
    if torch_model is None or not hasattr(torch_model, "named_modules"):
        torch_model = model
    if not hasattr(torch_model, "named_modules"):
        return {"unavailable": "quantized model is not a walkable "
                                "torch.nn.Module after model.quantize() "
                                "(cannot look up any snapshotted module by "
                                "name)"}

    quantized_by_name = dict(torch_model.named_modules())

    errors: dict[str, float | str] = {}
    for name, orig_weight in originals.items():
        role = _role_from_module_name(name)
        layer_idx = _layer_index_from_module_name(name)
        if role is None or layer_idx is None:
            continue  # not one of the seven roles this file records
        key = f"{layer_idx}.{role}"

        module = quantized_by_name.get(name)
        if module is None:
            errors[key] = "unavailable: module not found by this name in the quantized model"
            continue

        quant_weight = _effective_weight(module)
        if quant_weight is None:
            errors[key] = (f"unavailable: module class {type(module).__name__} exposes no "
                            "usable weight or dequantize accessor")
            continue

        if quant_weight.shape != orig_weight.shape:
            errors[key] = (f"unavailable: shape mismatch (original {tuple(orig_weight.shape)} "
                            f"vs quantized {tuple(quant_weight.shape)})")
            continue

        errors[key] = torch.mean((quant_weight - orig_weight) ** 2).item()

    if not errors:
        return {"unavailable": "no snapshotted module name mapped to a known "
                                "linear role (attn_q/k/v/o, ffn_gate/up/down) "
                                "-- unexpected naming convention for this model"}
    return errors


def _hash_directory(dir_path: str) -> str:
    """Deterministic sha256 digest of an entire output directory.

    gptqmodel's model.save() writes a directory (sharded safetensors plus
    config files), not a single file, so there is no single-file hash to
    record as RunManifest.output_sha256 the way Baselines 1-3's single
    .gguf files get one. Composed by walking every file under dir_path,
    sorting by path relative to dir_path (so the digest never depends on
    filesystem iteration order), and folding
    "<relative_path>\\n<sha256_of_file>\\n" for each into one running
    sha256. Reuses featherquant.run_manifest.sha256_file for the per-file
    hash rather than writing a second chunked hasher.

    A human can reproduce this exact digest with, from dir_path's parent:
        (cd dir_path && find . -type f | sed 's|^\\./||' | sort | \\
         while read -r f; do printf '%s\\n' "$f"; sha256sum "$f" | cut -d' ' -f1; \\
         done) | sha256sum
    """
    import hashlib
    import os

    from featherquant.run_manifest import sha256_file

    root = os.path.abspath(dir_path)
    rel_paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel_paths.append(os.path.relpath(full, root))
    rel_paths.sort()

    digest = hashlib.sha256()
    for rel in rel_paths:
        digest.update(f"{rel}\n".encode())
        digest.update(f"{sha256_file(os.path.join(root, rel))}\n".encode())
    return digest.hexdigest()


def _save_json(path: str, data: dict) -> None:
    """Write JSON atomically (tmp + replace), same pattern RunManifest.save
    uses, so a crash mid-write can never leave Task 17 reading a truncated
    layer-errors file."""
    import json
    import os

    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        raise RuntimeError(f"cannot save {path}: {exc}") from exc


def main() -> None:
    if len(sys.argv) != 4:
        print(f"{_USAGE}\nexpected 3 arguments, got {len(sys.argv) - 1}",
              file=sys.stderr)
        sys.exit(2)
    model_dir, out_dir, run_id = sys.argv[1:4]

    # Heavy, GPU-only imports deferred to here — after argv validation —
    # so a bare/invalid invocation never needs torch or gptqmodel installed.
    import json
    import time

    import torch
    from gptqmodel import GPTQModel, QuantizeConfig

    from featherquant.run_manifest import RunManifest

    corpus_path = "bench/data/wiki.test.raw"
    try:
        with open(corpus_path) as f:
            corpus_text = f.read()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read calibration corpus {corpus_path}: {exc}. Run "
            "bench/harness/fetch_calibration_corpus.sh first."
        ) from exc
    texts = _chunk_calibration_corpus(
        corpus_text, CALIBRATION_SAMPLES, CALIBRATION_SEQLEN_TOKENS,
        CALIBRATION_CHARS_PER_TOKEN)

    cfg = QuantizeConfig(bits=4, group_size=128, damp_percent=0.01,
                          desc_act=False, sym=True)
    t0 = time.monotonic()
    try:
        model = GPTQModel.load(model_dir, cfg)
    except Exception as exc:
        raise RuntimeError(f"reference GPTQ failed: {exc}") from exc

    # Deliberately outside the try/except above: this is our own harness
    # instrumentation, not part of the GPTQ run itself. A bug here must
    # never be reported as "reference GPTQ failed" when GPTQ never failed.
    originals = _snapshot_linear_weights(model)

    try:
        model.quantize(texts)
        model.save(out_dir)
    except Exception as exc:
        raise RuntimeError(f"reference GPTQ failed: {exc}") from exc
    runtime = time.monotonic() - t0

    layer_errors = _compute_layer_errors(model, originals)
    layer_errors_path = f"bench/manifests/{run_id}_layer_errors.json"
    _save_json(layer_errors_path, layer_errors)

    m = RunManifest.new(run_id, {"id": model_dir, "revision": "local",
                                  "sha256": "unknown"},
                         "gptq_reference_4bit_g128", 0, "nvme")
    m.enforcement = "none_unconstrained"
    m.runtime_seconds = round(runtime, 3)
    m.peak_observed_bytes = torch.cuda.max_memory_allocated()
    m.output_sha256 = _hash_directory(out_dir)
    m.quality = {"ppl": None,
                 "ppl_dataset": "wikitext-2-raw/wiki.test.raw c=512 tokenizer=qwen3",
                 "tasks": {}}
    m.save(f"bench/manifests/{run_id}.json")
    print(json.dumps({"runtime_s": runtime,
                       "layer_errors_path": layer_errors_path}, indent=2))


if __name__ == "__main__":
    main()
