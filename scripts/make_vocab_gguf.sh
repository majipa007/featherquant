#!/usr/bin/env bash
# Produce the tiny metadata/tokenizer-only GGUF featherquant needs for
# safetensors input. Usage: scripts/make_vocab_gguf.sh HF_DIR OUT.gguf
set -euo pipefail
CONVERT_PY=${CONVERT_PY:-/home/sukuna/models/.convert-venv/bin/python}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-/home/sukuna/llama.cpp}
"$CONVERT_PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$1" --outfile "$2" --vocab-only
