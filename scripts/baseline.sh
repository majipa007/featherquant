#!/usr/bin/env bash
# Phase 0 baseline: pinned llama-quantize Q8_0 run with peak-RSS capture.
# Usage: LLAMA_CPP_DIR=~/llama.cpp scripts/baseline.sh SRC.gguf REF_OUT.gguf
set -euo pipefail
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout with built tools}"
SRC=$1; REF_OUT=$2
# Record the exact llama.cpp revision so the baseline is reproducible.
git -C "$LLAMA_CPP_DIR" rev-parse HEAD | tee baseline_commit.txt
# /usr/bin/time -v reports "Maximum resident set size" = conventional peak RSS.
# Binaries dir is overridable: some checkouts build into build-cpu/ etc.
BIN="${LLAMA_BIN:-$LLAMA_CPP_DIR/build/bin}"
/usr/bin/time -v "$BIN/llama-quantize" "$SRC" "$REF_OUT" Q8_0 \
  2> baseline_time.txt
grep -E 'Maximum resident|Elapsed' baseline_time.txt
sha256sum "$REF_OUT"
