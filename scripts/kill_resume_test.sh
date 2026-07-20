#!/usr/bin/env bash
# Torture test: repeatedly SIGKILL featherquant at random moments, resume,
# until it completes; then verify against an uninterrupted reference run.
# Usage: scripts/kill_resume_test.sh SRC.gguf OUT.gguf [BUDGET]
set -uo pipefail
SRC=$1; OUT=$2; BUDGET=${3:-1GB}
PY=$(command -v python)
rm -f "$OUT" "$OUT.manifest.json" "$OUT.ref" "$OUT.ref.manifest.json"

# Uninterrupted reference run.
"$PY" -m featherquant.cli --model "$SRC" --output "$OUT.ref" \
  --max-ram "$BUDGET" >/dev/null || { echo "reference run failed"; exit 1; }

tries=0
while true; do
  tries=$((tries+1))
  "$PY" -m featherquant.cli --model "$SRC" --output "$OUT" \
    --max-ram "$BUDGET" --resume >/dev/null 2>&1 &
  pid=$!
  # Kill at a random point (0-8s). Must exceed resume's fixed startup +
  # verify overhead sometimes, or no attempt can make net progress.
  sleep "$((RANDOM % 8)).$((RANDOM % 9))"
  kill -9 "$pid" 2>/dev/null
  wait "$pid"; code=$?
  if [ "$code" -eq 0 ]; then break; fi     # finished before the kill landed
  if [ "$tries" -gt 200 ]; then echo "FAIL: no completion in 200 kills"; exit 1; fi
done

if cmp -s "$OUT" "$OUT.ref"; then
  echo "PASS: byte-identical after $tries interrupted runs"
else
  echo "FAIL: output differs from uninterrupted reference"
  exit 1
fi
