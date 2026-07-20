#!/usr/bin/env bash
# Run featherquant inside an OS-enforced memory ceiling (cgroup v2 via
# systemd-run --user). Proves the "externally enforced bound" claim:
# if RSS ever exceeds MemoryMax the kernel OOM-kills the job and we exit
# non-zero. Swap is disabled inside the scope so the ceiling is honest.
#
# Usage: scripts/memlimit_run.sh MODEL OUT [MEMORY_MAX] [BUDGET]
#   MEMORY_MAX  systemd size (default 1G) — the hard external ceiling
#   BUDGET      featherquant --max-ram   (default same as MEMORY_MAX)
set -uo pipefail
MODEL=$1; OUT=$2; LIMIT=${3:-1G}; BUDGET=${4:-$LIMIT}
PY=$(command -v python)

# NOTE: --wait is invalid with --scope on this systemd; a scope unit already
# runs the command in the foreground and propagates its exit code.
if systemd-run --user --scope --collect --same-dir \
    -p MemoryMax="$LIMIT" -p MemorySwapMax=0 \
    "$PY" -m featherquant.cli --model "$MODEL" --output "$OUT" \
    --format q8_0 --max-ram "$BUDGET" --report "${OUT%.gguf}.report.json"; then
  echo "PASS: completed inside $LIMIT external ceiling"
else
  echo "FAIL: killed or errored under $LIMIT ceiling (OOM if exit=137)"
  exit 1
fi
