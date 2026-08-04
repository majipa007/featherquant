#!/usr/bin/env bash
# bench/harness/run_under_ceiling.sh — run a command inside a cgroup v2 ceiling.
#
# Usage: run_under_ceiling.sh LIMIT CMD...
# Swap is disabled inside the scope so the ceiling is honest. Exit 137 means
# the kernel OOM-killed the job — that is a measurement, not a bug.
set -uo pipefail
LIMIT=$1; shift
exec systemd-run --user --scope --collect --same-dir \
  -p MemoryMax="$LIMIT" -p MemorySwapMax=0 "$@"
