#!/usr/bin/env bash
# bench/harness/run_baseline.sh — run any command, emit a spec §6 run manifest.
#
# Usage: run_baseline.sh CMD_JSON RUN_ID METHOD MODEL_ID OUTPUT [STORAGE] [BUDGET_BYTES]
#   CMD_JSON  JSON array of argv, e.g. '["llama-quantize","in.gguf","out.gguf","Q4_K_M"]'
#   OUTPUT    artifact whose sha256 goes into the manifest
#   STORAGE   nvme|sata_ssd|hdd   (default nvme)
#   BUDGET_BYTES  declared ceiling; 0 = unconstrained baseline
# Manifests land in $FQ_MANIFEST_DIR (default bench/manifests).
#
# Assumption: this script imports featherquant.run_manifest through a Python
# heredoc, so it resolves the interpreter itself — repo_root/.venv/bin/python,
# found relative to this script's own path — instead of trusting $PATH. That
# means it works even under a restricted PATH (no venv activation needed) as
# long as the repo's .venv exists; run it from anywhere, no `cd` required.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY=$(command -v python3 || command -v python)
fi

CMD_JSON=$1; RUN_ID=$2; METHOD=$3; MODEL_ID=$4; OUTPUT=$5
STORAGE=${6:-nvme}; BUDGET=${7:-0}
DIR=${FQ_MANIFEST_DIR:-bench/manifests}
mkdir -p "$DIR"
TIMEFILE=$(mktemp)

# Cold page cache makes I/O numbers honest; needs root, so it is optional
# and recorded either way.
if [ "${FQ_DROP_CACHES:-0}" = "1" ]; then
  sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
fi

ARGVFILE=$(mktemp)
"$PY" - "$CMD_JSON" <<'PY' > "$ARGVFILE"
import json, shlex, sys
print(" ".join(shlex.quote(a) for a in json.loads(sys.argv[1])))
PY
ARGV=$(cat "$ARGVFILE"); rm -f "$ARGVFILE"

# -v gives "Maximum resident set size (kbytes)" and "Elapsed (wall clock) time".
/usr/bin/time -v sh -c "$ARGV" 2> "$TIMEFILE"
CODE=$?

"$PY" - "$TIMEFILE" "$RUN_ID" "$METHOD" "$MODEL_ID" "$OUTPUT" "$STORAGE" \
        "$BUDGET" "$CODE" "$DIR" <<'PY'
"""Parse GNU time output into a RunManifest and save it."""
import re
import sys

from featherquant.run_manifest import RunManifest, sha256_file

timefile, run_id, method, model_id, output, storage, budget, code, dirname = sys.argv[1:10]
text = open(timefile).read()
peak_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)[1])
wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", text)[1]
parts = [float(p) for p in wall.split(":")]
seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

m = RunManifest.new(run_id, {"id": model_id, "revision": "unknown",
                             "sha256": "unknown"},
                    method, int(budget), storage)
m.enforcement = "cgroup_v2_memory_max" if int(budget) else "none_unconstrained"
m.peak_observed_bytes = peak_kb * 1024
m.runtime_seconds = round(seconds, 3)
# 137 = SIGKILL, which under a cgroup ceiling means the OOM killer fired.
m.oom_killed = int(code) == 137
try:
    m.output_sha256 = sha256_file(output)
except RuntimeError:
    m.output_sha256 = None      # killed before the artifact existed
m.save(f"{dirname}/{run_id}.json")
print(f"wrote {dirname}/{run_id}.json (exit={code}, peak={peak_kb//1024} MiB)")
PY
rm -f "$TIMEFILE"
exit "$CODE"
