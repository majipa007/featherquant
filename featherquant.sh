#!/usr/bin/env bash
# featherquant.sh — one-file launcher for FeatherQuant.
#
#   ./featherquant.sh                 interactive wizard (asks model, format,
#                                     output, RAM budget, resume)
#   ./featherquant.sh [ARGS...]       bootstrap env, then pass ARGS straight
#                                     to the featherquant CLI
#
# First run bootstraps everything: installs uv if missing, creates .venv,
# syncs dependencies. Later runs are instant (stamp file).
#
# Test seams (used by tests/test_launcher.py):
#   FQ_DRY_RUN=1        print the final command to stdout instead of running
#   FQ_SKIP_BOOTSTRAP=1 skip environment setup
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
VENV=".venv"
PY="$VENV/bin/python"
STAMP="$VENV/.fq-deps-stamp"

# ---------- output helpers (stderr only; color only on a TTY) ----------
if [ -t 2 ]; then
    C_ACC=$'\033[36m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
    C_ACC=""; C_BOLD=""; C_DIM=""; C_ERR=""; C_OFF=""
fi
say() { printf '%s\n' "${C_ACC}>${C_OFF} $*" >&2; }
die() { printf '%s\n' "${C_ERR}error:${C_OFF} $*" >&2; exit 1; }

# ask PROMPT DEFAULT -> sets REPLY (empty input takes the default)
ask() {
    local prompt=$1 def=${2-}
    if [ -n "$def" ]; then
        printf '%s' "${C_BOLD}${prompt}${C_OFF} ${C_DIM}[${def}]${C_OFF}: " >&2
    else
        printf '%s' "${C_BOLD}${prompt}${C_OFF}: " >&2
    fi
    IFS= read -r REPLY || die "input closed"
    REPLY=${REPLY:-$def}
}

# ---------- environment bootstrap ----------
ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi
    say "uv not found — installing to ~/.local/bin"
    command -v curl >/dev/null 2>&1 || die "need curl to install uv (or install uv manually: https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
        || die "uv install failed (network?); install manually and re-run"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; open a new shell or add ~/.local/bin to PATH"
}

bootstrap() {
    [ -n "${FQ_SKIP_BOOTSTRAP:-}" ] && return 0
    if [ ! -x "$PY" ]; then
        ensure_uv
        say "creating virtualenv in $VENV"
        uv venv "$VENV" >/dev/null 2>&1 || die "could not create venv"
    fi
    # Re-sync only when pyproject changed since the last successful install.
    if [ ! -f "$STAMP" ] || [ "pyproject.toml" -nt "$STAMP" ]; then
        ensure_uv
        say "syncing dependencies (first run takes a minute)"
        uv pip install -p "$PY" -e . -q || die "dependency install failed"
        touch "$STAMP"
    fi
}

# ---------- command execution (dry-run seam) ----------
CMD=()
run_cmd() {
    if [ -n "${FQ_DRY_RUN:-}" ]; then
        printf '%q ' "${CMD[@]}"
        printf '\n'
        exit 0
    fi
    exec "${CMD[@]}"
}

# ---------- interactive wizard ----------
wizard() {
    printf '%s\n' "${C_BOLD}FeatherQuant${C_OFF} ${C_DIM}— quantize models larger than your RAM${C_OFF}" >&2
    printf '\n' >&2

    # 1. Model (file or HF safetensors directory)
    local model
    while :; do
        ask "Model (GGUF file or HF safetensors directory)"
        model=$REPLY
        [ -e "$model" ] && break
        printf '%s\n' "  ${C_ERR}not found:${C_OFF} $model" >&2
    done

    # 2. Vocab GGUF, only for directory input
    local vocab_args=()
    if [ -d "$model" ]; then
        say "directory input needs a metadata-only vocab GGUF"
        say "(create one with: scripts/make_vocab_gguf.sh \"$model\" vocab.gguf)"
        local vocab
        while :; do
            ask "Vocab GGUF path"
            vocab=$REPLY
            [ -f "$vocab" ] && break
            printf '%s\n' "  ${C_ERR}not found:${C_OFF} $vocab" >&2
        done
        vocab_args=(--vocab-gguf "$vocab")
    fi

    # 3. Format
    printf '%s\n' "${C_BOLD}Format${C_OFF}" >&2
    printf '%s\n' "  1) q8_0    ${C_DIM}8-bit, ~2x smaller, safest${C_OFF}" >&2
    printf '%s\n' "  2) q4_k_m  ${C_DIM}4-bit, ~4x smaller, needs llama.cpp libggml${C_OFF}" >&2
    ask "Choose" "1"
    local format
    case $REPLY in
        2) format="q4_k_m" ;;
        *) format="q8_0" ;;
    esac

    # 4. Output (default derived from the model name)
    local base stem out
    base=$(basename "$model")
    stem=${base%.gguf}
    ask "Output path" "./${stem}-${format}.gguf"
    out=$REPLY

    # 5. RAM budget
    ask "Max RAM budget (peak memory the run may use)" "2GB"
    local ram=$REPLY

    # 6. Resume, only when an interrupted run left a manifest
    local resume_args=()
    if [ -f "${out}.manifest.json" ]; then
        ask "Found an interrupted run for this output — resume it? (y/n)" "y"
        case $REPLY in
            [nN]*) : ;;
            *) resume_args=(--resume) ;;
        esac
    fi

    CMD=("$VENV/bin/featherquant" --model "$model" --output "$out"
         --format "$format" --max-ram "$ram")
    [ ${#vocab_args[@]} -gt 0 ] && CMD+=("${vocab_args[@]}")
    [ ${#resume_args[@]} -gt 0 ] && CMD+=("${resume_args[@]}")

    printf '\n' >&2
    say "running: ${CMD[*]}"
    run_cmd
}

# ---------- main ----------
bootstrap
if [ "$#" -gt 0 ]; then
    CMD=("$VENV/bin/featherquant" "$@")
    run_cmd
fi
wizard
