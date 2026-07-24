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
VENV="$SCRIPT_DIR/.venv"
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
    local p
    if [ -n "$def" ]; then
        p="${C_BOLD}${prompt}${C_OFF} ${C_DIM}[${def}]${C_OFF}: "
    else
        p="${C_BOLD}${prompt}${C_OFF}: "
    fi
    if [ -t 0 ]; then
        # Interactive terminal: readline gives Tab path-completion, arrow
        # keys, and line editing. (read -p writes the prompt to stderr.)
        IFS= read -r -e -p "$p" REPLY || die "input closed"
    else
        # Piped stdin (tests, scripts): plain read, prompt on stderr.
        printf '%s' "$p" >&2
        IFS= read -r REPLY || die "input closed"
    fi
    # Trim trailing whitespace: readline's Tab-completion appends a space
    # after completed filenames, which would fail path validation.
    REPLY=${REPLY%"${REPLY##*[![:space:]]}"}
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
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null \
        || die "uv install failed (network?); install manually and re-run"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; open a new shell or add ~/.local/bin to PATH"
}

bootstrap() {
    [ -n "${FQ_SKIP_BOOTSTRAP:-}" ] && return 0
    if [ ! -x "$PY" ]; then
        ensure_uv
        say "creating virtualenv in $VENV"
        uv venv "$VENV" >/dev/null || die "could not create venv"
    fi
    # Re-sync only when pyproject changed since the last successful install.
    if [ ! -f "$STAMP" ] || [ "$SCRIPT_DIR/pyproject.toml" -nt "$STAMP" ]; then
        ensure_uv
        say "syncing dependencies (first run takes a minute)"
        uv pip install -p "$PY" -e "$SCRIPT_DIR" -q || die "dependency install failed"
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
    [ -x "$VENV/bin/featherquant" ] || die "venv looks broken — delete $VENV and re-run"
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
        model=${REPLY/#\~\//$HOME/}
        [ -e "$model" ] && break
        printf '%s\n' "  ${C_ERR}not found:${C_OFF} $model" >&2
    done

    # 2. Vocab GGUF, only for directory input. Most users won't have one:
    #    pressing Enter generates it on the spot (FQ_VOCAB_GEN is a test seam
    #    that swaps out the real generator script).
    local vocab_args=()
    if [ -d "$model" ]; then
        say "directory input needs a small metadata-only vocab GGUF (tokenizer + config)"
        local vocab default_vocab
        default_vocab="./$(basename "$model")-vocab.gguf"
        while :; do
            ask "Vocab GGUF (Enter = generate it for you)" "$default_vocab"
            vocab=${REPLY/#\~\//$HOME/}
            [ -f "$vocab" ] && break
            if [ -d "$vocab" ]; then
                printf '%s\n' "  ${C_ERR}is a directory, need a .gguf file:${C_OFF} $vocab" >&2
                continue
            fi
            ask "$(basename "$vocab") does not exist — generate it from the model now? (y/n)" "y"
            case $REPLY in
                [nN]*) continue ;;
            esac
            say "generating vocab GGUF (uses llama.cpp's convert script; a few seconds)"
            if "${FQ_VOCAB_GEN:-$SCRIPT_DIR/scripts/make_vocab_gguf.sh}" "$model" "$vocab" >&2; then
                break
            fi
            printf '%s\n' "  ${C_ERR}generation failed${C_OFF} — needs a llama.cpp checkout (set LLAMA_CPP_DIR) and its convert venv (set CONVERT_PY); or provide an existing vocab GGUF" >&2
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
    out=${REPLY/#\~\//$HOME/}
    if [ -d "$out" ]; then
        # A directory means "put it in there" — derive the filename.
        out="${out%/}/${stem}-${format}.gguf"
        say "output is a directory — writing to $out"
    fi

    # 5. RAM budget. Bare numbers mean GB (nobody budgets bytes);
    #    anything unparseable re-asks instead of exploding later.
    local ram
    while :; do
        ask "Max RAM budget (peak memory the run may use)" "2GB"
        ram=$REPLY
        if printf '%s' "$ram" | grep -qE '^[0-9]+([.][0-9]+)?$'; then
            ram="${ram}GB"
            say "interpreting as $ram"
            break
        elif printf '%s' "$ram" | grep -qiE '^[0-9]+([.][0-9]+)? *[KMGT]i?B?$'; then
            break
        fi
        printf '%s\n' "  ${C_ERR}not a size:${C_OFF} $ram (try 2GB or 512MB)" >&2
    done

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
