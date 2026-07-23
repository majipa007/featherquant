# One-File Launcher Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single `featherquant.sh` at the repo root that bootstraps everything (uv → venv → deps) and then either passes arguments straight to the CLI or walks the user through a friendly interactive config wizard — making the repo feel like a ready-to-run terminal tool.

**Architecture:** One POSIX-ish bash script (bash 4+ features allowed; WSL/Linux target) with three layers: `bootstrap()` (idempotent env setup with a dep-stamp so repeat runs are instant), argument passthrough (any args → `.venv/bin/featherquant "$@"`), and `wizard()` (no args → prompts for model, format, output, budget, resume; composes the command). Two test seams make it verifiable: `FQ_DRY_RUN=1` prints the composed command to stdout instead of exec'ing it, and `FQ_SKIP_BOOTSTRAP=1` skips env setup — so pytest can drive the wizard through stdin and assert on the exact command line.

**Tech Stack:** bash (no whiptail/dialog — plain `read` prompts with ANSI color when stderr is a TTY), uv (installed from astral.sh if missing), existing pytest suite for subprocess-level tests.

## Global Constraints

- Single new file `featherquant.sh` at repo root, executable; tests in `tests/test_launcher.py`. No new Python dependencies.
- Script must be run-from-anywhere: it `cd`s to its own directory first.
- All human-facing output (prompts, status) goes to **stderr**; stdout is reserved for the dry-run command echo (mirrors the CLI's stderr-UI/stdout-JSON split).
- Colors only when stderr is a TTY (`[ -t 2 ]`); zero escape codes when piped.
- Bootstrap idempotence: second run with a synced venv must not reinstall anything (stamp file `.venv/.fq-deps-stamp`, invalidated when `pyproject.toml` is newer).
- uv install path: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `export PATH="$HOME/.local/bin:$PATH"`; fail with a clear message if curl/network unavailable.
- Wizard validates: model path exists; directory input requires a vocab GGUF (existing file); format menu limited to `q8_0`/`q4_k_m`; detects `<output>.manifest.json` and offers `--resume`.
- Test seams: `FQ_DRY_RUN=1` → print command via `printf '%q '` to stdout, exit 0; `FQ_SKIP_BOOTSTRAP=1` → skip `bootstrap()`. Both must be documented in the script header comment.
- Gates stay green: `.venv/bin/pytest`, `.venv/bin/ruff check featherquant tests`, `.venv/bin/mypy featherquant` (script is bash — ruff/mypy untouched by it; pytest gains the launcher tests). Run `shellcheck featherquant.sh` if shellcheck is installed; fix findings; if not installed, note that in the commit message body.
- Commits authored by majipa007, no co-author line. TDD: tests first for every behavior.

## File Structure

```
featherquant.sh          — bootstrap + passthrough + wizard (new, the deliverable)
tests/test_launcher.py   — subprocess tests driving both modes (new)
README.md                — Quick start section update (Task 2)
```

---

### Task 1: Bootstrap + passthrough with test seams

**Files:**
- Create: `featherquant.sh`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Consumes: existing `.venv/bin/featherquant` console script; `pyproject.toml` at repo root.
- Produces (Task 2 extends the same file): functions `say`, `die`, `ask`, `ensure_uv`, `bootstrap`, `run_cmd`, array variable `CMD`; env seams `FQ_DRY_RUN`, `FQ_SKIP_BOOTSTRAP`. Passthrough behavior: `./featherquant.sh <args...>` runs `.venv/bin/featherquant <args...>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_launcher.py`:

```python
"""Subprocess tests for featherquant.sh (bootstrap seams + passthrough)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = str(ROOT / "featherquant.sh")


def run_sh(args=(), stdin="", extra_env=None, timeout=60):
    """Run the launcher with the dry-run seam on; return CompletedProcess."""
    env = {**os.environ, "FQ_DRY_RUN": "1", "FQ_SKIP_BOOTSTRAP": "1",
           **(extra_env or {})}
    return subprocess.run(["bash", SH, *args], input=stdin, text=True,
                          capture_output=True, env=env, cwd=ROOT,
                          timeout=timeout)


def test_passthrough_composes_cli_command():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert ".venv/bin/featherquant" in out
    assert "--model m.gguf" in out and "--max-ram 1GB" in out


def test_stdout_carries_only_the_command():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert r.returncode == 0
    assert len(r.stdout.strip().splitlines()) == 1  # prompts/status live on stderr


def test_no_escape_codes_when_piped():
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"])
    assert "\x1b[" not in r.stdout + r.stderr


def test_bootstrap_fast_path_skips_reinstall():
    # Real repo already has a synced venv + stamp: a run WITHOUT the skip
    # seam must not install anything (idempotence) and still succeed.
    r = run_sh(["--model", "m.gguf", "--output", "o.gguf", "--max-ram", "1GB"],
               extra_env={"FQ_SKIP_BOOTSTRAP": ""})
    assert r.returncode == 0, r.stderr
    assert "installing" not in r.stderr.lower()
    assert "syncing" not in r.stderr.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_launcher.py -v`
Expected: 4 FAIL — bash exits 127-style failure because `featherquant.sh` does not exist (`r.returncode != 0`).

- [ ] **Step 3: Write `featherquant.sh` (bootstrap + passthrough; wizard arrives in Task 2)**

```bash
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

# ---------- main ----------
bootstrap
if [ "$#" -gt 0 ]; then
    CMD=("$VENV/bin/featherquant" "$@")
    run_cmd
fi
die "wizard not implemented yet"   # replaced in Task 2
```

Run: `chmod +x featherquant.sh`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_launcher.py -v`
Expected: 4 passed. If `test_bootstrap_fast_path_skips_reinstall` fails because the stamp doesn't exist yet, run `touch .venv/.fq-deps-stamp` once — the stamp is created by the first real bootstrap, which CI/dev machines hit naturally; the test asserts the fast path, not the first install.

- [ ] **Step 5: Shellcheck (if present) + full suite + commit**

Run: `command -v shellcheck && shellcheck featherquant.sh || echo "shellcheck not installed"` — fix any findings.
Run: `.venv/bin/pytest` — all green.

```bash
git add featherquant.sh tests/test_launcher.py
git commit -m "feat: one-file launcher with uv bootstrap and CLI passthrough"
```

---

### Task 2: Interactive wizard + README quick start

**Files:**
- Modify: `featherquant.sh` (replace the final `die "wizard not implemented yet"` line with `wizard`, add the `wizard()` function above `# ---------- main ----------`)
- Modify: `README.md` (Install section)
- Test: `tests/test_launcher.py` (append)

**Interfaces:**
- Consumes: Task 1's `ask`, `say`, `die`, `CMD`, `run_cmd`, seams.
- Produces: `wizard()` — prompts in order: model path (re-asks until it exists), vocab GGUF (only when model is a directory; re-asks until it exists), format menu (1=q8_0 default, 2=q4_k_m), output path (default `./<model-stem>-<format>.gguf`), max RAM (default `2GB`), resume y/n (only when `<output>.manifest.json` exists, default y). Composes `CMD=(.venv/bin/featherquant --model … --output … --format … --max-ram … [--vocab-gguf …] [--resume])` and calls `run_cmd`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_launcher.py`)**

```python
def _touch(p: Path, data: bytes = b"GGUF") -> Path:
    p.write_bytes(data)
    return p


def test_wizard_composes_full_command(tmp_path):
    model = _touch(tmp_path / "tiny.gguf")
    # inputs: model, format=1 (q8_0), output=default, ram=default
    stdin = f"{model}\n1\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--model" in out and "tiny.gguf" in out
    assert "--format q8_0" in out
    assert "--max-ram 2GB" in out
    assert "tiny-q8_0.gguf" in out            # derived default output name


def test_wizard_reasks_on_missing_model(tmp_path):
    model = _touch(tmp_path / "real.gguf")
    stdin = f"/nope/missing.gguf\n{model}\n1\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "not found" in r.stderr
    assert "real.gguf" in r.stdout


def test_wizard_directory_model_requires_vocab(tmp_path):
    d = tmp_path / "hfmodel"
    d.mkdir()
    vocab = _touch(tmp_path / "vocab.gguf")
    # inputs: model dir, vocab, format=2 (q4_k_m), output default, ram 1GB
    stdin = f"{d}\n{vocab}\n2\n\n1GB\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--vocab-gguf" in out and "vocab.gguf" in out
    assert "--format q4_k_m" in out and "--max-ram 1GB" in out


def test_wizard_offers_resume_when_manifest_exists(tmp_path):
    model = _touch(tmp_path / "m.gguf")
    out_path = tmp_path / "out.gguf"
    _touch(out_path.with_suffix(".gguf.manifest.json"), b"{}")
    # inputs: model, format=1, output=explicit, ram default, resume default(y)
    stdin = f"{model}\n1\n{out_path}\n\n\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "--resume" in r.stdout


def test_wizard_can_decline_resume(tmp_path):
    model = _touch(tmp_path / "m.gguf")
    out_path = tmp_path / "out.gguf"
    _touch(out_path.with_suffix(".gguf.manifest.json"), b"{}")
    stdin = f"{model}\n1\n{out_path}\n\nn\n"
    r = run_sh(stdin=stdin)
    assert r.returncode == 0, r.stderr
    assert "--resume" not in r.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_launcher.py -v`
Expected: the 5 new tests FAIL with `wizard not implemented yet` in stderr (returncode 1); Task 1's 4 still pass.

- [ ] **Step 3: Implement the wizard**

In `featherquant.sh`, insert above the `# ---------- main ----------` section:

```bash
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
```

And replace the last line of the script:

```bash
die "wizard not implemented yet"   # replaced in Task 2
```

with:

```bash
wizard
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_launcher.py -v`
Expected: 9 passed. Then the full suite: `.venv/bin/pytest` — all green.

- [ ] **Step 5: Manual smoke check**

```bash
printf '/home/sukuna/models/qwen3-0.6b-bf16.gguf\n1\n/tmp/claude-1000/wiz.gguf\n1GB\n' | FQ_DRY_RUN=1 ./featherquant.sh
```
Expected: single stdout line ending with `--model /home/sukuna/models/qwen3-0.6b-bf16.gguf --output /tmp/claude-1000/wiz.gguf --format q8_0 --max-ram 1GB`. Optionally run it for real (no FQ_DRY_RUN) and watch the dashboard.

- [ ] **Step 6: README quick start + shellcheck + commit**

In `README.md`, replace the Install section body:

```markdown
## Quick start

​```bash
git clone git@github.com:majipa007/featherquant.git && cd featherquant
./featherquant.sh          # first run bootstraps uv + venv + deps, then asks
                           # for your model, format, output, and RAM budget
​```

Or pass CLI flags straight through: `./featherquant.sh --model m.gguf
--output o.gguf --max-ram 1GB`. Manual setup still works:

​```bash
uv venv .venv
uv pip install -p .venv/bin/python -e '.[dev]'
​```
```

(Strip the zero-width markers around the inner fences when transcribing.)

Run: `command -v shellcheck && shellcheck featherquant.sh || echo "shellcheck not installed"` — fix findings.

```bash
git add featherquant.sh tests/test_launcher.py README.md
git commit -m "feat: interactive config wizard in the launcher"
```

---

## Self-review notes

- Spec coverage: single sh file ✓; venv check ✓; uv check + install ✓; uv installs deps into venv ✓; runs the python code ✓ (exec CLI); "good config option — where the model is and all that" ✓ (wizard: model/vocab/format/output/RAM/resume with validation and derived defaults).
- Type consistency: `ask` sets `REPLY`; `CMD` array + `run_cmd` shared between passthrough and wizard; seams named identically in both tasks and in tests.
- Deliberate simplifications: no whiptail/fzf menus (plain numbered prompt — zero deps); wizard doesn't offer `--report`/`--ui`/`--ggml-lib` (CLI flags exist for power users; passthrough covers them).
