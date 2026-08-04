#!/usr/bin/env bash
# bench/harness/fetch_calibration_corpus.sh [DEST]
#
# Fetches and pins the wikitext-2-raw calibration/perplexity corpus that
# every baseline manifest's quality.ppl_dataset (spec §6) is measured
# against. The first run downloads the upstream zip, extracts wiki.test.raw
# from it, and pins its sha256 next to it; every later run re-verifies the
# on-disk file against that pin and fails loudly on any mismatch — a
# silently-changed corpus would invalidate every ppl number already measured
# against it, so verification is the point of this script, not the download.
#
# Usage: fetch_calibration_corpus.sh [DEST]
#   DEST  path to the corpus file (default: bench/data/wiki.test.raw)
#         the pin lives alongside it at "$DEST.sha256"
set -uo pipefail

# Same archive llama.cpp's own scripts/get-wikitext-2.sh uses. The plain
# wikitext-2-raw/wiki.test.raw path upstream once served directly 404s as of
# 04/08/2026 — ggml-org/ci now only ships the zip.
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
ZIP_MEMBER="wikitext-2-raw/wiki.test.raw"
DEST=${1:-bench/data/wiki.test.raw}
SHA_FILE="${DEST}.sha256"
DEST_DIR=$(dirname "$DEST")

if ! mkdir -p "$DEST_DIR"; then
  echo "ERROR: cannot create directory \"$DEST_DIR\"" >&2
  exit 1
fi

TMP_ZIP=""
TMP_EXTRACT=""
cleanup() {
  [ -n "$TMP_ZIP" ] && rm -f "$TMP_ZIP"
  [ -n "$TMP_EXTRACT" ] && rm -rf "$TMP_EXTRACT"
}
trap cleanup EXIT

if [ ! -f "$DEST" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found; install it (e.g. apt-get install curl) to fetch \"$URL\"" >&2
    exit 1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip not found; install it (e.g. apt-get install unzip) to extract \"$ZIP_MEMBER\"" >&2
    exit 1
  fi

  echo "corpus not found at \"$DEST\"; downloading $URL"
  TMP_ZIP=$(mktemp "${DEST}.zip.XXXXXX")
  if ! curl -fL -o "$TMP_ZIP" "$URL"; then
    echo "ERROR: failed to download \"$URL\"" >&2
    exit 1
  fi

  TMP_EXTRACT=$(mktemp -d "${DEST_DIR}/extract.XXXXXX")
  if ! unzip -q -o "$TMP_ZIP" "$ZIP_MEMBER" -d "$TMP_EXTRACT"; then
    echo "ERROR: failed to extract \"$ZIP_MEMBER\" from \"$TMP_ZIP\"" >&2
    exit 1
  fi
  if ! mv -f "$TMP_EXTRACT/$ZIP_MEMBER" "$DEST"; then
    echo "ERROR: cannot move extracted file into place at \"$DEST\"" >&2
    exit 1
  fi
  rm -f "$TMP_ZIP"; rm -rf "$TMP_EXTRACT"
  TMP_ZIP=""; TMP_EXTRACT=""
fi

if [ ! -f "$SHA_FILE" ]; then
  # First time this corpus has been seen here: pin it, don't fail. The
  # checksum covers the extracted corpus file itself, never the zip — that
  # is what perplexity runs actually consume.
  if ! sha256sum "$DEST" > "$SHA_FILE"; then
    echo "ERROR: cannot compute/pin checksum for \"$DEST\"" >&2
    exit 1
  fi
  echo "pinned checksum for \"$DEST\" -> \"$SHA_FILE\""
  exit 0
fi

if ! sha256sum -c "$SHA_FILE"; then
  echo "ERROR: \"$DEST\" failed checksum verification against \"$SHA_FILE\"" \
       "(corpus changed or corrupted; every ppl number measured against it is now suspect)" >&2
  exit 1
fi

echo "\"$DEST\" verified against pinned checksum \"$SHA_FILE\""
