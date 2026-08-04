#!/usr/bin/env bash
# bench/harness/fetch_calibration_corpus.sh [DEST]
#
# Fetches and pins the wikitext-2-raw calibration/perplexity corpus that
# every baseline manifest's quality.ppl_dataset (spec §6) is measured
# against. The first run downloads the file and pins its sha256 next to it;
# every later run re-verifies the on-disk file against that pin and fails
# loudly on any mismatch — a silently-changed corpus would invalidate every
# ppl number already measured against it, so verification is the point of
# this script, not the download.
#
# Usage: fetch_calibration_corpus.sh [DEST]
#   DEST  path to the corpus file (default: bench/data/wiki.test.raw)
#         the pin lives alongside it at "$DEST.sha256"
set -uo pipefail

URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw/wiki.test.raw"
DEST=${1:-bench/data/wiki.test.raw}
SHA_FILE="${DEST}.sha256"
DEST_DIR=$(dirname "$DEST")

if ! mkdir -p "$DEST_DIR"; then
  echo "ERROR: cannot create directory \"$DEST_DIR\"" >&2
  exit 1
fi

TMP=""
cleanup() { [ -n "$TMP" ] && rm -f "$TMP"; }
trap cleanup EXIT

if [ ! -f "$DEST" ]; then
  echo "corpus not found at \"$DEST\"; downloading from $URL"
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found; cannot fetch \"$URL\"" >&2
    exit 1
  fi
  TMP=$(mktemp "${DEST}.XXXXXX")
  if ! curl -fL -o "$TMP" "$URL"; then
    echo "ERROR: failed to download \"$URL\" to \"$DEST\"" >&2
    exit 1
  fi
  if ! mv -f "$TMP" "$DEST"; then
    echo "ERROR: cannot move downloaded file into place at \"$DEST\"" >&2
    exit 1
  fi
  TMP=""
fi

if [ ! -f "$SHA_FILE" ]; then
  # First time this corpus has been seen here: pin it, don't fail.
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
