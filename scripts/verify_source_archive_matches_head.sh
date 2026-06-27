#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-$ROOT/../RouterSense.TAR.GZ}"
TMP_DIR="$(mktemp -d)"
TMP_LIST="$(mktemp)"
TMP_SHA="$(mktemp)"

cleanup() {
  rm -rf "$TMP_DIR"
  rm -f "$TMP_LIST" "$TMP_SHA"
}
trap cleanup EXIT

cd "$ROOT"

HEAD_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "verify_source_archive_matches_head.sh: tracked git status is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi
git diff --exit-code >/dev/null

tar -xzf "$ARCHIVE" -C "$TMP_DIR"

ARCHIVE_COMMIT="$(cat "$TMP_DIR/SOURCE_COMMIT.txt")"
if [[ "$ARCHIVE_COMMIT" != "$HEAD_COMMIT" ]]; then
  echo "verify_source_archive_matches_head.sh: archive commit $ARCHIVE_COMMIT != HEAD $HEAD_COMMIT" >&2
  exit 1
fi

find "$TMP_DIR" -type f ! -name 'PACKAGE_MANIFEST.sha256' ! -name 'SOURCE_COMMIT.txt' ! -name 'SOURCE_TREE_SHA256.txt' -printf '%P\n' | sort > "$TMP_LIST"
{
  echo "# RouterSense source package manifest"
  while IFS= read -r relpath; do
    sha256sum "$ROOT/$relpath"
  done < <(git ls-files | sort)
} > "$TMP_SHA"

if ! diff -u "$TMP_SHA" "$TMP_DIR/PACKAGE_MANIFEST.sha256" >/dev/null; then
  echo "verify_source_archive_matches_head.sh: PACKAGE_MANIFEST.sha256 does not match HEAD tree" >&2
  exit 1
fi

TREE_SHA="$(sha256sum "$TMP_DIR/PACKAGE_MANIFEST.sha256" | awk '{print $1}')"
if [[ "$TREE_SHA" != "$(cat "$TMP_DIR/SOURCE_TREE_SHA256.txt")" ]]; then
  echo "verify_source_archive_matches_head.sh: SOURCE_TREE_SHA256.txt does not match manifest hash" >&2
  exit 1
fi

echo "archive matches HEAD: $ARCHIVE"
