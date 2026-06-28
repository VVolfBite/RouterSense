#!/usr/bin/env bash
set -euo pipefail

SCOPE="mainline"
ARCHIVE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *)
      ARCHIVE="$1"
      shift
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$ARCHIVE" ]]; then
  ARCHIVE="$ROOT/../RouterSense.TAR.GZ"
fi

TMP_DIR="$(mktemp -d)"
TMP_EXPECTED_ARCHIVE="$(mktemp --suffix=.tar.gz)"
cleanup() {
  rm -rf "$TMP_DIR"
  rm -f "$TMP_EXPECTED_ARCHIVE"
}
trap cleanup EXIT

cd "$ROOT"

HEAD_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "verify_source_archive_matches_head.sh: tracked git status is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

tar -xzf "$ARCHIVE" -C "$TMP_DIR"
if [[ ! -f "$TMP_DIR/SOURCE_COMMIT.txt" ]]; then
  echo "verify_source_archive_matches_head.sh: archive missing SOURCE_COMMIT.txt" >&2
  exit 1
fi
ARCHIVE_COMMIT="$(cat "$TMP_DIR/SOURCE_COMMIT.txt")"
if [[ "$ARCHIVE_COMMIT" != "$HEAD_COMMIT" ]]; then
  echo "verify_source_archive_matches_head.sh: archive commit $ARCHIVE_COMMIT != HEAD $HEAD_COMMIT" >&2
  exit 1
fi

bash "$ROOT/scripts/package_source_only.sh" --scope "$SCOPE" "$TMP_EXPECTED_ARCHIVE" >/dev/null
if ! cmp -s <(tar -xOzf "$TMP_EXPECTED_ARCHIVE" PACKAGE_MANIFEST.sha256) <(tar -xOzf "$ARCHIVE" PACKAGE_MANIFEST.sha256); then
  echo "verify_source_archive_matches_head.sh: PACKAGE_MANIFEST.sha256 does not match HEAD tree" >&2
  diff -u <(tar -xOzf "$TMP_EXPECTED_ARCHIVE" PACKAGE_MANIFEST.sha256) <(tar -xOzf "$ARCHIVE" PACKAGE_MANIFEST.sha256) >&2 || true
  exit 1
fi
if ! cmp -s <(tar -xOzf "$TMP_EXPECTED_ARCHIVE" SOURCE_TREE_SHA256.txt) <(tar -xOzf "$ARCHIVE" SOURCE_TREE_SHA256.txt); then
  echo "verify_source_archive_matches_head.sh: SOURCE_TREE_SHA256.txt does not match manifest hash" >&2
  exit 1
fi

echo "archive matches HEAD: $ARCHIVE"
