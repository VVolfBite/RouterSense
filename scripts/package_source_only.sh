#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/root/autodl-tmp/RouterSense.TAR.GZ}"
TMP_LIST="$(mktemp)"
TMP_TAR_LIST="$(mktemp)"

cleanup() {
  rm -f "$TMP_LIST" "$TMP_TAR_LIST"
}
trap cleanup EXIT

cd "$ROOT"

git ls-files > "$TMP_LIST"
grep -Ev '^(outputs|artifacts|logs)(/|$)|(^|/)\.pytest_cache(/|$)|(^|/)__pycache__(/|$)|(^|/).*\.log$|(^|/).*\.jsonl$|(^|/).*\.npy$|(^|/).*\.npz$|(^|/).*\.pt$|(^|/).*\.pth$|(^|/).*\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_LIST" > "${TMP_LIST}.filtered"
mv "${TMP_LIST}.filtered" "$TMP_LIST"

tar -czf "$OUT" -T "$TMP_LIST"
tar -tzf "$OUT" > "$TMP_TAR_LIST"

if grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_TAR_LIST" >/dev/null; then
  echo "package_source_only.sh: archive contains forbidden runtime or cache content" >&2
  grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_TAR_LIST" >&2
  exit 1
fi

echo "archive: $OUT"
echo "contents:"
cat "$TMP_TAR_LIST"
