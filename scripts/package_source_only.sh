#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/root/autodl-tmp/RouterSense.TAR.GZ}"
TMP_LIST="$(mktemp)"
TMP_TAR_LIST="$(mktemp)"
TMP_MANIFEST="$(mktemp)"
TMP_STAGE_DIR="$(mktemp -d)"

REQUIRED_FILES=(
  "src/routesense_poc2/stress.py"
  "experiment/poc2/stress_suite.py"
  "experiment/poc2/analyze_dependency_predictiveness.py"
  "experiment/poc2/analyze_stress_results.py"
  "docs/poc2_stress_suite_contract.md"
)

cleanup() {
  rm -f "$TMP_LIST" "$TMP_TAR_LIST" "$TMP_MANIFEST"
  rm -rf "$TMP_STAGE_DIR"
}
trap cleanup EXIT

cd "$ROOT"

git ls-files > "$TMP_LIST"
grep -Ev '^(outputs|artifacts|logs)(/|$)|(^|/)\.pytest_cache(/|$)|(^|/)__pycache__(/|$)|(^|/).*\.log$|(^|/).*\.jsonl$|(^|/).*\.npy$|(^|/).*\.npz$|(^|/).*\.pt$|(^|/).*\.pth$|(^|/).*\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_LIST" > "${TMP_LIST}.filtered"
mv "${TMP_LIST}.filtered" "$TMP_LIST"

for required in "${REQUIRED_FILES[@]}"; do
  if ! grep -Fx "$required" "$TMP_LIST" >/dev/null; then
    echo "package_source_only.sh: missing required tracked file: $required" >&2
    exit 1
  fi
done

{
  echo "# RouterSense source package manifest"
  while IFS= read -r path; do
    sha256sum "$path"
  done < "$TMP_LIST"
} > "$TMP_MANIFEST"
cp "$TMP_MANIFEST" "$TMP_STAGE_DIR/PACKAGE_MANIFEST.sha256"

tar -czf "$OUT" -T "$TMP_LIST" -C "$TMP_STAGE_DIR" PACKAGE_MANIFEST.sha256
tar -tzf "$OUT" > "$TMP_TAR_LIST"

if grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_TAR_LIST" >/dev/null; then
  echo "package_source_only.sh: archive contains forbidden runtime or cache content" >&2
  grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)' "$TMP_TAR_LIST" >&2
  exit 1
fi

for required in "${REQUIRED_FILES[@]}"; do
  if ! grep -Fx "$required" "$TMP_TAR_LIST" >/dev/null; then
    echo "package_source_only.sh: archive is missing required file: $required" >&2
    exit 1
  fi
done

if ! grep -Fx "PACKAGE_MANIFEST.sha256" "$TMP_TAR_LIST" >/dev/null; then
  echo "package_source_only.sh: archive is missing PACKAGE_MANIFEST.sha256" >&2
  exit 1
fi

echo "archive: $OUT"
echo "contents:"
cat "$TMP_TAR_LIST"
