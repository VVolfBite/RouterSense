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
TMP_EXPECTED="$(mktemp)"
TMP_ACTUAL="$(mktemp)"

cleanup() {
  rm -rf "$TMP_DIR"
  rm -f "$TMP_EXPECTED" "$TMP_ACTUAL"
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

python - "$SCOPE" > "$TMP_EXPECTED" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

scope = sys.argv[1]
paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
blocked_prefixes = (
    "artifacts/", "outputs/", "logs/", ".pytest_cache/", "__pycache__/",
    "model_cache/", "hf_cache/", ".cache/", "venv/", ".venv/", "deploy/logs/",
)
blocked_suffixes = (".log", ".jsonl", ".npy", ".npz", ".pt", ".pth", ".safetensors")
keep = []
for path in sorted(paths):
    if path in {'RS/artifacts/.gitkeep', 'RS/outputs/.gitkeep', 'RS/deploy/logs/.gitkeep'}:
        continue
    if any(path.startswith(prefix) for prefix in blocked_prefixes):
        continue
    if any(path.endswith(suffix) for suffix in blocked_suffixes):
        continue
    if scope == "mainline":
        if path.startswith(("legacy/poc1/", "legacy/poc2/", "experiment/poc1/", "experiment/poc2/", "src/routesense_poc1/", "src/routesense_poc2/")):
            continue
        if path in {"configs/poc1.yaml", "docs/poc2_correctness_audit.md", "docs/poc2_simulation_contract.md", "docs/poc2_stress_suite_contract.md", "deploy/run_poc1_linux.sh", "deploy/run_poc1_windows.ps1", "scripts/run_poc1.sh", "scripts/run_poc1_linux.sh", "scripts/run_poc1_windows.ps1", "scripts/trace_one_token.py", "scripts/run_action.py"}:
            continue
    keep.append(path)
print("# RouterSense source package manifest")
root = Path('.').resolve()
for rel in keep:
    data = (root / rel).read_bytes()
    print(f"{hashlib.sha256(data).hexdigest()}  {rel}")
PY

python - "$TMP_DIR/PACKAGE_MANIFEST.sha256" > "$TMP_ACTUAL" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).read_text(encoding='utf-8'), end='')
PY

if ! diff -u "$TMP_EXPECTED" "$TMP_ACTUAL" >/dev/null; then
  echo "verify_source_archive_matches_head.sh: PACKAGE_MANIFEST.sha256 does not match HEAD tree" >&2
  diff -u "$TMP_EXPECTED" "$TMP_ACTUAL" >&2 || true
  exit 1
fi

TREE_SHA="$(sha256sum "$TMP_DIR/PACKAGE_MANIFEST.sha256" | awk '{print $1}')"
if [[ "$TREE_SHA" != "$(cat "$TMP_DIR/SOURCE_TREE_SHA256.txt")" ]]; then
  echo "verify_source_archive_matches_head.sh: SOURCE_TREE_SHA256.txt does not match manifest hash" >&2
  exit 1
fi

echo "archive matches HEAD: $ARCHIVE"
