#!/usr/bin/env bash
set -euo pipefail

SCOPE="mainline"
OUT=""

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
      OUT="$1"
      shift
      ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="/root/autodl-tmp/RouterSense.TAR.GZ"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_LIST="$(mktemp)"
TMP_TAR_LIST="$(mktemp)"
TMP_MANIFEST="$(mktemp)"
TMP_TREE_SHA="$(mktemp)"
TMP_COMMIT="$(mktemp)"
TMP_STAGE_DIR="$(mktemp -d)"

cleanup() {
  rm -f "$TMP_LIST" "$TMP_TAR_LIST" "$TMP_MANIFEST" "$TMP_TREE_SHA" "$TMP_COMMIT"
  rm -rf "$TMP_STAGE_DIR"
}
trap cleanup EXIT

cd "$ROOT"

list_files() {
  python - "$SCOPE" <<'PY'
import subprocess
import sys
scope = sys.argv[1]
paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
blocked_prefixes = (
    'artifacts/', 'outputs/', 'logs/', '.pytest_cache/', '__pycache__/',
    'model_cache/', 'hf_cache/', '.cache/', 'venv/', '.venv/',
    'deploy/logs/',
)

blocked_suffixes = ('.log', '.jsonl', '.npy', '.npz', '.pt', '.pth', '.safetensors')

keep = []
for path in sorted(paths):
    if any(path.startswith(prefix) for prefix in blocked_prefixes):
        continue
    if any(path.endswith(suffix) for suffix in blocked_suffixes):
        continue
    if scope == 'mainline':
        if path.startswith('legacy/') and path != 'legacy/README.md':
            continue
        if path.startswith('experiment/poc1/') or path.startswith('experiment/poc2/'):
            continue
        if path.startswith('src/routesense_poc1/') or path.startswith('src/routesense_poc2/'):
            continue
        if path in {'configs/poc1.yaml', 'docs/poc2_correctness_audit.md', 'docs/poc2_simulation_contract.md', 'docs/poc2_stress_suite_contract.md'}:
            continue
        if path.startswith('scripts/') and path not in {'scripts/package_source_only.sh', 'scripts/verify_source_archive_matches_head.sh', 'scripts/README.md'}:
            continue
    keep.append(path)

for path in keep:
    print(path)
PY
}

list_files > "$TMP_LIST"

case "$SCOPE" in
  mainline)
    REQUIRED_FILES=(
      "README.md"
      "RS/README.md"
      "RS/pyproject.toml"
      "RS/artifacts/.gitkeep"
      "RS/outputs/.gitkeep"
      "RS/deploy/remote/.gitkeep"
      "RS/src/routesense/scheduler/.gitkeep"
      "RS/src/routesense/oracle/.gitkeep"
      "RS/configs/placement/.gitkeep"
      "RS/configs/workload/.gitkeep"
      "RS/configs/scheduler/.gitkeep"
      "RS/experiments/baseline/.gitkeep"
      "RS/experiments/stress/.gitkeep"
      "RS/experiments/paper/.gitkeep"
      "RS/src/routesense/__init__.py"
      "RS/src/routesense/runtime/single_gpu.py"
      "RS/src/routesense/trace/olmoe_router_trace.py"
      "RS/src/routesense/topology/inventory.py"
      "RS/src/routesense/topology/paths.py"
      "RS/src/routesense/evaluation/artifacts.py"
      "RS/deploy/README.md"
      "RS/deploy/inventory/README.md"
      "RS/deploy/inventory/hosts.example.yaml"
      "RS/deploy/inventory/hosts.local.yaml.example"
      "RS/deploy/scripts/check_cluster_access.sh"
      "RS/deploy/scripts/check_repo_parity.sh"
      "RS/deploy/scripts/sync_repo.sh"
      "RS/deploy/scripts/sync_model_cache.sh"
      "RS/deploy/scripts/launch_remote.sh"
      "RS/deploy/scripts/collect_logs.sh"
      "RS/deploy/scripts/stop_rs_jobs.sh"
      "RS/deploy/scripts/verify_gpu_env.py"
      "RS/experiments/deployment/single_gpu_olmoe_smoke.py"
      "RS/experiments/deployment/single_gpu_text_infer.py"
      "RS/experiments/deployment/single_gpu_router_trace_smoke.py"
      "RS/experiments/deployment/future_multinode_smoke.py"
      "RS/docs/router_trace_schema.md"
      "RS/docs/architecture_boundary.md"
      "RS/configs/model/olmoe_1b_7b_instruct.yaml"
      "RS/configs/topology/torchrun_2node_2gpu.yaml"
      "RS/tests/test_no_legacy_imports.py"
      "legacy/README.md"
      "scripts/package_source_only.sh"
      "scripts/verify_source_archive_matches_head.sh"
      "scripts/README.md"
      ".gitignore"
    )
    ;;
  full)
    REQUIRED_FILES=(
      "README.md"
      "RS/README.md"
      "RS/pyproject.toml"
      "legacy/README.md"
      "legacy/poc1/src/routesense_poc1/__init__.py"
      "legacy/poc2/src/routesense_poc2/__init__.py"
      "scripts/package_source_only.sh"
      "scripts/verify_source_archive_matches_head.sh"
      "scripts/README.md"
      ".gitignore"
    )
    ;;
  *)
    echo "unsupported scope: $SCOPE" >&2
    exit 2
    ;;
esac

sort -u "$TMP_LIST" -o "$TMP_LIST"

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

git rev-parse HEAD > "$TMP_COMMIT"
sha256sum "$TMP_MANIFEST" | awk '{print $1}' > "$TMP_TREE_SHA"
cp "$TMP_MANIFEST" "$TMP_STAGE_DIR/PACKAGE_MANIFEST.sha256"
cp "$TMP_COMMIT" "$TMP_STAGE_DIR/SOURCE_COMMIT.txt"
cp "$TMP_TREE_SHA" "$TMP_STAGE_DIR/SOURCE_TREE_SHA256.txt"

tar -czf "$OUT" -T "$TMP_LIST" -C "$TMP_STAGE_DIR" PACKAGE_MANIFEST.sha256 SOURCE_COMMIT.txt SOURCE_TREE_SHA256.txt
tar -tzf "$OUT" > "$TMP_TAR_LIST"

for forbidden in '^(?:outputs|artifacts|logs)(/|$)' '(^|/)\.pytest_cache(/|$)' '(^|/)__pycache__(/|$)' '(^|/).*\.log$' '(^|/).*\.jsonl$' '(^|/).*\.npy$' '(^|/).*\.npz$' '(^|/).*\.pt$' '(^|/).*\.pth$' '(^|/).*\.safetensors$' '(^|/)\.cache(/|$)' '(^|/)venv(/|$)' '(^|/)\.venv(/|$)' '^deploy/logs(/|$)' '^model_cache(/|$)' '^hf_cache(/|$)'; do
  if grep -Eq "$forbidden" <(grep -v '^RS/' "$TMP_TAR_LIST"); then
    echo "package_source_only.sh: archive contains forbidden runtime or cache content" >&2
    grep -En "$forbidden" <(grep -v '^RS/' "$TMP_TAR_LIST") >&2
    exit 1
  fi
done

for required in "${REQUIRED_FILES[@]}" PACKAGE_MANIFEST.sha256 SOURCE_COMMIT.txt SOURCE_TREE_SHA256.txt; do
  if ! grep -Fx "$required" "$TMP_TAR_LIST" >/dev/null; then
    echo "package_source_only.sh: archive is missing required file: $required" >&2
    exit 1
  fi
done

if ! cmp -s "$TMP_COMMIT" <(tar -xOzf "$OUT" SOURCE_COMMIT.txt); then
  echo "package_source_only.sh: SOURCE_COMMIT.txt does not match git HEAD" >&2
  exit 1
fi

echo "scope: $SCOPE"
echo "archive: $OUT"
echo "contents:"
cat "$TMP_TAR_LIST"
