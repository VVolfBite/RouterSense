#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/root/autodl-tmp/RouterSense.TAR.GZ}"
TMP_LIST="$(mktemp)"
TMP_TAR_LIST="$(mktemp)"
TMP_MANIFEST="$(mktemp)"
TMP_TREE_SHA="$(mktemp)"
TMP_COMMIT="$(mktemp)"
TMP_STAGE_DIR="$(mktemp -d)"

REQUIRED_FILES=(
  "src/routesense_poc2/stress.py"
  "experiment/poc2/stress_suite.py"
  "experiment/poc2/analyze_dependency_predictiveness.py"
  "experiment/poc2/analyze_stress_results.py"
  "docs/poc2_stress_suite_contract.md"
  "src/routesense/__init__.py"
  "src/routesense/runtime/single_gpu.py"
  "src/routesense/trace/olmoe_router_trace.py"
  "src/routesense/topology/inventory.py"
  "src/routesense/evaluation/artifacts.py"
  "deploy/inventory/hosts.example.yaml"
  "deploy/inventory/hosts.local.yaml.example"
  "deploy/scripts/check_cluster_access.sh"
  "deploy/scripts/check_repo_parity.sh"
  "deploy/scripts/sync_repo.sh"
  "deploy/scripts/sync_model_cache.sh"
  "deploy/scripts/launch_remote.sh"
  "deploy/scripts/collect_logs.sh"
  "deploy/scripts/stop_rs_jobs.sh"
  "experiments/deployment/single_gpu_olmoe_smoke.py"
  "experiments/deployment/single_gpu_text_infer.py"
  "experiments/deployment/single_gpu_router_trace_smoke.py"
  "experiments/deployment/future_multinode_smoke.py"
  "docs/router_trace_schema.md"
  "configs/model/olmoe_1b_7b_instruct.yaml"
  "configs/topology/torchrun_2node_2gpu.yaml"
)

cleanup() {
  rm -f "$TMP_LIST" "$TMP_TAR_LIST" "$TMP_MANIFEST" "$TMP_TREE_SHA" "$TMP_COMMIT"
  rm -rf "$TMP_STAGE_DIR"
}
trap cleanup EXIT

cd "$ROOT"

git ls-files | sort > "$TMP_LIST"
grep -Ev '^(outputs|artifacts|logs)(/|$)|(^|/)\.pytest_cache(/|$)|(^|/)__pycache__(/|$)|(^|/).*\.log$|(^|/).*\.jsonl$|(^|/).*\.npy$|(^|/).*\.npz$|(^|/).*\.pt$|(^|/).*\.pth$|(^|/).*\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)|^deploy/logs(/|$)|^model_cache(/|$)|^hf_cache(/|$)' "$TMP_LIST" > "${TMP_LIST}.filtered"
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
git rev-parse HEAD > "$TMP_COMMIT"
sha256sum "$TMP_MANIFEST" | awk '{print $1}' > "$TMP_TREE_SHA"
cp "$TMP_MANIFEST" "$TMP_STAGE_DIR/PACKAGE_MANIFEST.sha256"
cp "$TMP_COMMIT" "$TMP_STAGE_DIR/SOURCE_COMMIT.txt"
cp "$TMP_TREE_SHA" "$TMP_STAGE_DIR/SOURCE_TREE_SHA256.txt"

tar -czf "$OUT" -T "$TMP_LIST" -C "$TMP_STAGE_DIR" PACKAGE_MANIFEST.sha256 SOURCE_COMMIT.txt SOURCE_TREE_SHA256.txt
tar -tzf "$OUT" > "$TMP_TAR_LIST"

if grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)|^deploy/logs(/|$)|^model_cache(/|$)|^hf_cache(/|$)' "$TMP_TAR_LIST" >/dev/null; then
  echo "package_source_only.sh: archive contains forbidden runtime or cache content" >&2
  grep -E '(^|/)(outputs|artifacts|logs)(/|$)|\.pytest_cache/|__pycache__/|\.log$|\.jsonl$|\.npy$|\.npz$|\.pt$|\.pth$|\.safetensors$|(^|/)\.cache(/|$)|(^|/)venv(/|$)|(^|/)\.venv(/|$)|^deploy/logs(/|$)|^model_cache(/|$)|^hf_cache(/|$)' "$TMP_TAR_LIST" >&2
  exit 1
fi

for required in "${REQUIRED_FILES[@]}"; do
  if ! grep -Fx "$required" "$TMP_TAR_LIST" >/dev/null; then
    echo "package_source_only.sh: archive is missing required file: $required" >&2
    exit 1
  fi
done

for required in PACKAGE_MANIFEST.sha256 SOURCE_COMMIT.txt SOURCE_TREE_SHA256.txt; do
  if ! grep -Fx "$required" "$TMP_TAR_LIST" >/dev/null; then
    echo "package_source_only.sh: archive is missing $required" >&2
    exit 1
  fi
done

if ! cmp -s "$TMP_COMMIT" <(tar -xOzf "$OUT" SOURCE_COMMIT.txt); then
  echo "package_source_only.sh: SOURCE_COMMIT.txt does not match git HEAD" >&2
  exit 1
fi

echo "archive: $OUT"
echo "contents:"
cat "$TMP_TAR_LIST"
