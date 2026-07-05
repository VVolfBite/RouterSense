#!/usr/bin/env bash
set -euo pipefail

scope="mainline"
if [[ "${1:-}" == "--scope" ]]; then
  scope="${2:?missing scope value}"
  shift 2
fi
archive_path="${1:?missing archive path}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cd "$repo_root"
if git rev-parse HEAD >/dev/null 2>&1; then
  git rev-parse HEAD > "$tmp_dir/SOURCE_COMMIT.txt"
elif [[ -f "$repo_root/SOURCE_COMMIT.txt" ]]; then
  cp "$repo_root/SOURCE_COMMIT.txt" "$tmp_dir/SOURCE_COMMIT.txt"
else
  echo "unknown" > "$tmp_dir/SOURCE_COMMIT.txt"
fi

if [[ "$scope" == "mainline" ]]; then
  tar -czf "$archive_path" \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='RS/deploy/inventory/hosts.local.yaml' \
    --exclude='RS/artifacts/*' \
    --exclude='RS/artifacts' \
    --exclude='RS/archives' \
    --exclude='RS/archives/*' \
    --exclude='RS/outputs' \
    --exclude='RS/outputs/*' \
    --exclude='RS/deploy/logs' \
    --exclude='RS/deploy/logs/*' \
    --exclude='RS/prompts/logs' \
    --exclude='RS/prompts/logs/*' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='*.ckpt' \
    --exclude='*.log' \
    --exclude='*.jsonl' \
    --exclude='*.npy' \
    --exclude='*.npz' \
    README.md .gitignore RS legacy/README.md \
    -C "$tmp_dir" SOURCE_COMMIT.txt
else
  tar -czf "$archive_path" \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='legacy/**/outputs' \
    --exclude='legacy/**/outputs/*' \
    --exclude='legacy/**/artifacts' \
    --exclude='legacy/**/artifacts/*' \
    --exclude='legacy/**/logs' \
    --exclude='legacy/**/logs/*' \
    --exclude='legacy/**/*.log' \
    --exclude='RS/deploy/inventory/hosts.local.yaml' \
    --exclude='RS/artifacts/*' \
    --exclude='RS/artifacts' \
    --exclude='RS/archives' \
    --exclude='RS/archives/*' \
    --exclude='RS/outputs' \
    --exclude='RS/outputs/*' \
    --exclude='RS/deploy/logs' \
    --exclude='RS/deploy/logs/*' \
    --exclude='RS/prompts/logs' \
    --exclude='RS/prompts/logs/*' \
    --exclude='*.tar.gz' \
    --exclude='*.zip' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='*.ckpt' \
    --exclude='*.log' \
    --exclude='*.jsonl' \
    --exclude='*.npy' \
    --exclude='*.npz' \
    README.md .gitignore RS legacy \
    -C "$tmp_dir" SOURCE_COMMIT.txt
fi

tar -tzf "$archive_path"
