#!/usr/bin/env bash
set -euo pipefail

scope="mainline"
if [[ "${1:-}" == "--scope" ]]; then
  scope="${2:?missing scope value}"
  shift 2
fi
archive_path="${1:?missing archive path}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cd "$repo_root"
git rev-parse HEAD > "$tmp_dir/SOURCE_COMMIT.txt"

if [[ "$scope" == "mainline" ]]; then
  tar -czf "$archive_path" \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='RS/deploy/inventory/hosts.local.yaml' \
    --exclude='RS/artifacts/*' --exclude='!RS/artifacts/.gitkeep' \
    --exclude='RS/outputs/*' --exclude='!RS/outputs/.gitkeep' \
    --exclude='RS/deploy/logs/*' --exclude='!RS/deploy/logs/.gitkeep' \
    README.md .gitignore RS legacy/README.md \
    -C "$tmp_dir" SOURCE_COMMIT.txt
else
  tar -czf "$archive_path" \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='RS/deploy/inventory/hosts.local.yaml' \
    --exclude='RS/artifacts/*' --exclude='!RS/artifacts/.gitkeep' \
    --exclude='RS/outputs/*' --exclude='!RS/outputs/.gitkeep' \
    --exclude='RS/deploy/logs/*' --exclude='!RS/deploy/logs/.gitkeep' \
    README.md .gitignore RS legacy \
    -C "$tmp_dir" SOURCE_COMMIT.txt
fi

tar -tzf "$archive_path"
