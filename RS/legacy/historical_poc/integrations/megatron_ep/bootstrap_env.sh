#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$ROOT/requirements-lock.txt"

python -m pip install --upgrade pip
python -m pip install -r "$REQ"

echo "Megatron EP environment bootstrap complete."
