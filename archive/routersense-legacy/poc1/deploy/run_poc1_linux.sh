#!/usr/bin/env bash
# Deployment-facing Linux entrypoint.
# Keeps deployment commands under deploy/ while reusing the existing platform script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT_DIR/scripts/run_poc1_linux.sh" "$@"
