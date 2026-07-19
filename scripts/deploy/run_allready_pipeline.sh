#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
exec "$PYTHON_BIN" "$ROOT/scripts/deploy/run_allready_pipeline.py" "$@"
