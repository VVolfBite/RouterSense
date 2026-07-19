#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if (( $# == 0 )); then set -- "$DEFAULT_INVENTORY"; fi
exec "$PYTHON_BIN" "$ROOT/scripts/deploy/prepare_cluster_environment.py" "$@"
