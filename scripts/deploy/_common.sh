#!/usr/bin/env bash
# Shared deployment bootstrap. Keep deploy entrypoints relocatable while the
# canonical repository-root definitions remain in scripts/_common.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
