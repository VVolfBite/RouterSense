#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_INVENTORY="${DEFAULT_INVENTORY:-$ROOT/deploy/inventory/hosts.local.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
