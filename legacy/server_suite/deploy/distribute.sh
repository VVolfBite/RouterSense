#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?deployment yaml required}
BUNDLE=${2:?bundle directory required}
python -m routersense_sched.pipeline.deploy distribute --deployment "$CONFIG" --bundle "$BUNDLE"
