#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?deployment yaml required}
RUN_NAME=${2:?run name required}
DEST=${3:-outputs}
python -m routersense_sched.pipeline.deploy collect --deployment "$CONFIG" --run-name "$RUN_NAME" --destination "$DEST"
