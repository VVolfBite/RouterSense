#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?deployment yaml required}
python -m routersense_sched.pipeline.deploy start --deployment "$CONFIG"
