#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="legacy/poc1/src${PYTHONPATH:+:$PYTHONPATH}"

python -m routesense_poc1.cli doctor \
  --config legacy/poc1/configs/smoke.yaml \
  --output-dir legacy/poc1/outputs/smoke_doctor

python -m routesense_poc1.cli inspect-model \
  --config legacy/poc1/configs/smoke.yaml \
  --output-dir legacy/poc1/outputs/smoke_inspect

python -m routesense_poc1.cli collect-trace \
  --config legacy/poc1/configs/smoke.yaml \
  --output-dir legacy/poc1/outputs/smoke_trace

python -m routesense_poc1.cli ablate \
  --config legacy/poc1/configs/smoke.yaml \
  --output-dir legacy/poc1/outputs/smoke_ablate
