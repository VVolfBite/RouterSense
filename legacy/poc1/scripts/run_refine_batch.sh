#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RouterSense

configs=(
  "legacy/poc1/configs/code_long.yaml:legacy/poc1/outputs/code_long_0924"
  "legacy/poc1/configs/math_long.yaml:legacy/poc1/outputs/math_long_0924"
  "legacy/poc1/configs/dialog_long.yaml:legacy/poc1/outputs/dialog_long_0924"
  "legacy/poc1/configs/all_layers_code.yaml:legacy/poc1/outputs/all_layers_code_0924"
)

for item in "${configs[@]}"; do
  config="${item%%:*}"
  output="${item##*:}"
  echo "[run_refine_batch] running $config -> $output"
  PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli run-all --config "$config" --output-dir "$output"
done
