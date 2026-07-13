# M-HOTPATH-CPU2 Report

## Goal
Close GPU acceptance measurement contracts without running GPU tests.

## Starting SHA
`e3b5750650889d568474243a2c555cee82a59bf1`

## Final SHA
Recorded in the commit containing this dialogue directory and in the final handoff response.

## Root Cause
The previous aggregation rejected missing rank fields but did not enforce exact hook counts or U-zero raw-U build upper bounds. Layer IDs also came from sets/maps without one central stable ordering helper.

## Changes
- Added exact selected/prediction-source hook count contract.
- Added U-zero raw-U build upper-bound checks per rank, all rank, and per selected layer.
- Added stable layer ID sorting helper and applied it to runtime scope and by-layer export fields.
- Added focused tests and refreshed GPU validation manifest content.

## Tests
- `python -m compileall src experiments tests`
- `git diff --check`
- `PYTHONPATH=src:. pytest -q tests/contract/test_gpu_child_config_and_a2_metrics.py tests/contract/test_runtime_measurement_semantics.py`

## Not Run
GPU/NCCL, full 4GPU bring-up, C2, seven-strategy A2, target lifecycle matrix, large workload sweep.

## Current Conclusion
`HOTPATH_CPU_GPU_CONTRACT_READY`

## Next GPU Commands
See `dialogue/evidence/gpu_validation_manifest.json` for the frozen count and performance smoke commands.
