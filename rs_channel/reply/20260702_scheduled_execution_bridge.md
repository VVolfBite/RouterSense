# Scheduled Execution Bridge Status

Date: 2026-07-02 UTC

## What was added

- Real execution backend:
  - `RS/src/routesense/runtime/distributed_ep/core/nccl_executor.py`
- Scheduled phase execution hook:
  - `RS/src/routesense/runtime/distributed_ep/core/collective.py`
- Scheduler-to-executor bridge:
  - `RS/src/routesense/runtime/distributed_ep/adapter/runner.py`
- Real experiment entrypoint:
  - `RS/experiments/distributed/exp_scheduled_execution.py`
- Cluster launch helper:
  - `RS/scripts/run_scheduling_experiment.sh`

## Functional effect

- The runtime can now take a scheduler result and convert it into per-rank ordered send/recv lists.
- `CollectiveOps` now has a scheduled execution path without changing the old `dispatch()` / `return_results()` APIs.
- The new experiment entrypoint supports:
  - `--single-gpu` debug mode
  - `torchrun` multi-rank execution mode
- Inventory-driven deployment stays aligned with the existing topology schema under `deploy/inventory/hosts.local.yaml`.

## Validation

- `python -m py_compile` on new runtime and experiment files: passed
- `bash -n RS/scripts/run_scheduling_experiment.sh`: passed
- `python experiments/distributed/exp_scheduled_execution.py --help`: passed
- `python -m pytest RS/tests/test_scheduled_execution_bridge.py -q`: `2 passed`
- `python -m pytest RS/tests/test_poc_line1.py -q`: `18 passed`
- `python -m pytest RS/tests -q`: `40 passed, 1 warning`

## Referenced artifacts

- Stable scheduler benchmark result:
  - `RS/artifacts/poc_line1/scale_test_n4_sample500/summary.json`
- Ongoing large-scale background run:
  - `RS/artifacts/poc_line1/scale_test_n8_sample500/`

## Notes

- The bridge preserves current function-style evaluation code.
- Greedy-style strategies that do not emit an explicit schedule fall back to a deterministic LPT-style synthesized chunk order, so the execution layer still has something concrete to consume.
