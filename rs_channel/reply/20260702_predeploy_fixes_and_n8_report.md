# Predeploy Fixes and N=8 Report

Date: 2026-07-02 UTC

## Deployment fixes completed

The following blocking predeploy issues are fixed:

1. `build_distributed_runner_plan()` now uses the actual `rank` as `origin_rank`.
2. `execute_scheduled_inference()` now supports distributed matrix aggregation via `all_reduce`, so the scheduler sees a global communication matrix rather than one rank's local shards.
3. `NCCLExecutor` now resolves its default CUDA device from `LOCAL_RANK` instead of `torch.cuda.current_device()`.
4. `NCCLExecutor.execute_phase()` now posts all sends asynchronously, waits receives first, then waits sends, which is safer than per-send blocking waits.
5. `execute_scheduled_inference()` now inserts an expert-compute delay simulation before combine when `expert_compute_delay > 0`.
6. `run_scheduling_experiment.sh` now emits `NCCL_DEBUG=INFO`, `NCCL_SOCKET_TIMEOUT=30`, and `NCCL_IB_DISABLE=0`.

## Validation

- `python -m py_compile` on updated runtime + experiment files: passed
- `bash -n RS/scripts/run_scheduling_experiment.sh`: passed
- `python -m pytest RS/tests/test_scheduled_execution_bridge.py RS/tests/test_multi_rank_aggregation.py -q`
  - `4 passed`
- `python -m pytest RS/tests -q`
  - `42 passed, 1 warning`

## N=8 sample64 results

### Strong-candidate screen

Artifact:
- `RS/artifacts/poc_line1/scale_test_n8_sample64_strong_light/summary.json`

Key results:
- `birkhoff`: `19.45%` improvement, `0.55ms`
- `ibbr`: `21.86%` improvement, `5.11ms`
- `simulated_annealing`: `20.88%` improvement, `3.44ms`
- `grasp`: `6.83%` improvement, `3.07ms`

Interim read:
- `ibbr`, `simulated_annealing`, and `birkhoff` remain the only useful candidates in this light screen.
- `grasp` is weak and no longer competitive.

### Oracle focus

Artifact:
- `RS/artifacts/poc_line1/scale_test_n8_sample64_oracle_focus/summary.json`

Key results:
- `oracle_perfect`: `44.04%` mean improvement
- `oracle_predicted`: `44.30%` mean improvement
- `birkhoff`: `19.45%` mean improvement
- solver statuses:
  - perfect: `656 OPTIMAL`, `304 FEASIBLE`
  - predicted: `666 OPTIMAL`, `294 FEASIBLE`

## Interpretation

The current N=8 data does **not** support the claim that larger N reduces the optimization space.

What actually happened:
- The **oracle gap increased** from the earlier N=4 result (~40%) to N=8 (~44%).
- The **fast heuristics did not scale up proportionally**, so the gap between practical heuristics and oracle widened.

This means:
- cross-phase scheduling value is still present, and possibly larger, at N=8;
- the current heuristic family is the bottleneck, not the problem itself.

In short: larger N did not collapse the opportunity; it exposed that our practical schedulers are not yet capturing enough of the available joint-scheduling gain.
