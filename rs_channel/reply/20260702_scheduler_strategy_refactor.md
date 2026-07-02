# Scheduler Strategy Refactor Status

Date: 2026-07-02 UTC

## What changed

- Added a unified strategy interface in `RS/src/routesense/scheduler/strategy.py`.
- Added strategy wrappers for the existing pairwise schedulers in `RS/src/routesense/scheduler/strategies.py`.
- Updated `RS/src/routesense/scheduler/__init__.py` to expose strategy APIs and trigger strategy registration.
- Reworked `RS/src/routesense/runtime/distributed_ep/core/scheduler.py` so runtime can hold and switch a named scheduling strategy while keeping the legacy `plan()` compatibility stub.
- Added regression coverage in `RS/tests/test_scheduler_strategy.py`.

## Validation

- `python -m pytest RS/tests/test_scheduler_strategy.py -q`
  - `3 passed`
- `python -m pytest RS/tests/test_poc_line1.py -q`
  - `18 passed`
- `python -m pytest RS/tests -q`
  - `38 passed, 1 warning`

## Current experiment status

- Finished reference artifact:
  - `RS/artifacts/poc_line1/scale_test_n4_sample500/summary.json`
- Background large-scale run still active:
  - PID `43322`
  - target output: `RS/artifacts/poc_line1/scale_test_n8_sample500/`

## Notes

- Strategy mode is additive only. Existing function-style callers in evaluation remain unchanged.
- Current runtime bridge now supports dynamic strategy switching by name, which is the needed hook for later distributed deployment integration.
