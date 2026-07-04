# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/rs/contracts/`: shared serializable contracts for offline, online, and legacy paths.
- `src/rs/offline/`: oracle-allowed analysis, calibration, simulator, and reporting code.
- `src/rs/online/`: future online EP runtime namespace.
- `src/rs/legacy/`: deprecated compatibility shims and legacy trace replay metadata.
- `src/rs/`: main runtime, trace, topology, evaluation, and shared package code.
- `src/rs/runtime/single_gpu.py`: single-GPU real inference helpers.
- `src/rs/runtime/distributed_ep/core/`: model-agnostic communication and runtime control.
- `src/rs/runtime/distributed_ep/adapter/`: OLMoE-specific EP adapter layer.
- `deploy/`: remote minimal runtime environment and cluster scripts.
- `scripts/`: single-GPU smoke, architecture probe, cluster checks, and deployment helpers.
- `experiments/poc_line1/`: historical / retained offline scheduler analysis and trace-driven experiments.
- `experiments/offline/`: formal offline entrypoints.
- `experiments/online/`: formal online entrypoints.
- `experiments/legacy/`: deprecated legacy compatibility entrypoints.
- `experiments/distributed/`: deprecated compatibility bring-up scripts for legacy trace replay.
- `experiments/ablation/`: reserved configs/scripts for real ablation follow-up.
- `configs/`: project-level model and topology configuration.
- `docs/`: architecture and deployment contracts.
- `tests/`: mainline regression tests.

## Current Phase

The current mainline is being prepared for a semantically correct distributed
OLMoE bring-up. Single-node validation lives in `scripts/`, and distributed EP
code is being rebuilt under `src/rs/runtime/distributed_ep/` with a strict
`core/adapter` split.

Important scope boundary:

- `offline` may use oracle/full-trace information and must not be presented as
  deployed EP runtime measurement
- `online` is the future real EP runtime namespace, but Phase 1 does not
  implement it yet
- the old distributed execution harness is now explicitly
  `legacy_trace_replay`
- current 2-rank results are only eligible for wiring, correctness protocol,
  and collective calibration claims
- current future-trace use in the legacy replay path is oracle lookahead, not
  online prediction
- offline scheduler wins are not evidence of NCCL wall-clock speedup until the
  transport backend realizes the corresponding execution semantics

## Formal Experiment Entry Points

- offline router trace:
  `experiments/offline/exp_router_prediction.py`
- offline calibrated analysis gate:
  `experiments/offline/fit_ep_cost_model.py`
- offline calibrated schedule placeholder:
  `experiments/offline/exp_calibrated_schedule.py`
- online native EP placeholder:
  `experiments/online/bench_native_ep.py`
- online scheduled EP placeholder:
  `experiments/online/bench_scheduled_ep.py`
- deprecated legacy replay shim:
  `experiments/legacy/exp_trace_replay.py`

This tree is the only formal RouteSense development path. Legacy POC code remains outside `RS/`.
