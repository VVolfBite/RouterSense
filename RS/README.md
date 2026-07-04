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
- `online` now contains:
  - a verified `world_size=1` local-MoE reconstruction parity harness
  - a verified `world_size=2` route-partition and metadata/count-agreement
    harness
  - but it is not yet a real EP runtime
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
- online single-rank local-MoE observation:
  `experiments/online/collect_native_ep_trace.py`
- online scheduled EP placeholder:
  `experiments/online/bench_scheduled_ep.py`
- deprecated legacy replay shim:
  `experiments/legacy/exp_trace_replay.py`

This tree is the only formal RouteSense development path. Legacy POC code remains outside `RS/`.

## Online Boundary

Current online-adjacent capability is intentionally narrow:

- `bench_native_ep.py --world-size 1` verifies single-layer local-MoE
  reconstruction parity against a captured HuggingFace OLMoE `mlp(...)` output
- `collect_native_ep_trace.py --world-size 1` exports a
  `trace_origin=observed_single_rank_local_moe` artifact
- `torchrun --nproc_per_node=2 experiments/online/bench_native_ep.py --world-size 2 --route-partition-only ...`
  verifies rank-local route construction, local/remote partition, placement
  hashing, and distributed send/recv count agreement
- `torchrun --nproc_per_node=2 experiments/online/bench_native_ep.py --world-size 2 --route-partition-only --hidden-dispatch-only ...`
  additionally performs truthful hidden-state dispatch on the remote routes
- `torchrun --nproc_per_node=2 experiments/online/collect_native_ep_trace.py --world-size 2 --route-partition-only ...`
  exports a truthful
  `trace_origin=observed_online_ws2_route_partition` artifact
- `torchrun --nproc_per_node=2 experiments/online/collect_native_ep_trace.py --world-size 2 --route-partition-only --hidden-dispatch-only ...`
  exports a truthful
  `trace_origin=observed_online_ws2_hidden_dispatch` artifact

This does not constitute:

- native A2A EP runtime completion
- hidden-state combine
- remote expert compute
- distributed numerical MoE parity
- transport-calibrated observation
- multi-rank expert residency validation

`fit_ep_cost_model.py` now rejects the single-rank local-MoE artifact. Formal
calibrated offline analysis also rejects the `world_size=2`
`observed_online_ws2_route_partition` artifact because it still lacks real
dispatch/combine hidden transport, expert compute, and distributed numerical
correctness. It also rejects the `observed_online_ws2_hidden_dispatch` artifact
because it still lacks expert compute, inverse combine, and distributed
numerical correctness. Formal calibrated offline analysis remains gated on
future multi-rank `observed_online_native_ep` traces with real transport and
timing data.
