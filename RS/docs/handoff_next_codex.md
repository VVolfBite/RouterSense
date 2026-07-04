# RouteSense Handoff For Next Codex

## 1. Mainline Status

The formal mainline is `RS/`. Historical POC material under `legacy/` is
reference only and must not be used to mask gaps in the current runtime.

Current distributed runtime status:

- supported runtime mode: `trace_replay`
- unsupported runtime mode: `real_ep`
- current claim scope: `transport_replay_only`
- current future information mode in the replay path: oracle lookahead, not
  online prediction

This means the current distributed path is a replay harness for distributed MoE
transport behavior, not yet a true expert-parallel runtime.

## 2. What Was Corrected In Phase A

The mainline previously mixed misleading names and success semantics. Current
mainline terminology is now:

- `unscheduled_collective_replay`
- `wave_collective_replay`
- `scheduled_collective_partition_replay`

Obsolete names that should not be reintroduced in formal docs or reports:

- `native_baseline`
- `wave_collective`
- `scheduled_transport`

The adapter residency description was also corrected:

- old: `physically_sharded_experts`
- current: `rank_local_expert_weight_cache_from_full_model`

That wording matters because the current adapter still derives rank-local
weights from a full-model load and therefore is not physically sharded.

## 3. Current Result JSON Contract

Current replay results should explicitly expose:

- `execution_mode=trace_replay`
- `claim_scope=transport_replay_only`
- `is_real_ep_runtime=false`
- `uses_oracle_future_trace=true`
- `baseline_semantics=unscheduled_collective_replay|scheduled_collective_replay`
- `correctness_status=not_checked|passed|failed|unsupported`

Validation-disabled runs must report:

- `correctness_status=not_checked`

They must not fabricate a correctness pass.

## 4. Current Capability Boundary

What the current code can support:

- trace collection from the full model
- trace-derived dispatch-plan construction
- distributed collective replay over those plans
- scheduled-versus-unscheduled replay bridge wiring
- replay-scope correctness plumbing and reporting

What the current code does not yet support:

- real EP source ownership semantics
- full local-route preservation and combine semantics
- real EP baseline semantics
- online prediction semantics
- network-realized matching execution semantics

Do not describe the current code as "real EP runtime", "native EP baseline",
or "online prediction" until those pieces are implemented and tested.

## 5. Benchmark Interpretation Rule

Current 2-rank experiments are only suitable for:

- wiring
- correctness protocol
- collective calibration

They are not suitable for:

- scheduler speedup claims
- joint-scheduling benefit claims
- production EP throughput claims

Reason:

- current runtime is still replay-only
- current future-trace use is oracle lookahead
- current collective backend does not realize endpoint matching semantics
- a 2-rank topology does not expose the multi-matching structure that the
  offline PoC relied on

## 6. Files To Read First

If resuming work, read these first:

1. `README.md`
2. `RS/README.md`
3. `RS/report.md`
4. `RS/experiments/distributed/exp_wave_execution.py`
5. `RS/src/rs/runtime/distributed_ep/adapter/runner.py`
6. `RS/src/rs/runtime/distributed_ep/core/wave_executor.py`
7. `RS/src/rs/runtime/distributed_ep/core/wave_planner.py`
8. `RS/src/rs/runtime/distributed_ep/core/manifest.py`
9. `RS/tests/test_scheduled_execution_bridge.py`

## 7. Immediate Next Technical Priorities

After Phase A, the next priorities should remain:

1. restore real dispatch/combine semantics
2. remove synthetic default source ownership
3. stop dropping local routes
4. replace weak replay validation with explicit route and numerical checks
5. add manifest invariants and distributed consistency failures
6. then repair timing fairness and benchmark protocol

Do not jump straight to larger performance runs before those semantic fixes are
in place.
