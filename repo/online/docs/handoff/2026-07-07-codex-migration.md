# Codex Migration Handoff

This file is the short handoff for continuing RouterSense work on the next
machine, expected to be a single-node `2080Ti x4` server.

## Current Goal

The immediate goal is not large-scale policy comparison. The immediate goal is
to identify why the online planning path is too slow, then decide whether the
runtime path is viable before making further scheduling claims.

## Current Repository State

The formal online runtime path is already wired end-to-end:

- `attach_formal_online_runtime()` wraps Megatron dispatcher entrypoints.
- `before_token_dispatch()` / `before_token_combine()` build live phase
  contexts from real dispatcher state.
- `run_phase_plan_agreement()` gathers per-rank context, builds the root plan,
  broadcasts it, and verifies `plan_hash`.
- `MegatronPhaseTransportAdapter` intercepts Megatron `all_to_all` and executes
  the agreed `PhaseExecutionPlan` wave-by-wave.
- `routersense_p0p1p2_hint` consumes `calibrated_artifact` P2 hints online.
- `MultiphasePendingWindowPolicy` performs online joint planning at phase
  boundaries, then compiles the current phase slice back into a standard
  `PhaseExecutionPlan`.

Important boundary:

- This is already online joint planning plus phase-local execution.
- This is not yet a live cross-phase pending queue that interleaves true P0 and
  released P1 payloads inside one runtime-global executor.

## Recent Runtime Timing Instrumentation

The following planning stages are now exported to
`rank*_planning_timing.jsonl`:

- `build_p2_hint`
- `record_window_state`
- `build_phase_ready_context`
- `store_prepared_plan`
- `prepared_phase_plan_shadow`
- `run_phase_plan_agreement`

The following agreement timings are also attached directly to plan/timeline
metrics:

- `all_gather_time_us`
- `build_plan_time_us`
- `broadcast_time_us`
- `verify_time_us`
- `total_agreement_time_us`

Relevant files:

- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/control/plan_agreement.py`
- `experiments/online/support/phase_executor_artifacts.py`
- `experiments/online/support/comparison_metrics.py`

## What We Need Next On 2080Ti x4

Run the smallest possible online experiments to answer one question:

> Why is online plan + agreement + execution so slow?

Start with only these cases:

1. `disabled`
2. `birkhoff_phase_local` or `greedy_ready_set`
3. `routersense_p0p1p2_hint`

Focus on these metrics first:

- `communication_phase_window_us`
- `communication_collective_active_us`
- `all_gather_time_us`
- `build_plan_time_us`
- `broadcast_time_us`
- `verify_time_us`
- `build_p2_hint_time_us`
- `build_phase_ready_context_time_us`
- `store_prepared_plan_time_us`
- `prepared_phase_plan_shadow_time_us`
- `run_phase_plan_agreement_time_us`

Do not start with a large strategy sweep. First profile the runtime path.

## Likely Current Hot Spots

The likely heavy path is one or more of:

- very small `bucket_rows` causing wave explosion,
- Python object-heavy `all_gather_object` / `broadcast_object_list`,
- large plan serialization / hashing payloads,
- oversized diagnostics embedded in plan metrics,
- logical planning and compile both happening in the hot path.

## Offline Replay Guidance

One native collection run is enough for repeated offline scheduling replay.

Use online/native collection to preserve:

- observer facts,
- phase contexts,
- transport bundles,
- scheduled plans,
- transport execution,
- control timeline,
- plan arrival records,
- planning timing,
- prepared phase plan shadows,
- pending-window driver records.

Then use the same collected flow facts to replay many scheduling variants
offline without reopening GPUs.

## `.codex` Migration

A repository-local copy of `~/.codex` has been migrated into:

- `.codex/`

The copied tree intentionally preserves the local Codex working context for
machine-to-machine continuation. The previously exposed API key was redacted in
the migrated copy.
