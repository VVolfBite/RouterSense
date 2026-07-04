# RouteSense Handoff For Next Codex

## 1. Boundary First

The formal mainline now has three lanes:

- `offline`
- `online`
- `legacy`

Treat them as different truth domains.

## 2. What They Mean

### `offline`

Use for:

- router prediction analysis
- oracle/full-trace studies
- calibrated counterfactual analysis

Do not use for:

- production EP latency claims
- deployable prediction claims when using oracle future trace

### `online`

Reserved for:

- real per-rank local input ownership
- real expert residency
- real EP execution
- no future truth in the runtime hot path

Current Phase 1 status:

- package skeleton exists
- real runtime is not implemented yet

### `legacy`

Use only for:

- deprecated compatibility path of the current distributed trace replay harness

The current old harness must be described as:

- `execution_mode=legacy_trace_replay`
- `pipeline=legacy`
- `trace_origin=legacy_trace_replay`

It is not online EP runtime.

## 3. New Mainline Layout

Important new package roots:

- `src/rs/contracts/`
- `src/rs/offline/`
- `src/rs/online/`
- `src/rs/legacy/`

Important new experiment roots:

- `experiments/offline/`
- `experiments/online/`
- `experiments/legacy/`

## 4. Provenance Rules

Current shared metadata contract includes:

- `pipeline`
- `claim_scope`
- `trace_origin`
- `future_information_mode`
- `is_real_ep_runtime`
- `source_ownership_mode`
- `expert_residency_mode`
- `transport_backend`
- `correctness_status`
- `performance_claim_eligible`

Key rules already enforced:

- legacy replay cannot present itself as online
- offline calibrated analysis must reject non-`observed_online_native_ep` input
- online scheduler hint mode must reject `oracle_full_trace`
- all-to-all backend is not marked as matching-realized

## 5. Current Runnable Entry Points

Runnable now:

1. `experiments/offline/exp_router_prediction.py`
2. `experiments/legacy/exp_trace_replay.py`

Present but expected to fail fast:

1. `experiments/offline/fit_ep_cost_model.py`
2. `experiments/offline/exp_calibrated_schedule.py`
3. `experiments/online/collect_native_ep_trace.py`
4. `experiments/online/bench_native_ep.py`
5. `experiments/online/bench_scheduled_ep.py`

That is intentional. Phase 1 is about truthful boundaries, not pretending the
online runtime exists.

## 6. What To Work On Next

Next real milestone is Phase 2:

1. real online native EP ownership and routing
2. full-checkpoint-then-prune expert residency
3. native variable-size A2A dispatch/combine
4. world-size-1 parity
5. world-size-2 correctness
6. online observer trace export

Do not skip to scheduled P2P benchmark claims before native online EP is
implemented and validated.
