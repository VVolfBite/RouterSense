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

Current status is split in two:

- Phase 2a completed:
  - `world_size=1` local-MoE reconstruction parity
  - truthful `observed_single_rank_local_moe` observation export
- Phase 2b partially completed:
  - `world_size=2` route partition and metadata/count agreement
  - truthful `observed_online_ws2_route_partition` observation export
- Phase 2c partially completed:
  - `world_size=2` hidden-state dispatch only
  - truthful `observed_online_ws2_hidden_dispatch` observation export
- Phase 2d partially completed:
  - WS=2 MoE-layer harness with dispatch -> owner compute -> inverse combine
  - distributed numerical parity at the single-layer harness level
- Still not completed:
  - real full-model EP runtime
  - true expert shard checkpoint residency
  - multi-layer distributed forward replacement
  - multi-node validation

### `legacy`

Use only for:

- deprecated compatibility path of the current distributed trace replay harness

The current old harness must be described as:

- `execution_mode=legacy_trace_replay`
- `pipeline=legacy`
- `trace_origin=legacy_trace_replay`

It is not online EP runtime.

## 3. New Mainline Layout

This note is historical. The current formal mainline now lives under:

- `src/rs/core/`
- `src/rs/scheduling/`
- `src/rs/runtime/offline/`
- `src/rs/runtime/online/megatron_ep/`

Important new experiment roots:

- `experiments/offline/`
- `experiments/online/`
- `legacy/historical_poc/experiments_legacy/`

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
- offline calibrated analysis must reject anything other than real multi-rank
  `observed_online_native_ep` input; the current harness uses
  `observed_online_ws2_moe_layer_harness`
- offline calibrated analysis must explicitly reject
  `observed_online_ws2_route_partition`
- offline calibrated analysis must explicitly reject
  `observed_online_ws2_hidden_dispatch`
- online scheduler hint mode must reject `oracle_full_trace`
- all-to-all backend is not marked as matching-realized

## 5. Current Runnable Entry Points

Runnable now:

1. `experiments/offline/collect_router_trace.py`
2. `experiments/offline/analyze_cross_layer_prediction.py`
3. `legacy/historical_poc/experiments_legacy/exp_trace_replay.py`
4. `experiments/online/collect_native_ep_trace.py`

Present but expected to fail fast or reject non-qualifying input:

1. historical offline study entrypoints parked under `legacy/historical_poc/experiments_offline/`
2. historical online benchmark harness parked under `legacy/historical_poc/experiments_online/`

That is intentional. This file is retained only as a historical handoff note;
current canonical status is documented in `docs/handoff/pre_evaluation_handoff.md`.

## 6. What To Work On Next

Next real milestone is Phase 2:

1. real online native EP ownership and routing
2. ws2 count agreement for route metadata
3. native variable-size hidden dispatch
4. true two-GPU NCCL artifact for the current WS=2 harness
5. then real full-model EP replacement beyond the harness
6. then multi-node

Do not skip to scheduled P2P benchmark claims before native online EP is
implemented and validated.
