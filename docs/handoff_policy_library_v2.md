# RouterSense Scheduling Policy Library v2 Handoff

## Added Policies And References

- `islip_round_robin`: deterministic iSLIP-style phase-local matching adaptation. It is online executor compatible.
- `birkhoff_von_neumann_fluid`: offline-only fluid crossbar BvN reference with certificate validation.
- `exact_small_instance_reference`: offline-only exact discrete reference for tiny `discrete_bucket_phase_sync_wave` instances.

Existing v1 policies remain:

- `phase_barrier_fifo`
- `greedy_ready_set`
- `birkhoff_phase_local`
- `aurora_order_fixed`
- `fast_bvn_single_tier`
- `routersense_multiphase_lookahead:{p0_only,p0_p1,p0_p1_p2}`

## Capability Boundaries

- `islip_round_robin` is allowed in selected-layer online correctness through the frozen phase-local executor.
- `birkhoff_von_neumann_fluid` is offline-only and has `online_executor_compatible=false`.
- `exact_small_instance_reference` is offline-only and has `online_executor_compatible=false`.
- RouterSense multiphase lookahead remains offline-only until a future `multiphase_pending_window` runtime capability exists.

## Reference Models

- `offline_fluid_crossbar`: used by `birkhoff_von_neumann_fluid`; not runtime-latency comparable.
- `discrete_bucket_phase_sync_wave`: used by online phase-local policies and exact small-instance reference.

## Exact Reference Bound

`exact_small_instance_reference` supports:

- `rank_count <= 4`
- `bucket_task_count <= 12`

Above the bound it returns `solver_status=unsupported_scale` and `certified_optimal=false`.

## Validation Evidence

The v2 validator covers:

- logical flow coverage
- per-wave matching legality
- amount conservation
- BvN certificate validation
- exact-reference comparison
- capability consistency for online/offline policy resolution

## Non-Claims

This release does not claim GPU runtime speedup, full Aurora reproduction, full FAST reproduction, online RouterSense multiphase execution, or P2 predictor validation.
