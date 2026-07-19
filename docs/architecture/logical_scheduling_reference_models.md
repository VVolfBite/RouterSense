# Logical Scheduling Reference Models

RouterSense scheduling artifacts use multiple logical models. They must not be mixed into a single runtime-performance ranking.

## Online Phase-Local Executor Model

Model: `discrete_bucket_phase_sync_wave`

- Unit of work: indivisible bucket task.
- Wave constraint: each rank has at most one outgoing and at most one incoming task.
- Wave duration in offline analysis: max byte count among tasks in the wave.
- Runtime compatibility: yes, for policies whose capability sets `supports_online_phase_local_execution=true`.

Policies in this model include:

- `phase_barrier_fifo`
- `greedy_ready_set`
- `islip_round_robin`
- `birkhoff_phase_local`
- `aurora_order_fixed`
- `fast_bvn_single_tier`

## Offline Fluid Crossbar Reference

Model: `offline_fluid_crossbar`

- Policy/reference: `birkhoff_von_neumann_fluid`.
- Traffic is divisible by service quantum.
- Fluid horizon is `max(max source load, max destination load)`.
- Dummy/idle edges may appear inside the certificate but never become real transport flows.
- Runtime compatibility: no.
- Runtime latency comparability: no.

This is the formal BvN fluid certificate. It is not the same as `birkhoff_phase_local` or `fast_bvn_single_tier`.

## Offline Exact Discrete Reference

Model: `discrete_bucket_phase_sync_wave`

- Policy/reference: `exact_small_instance_reference`.
- Certified only for `rank_count <= 4` and `bucket_task_count <= 12`.
- Objective is total logical makespan, the sum of wave durations.
- Above the scale limit it returns `solver_status=unsupported_scale`.

## RouterSense Multiphase Lookahead

Model: offline logical multiphase ready-set scheduling.

- Policies: `routersense_multiphase_lookahead:p0_only`, `p0_p1`, `p0_p1_p2`.
- P1 is route-derived blocked dependency, not prediction.
- P2 is forecast-only pressure and never executable transport.
- Online joint execution is not implemented in this round.

## Non-Claims

This policy library does not provide GPU latency conclusions. Logical wave counts, fluid horizons, and exact small-instance objectives are correctness and scheduling-quality evidence, not runtime speedup claims.
