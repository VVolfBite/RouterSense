# Runtime Overhead And Replay Gap

The negative real 4GPU `routersense_joint_priority_phase_sync` result does not invalidate offline joint-scheduling opportunity. It identifies the layer where the current implementation fails to convert that opportunity.

## Layered Evidence Model

Layer 0: O-local / O-joint exact opportunity

- Small exact fixture shows joint oracle can beat local oracle.
- This validates the theoretical opportunity, not the online implementation.

Layer 1: execution-window U policy opportunity

- `U_gated_maxweight_matching` and `U_barrier_criticality_global_matching` beat `B_birkhoff_wave` by about 8% in oracle-information execution-window replay.
- This uses future/actual P2 information and is not online-eligible.

Layer 2: safe-U replay policy

- `RS_safe_barrier_criticality` and `RS_safe_gated_greedy` are the only useful safe-U mainline candidates.
- They show around 11% paired-B replay gains.
- Other safe-U families are diagnostic or parity-only at present.

Layer 3: runtime-lookahead / phase-sync-compatible adapter

- Current `routersense_multiphase_lookahead:p0_p1_p2` loses to `birkhoff_phase_local`.
- Current `routersense_joint_priority_phase_sync` loses even more in replay.
- This is the first major conversion gap.

Layer 4: online phase-sync runtime with control overhead

- Real 4GPU run shows the bridge enters runtime, but is slower than Birkhoff.
- The slowdown is mixed:
  - dispatch/combine hook-path duration is worse
  - scheduling/control overhead is also worse
- Top overhead sources are dispatch/combine hooks, `predict_next_dispatch`, and window-state recording.
- Current trustworthy runtime comparisons are:
  - `total_forward_us`
  - inclusive dispatch/combine hook path duration
  - named control substages inside those hooks
  - unattributed hook time after subtracting confirmed nested substages
- Current `rank*_transport_execution.jsonl` artifacts do not expose reliable cross-rank transport timestamps, so true NCCL transport makespan is still unavailable.

Layer 5: future async_release runtime

- Current async_release is a fail-closed framework and simulator.
- Real collectives are not implemented or validated.
- It should not be used as performance evidence yet.

## Current Bottleneck

The primary bottleneck is not that joint scheduling space disappeared. The bottleneck is:

```text
execution-window / safe-U opportunity
  -> current phase_sync adapter
  -> runtime hook/control overhead
```

Both conversion steps currently lose.

## Runtime Fix Order

1. Reduce or cache `predict_next_dispatch`.
2. Remove `prepared_phase_plan_shadow` from the hot path.
3. Defer or compact `record_window_state`.
4. Compact `store_prepared_plan`.
5. Re-measure actual transport timing after control-cost reduction once timestamped transport artifacts exist.

Current online bridge semantic fixes already landed:

- logical `P2(L)` prepared priority maps to runtime `P0(L+1)`
- stale logical `P0/P1` prepared edges are excluded from next-layer priority
- prepared priority is now an advisory/tie-break signal instead of an unconditional override

## Non-Claims

- Do not use Run C as performance evidence; it is a bridge probe.
- Do not call execution-window U replay an online upper bound.
- Do not claim async_release performance until real collectives are implemented and validated.
- Do not label dispatch/combine hook time as transport communication makespan.
