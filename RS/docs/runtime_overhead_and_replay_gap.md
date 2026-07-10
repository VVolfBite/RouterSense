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
  - communication proxy is worse
  - scheduling/control overhead is also worse
- Top overhead sources are dispatch/combine hooks, `predict_next_dispatch`, and window-state recording.

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
5. Re-measure communication proxy before changing algorithmic weights.

## Non-Claims

- Do not use Run C as performance evidence; it is a bridge probe.
- Do not call execution-window U replay an online upper bound.
- Do not claim async_release performance until real collectives are implemented and validated.
