# Multiphase Global Matching Study

## Scope

This study adds a new `U_` scheduler family for true multi-phase joint scheduling:

- `U_gated_greedy_maximal`
- `U_gated_maxweight_matching`
- `U_barrier_criticality_global_matching`
- `U_barrier_price_adaptive_matching`

The strong historical comparison baselines are kept as `D_` schedulers:

- `D_birkhoff`
- `D_barrier_aware_birkhoff`
- `D_cp_lpt`
- `D_lagrangian`
- `D_ibbr`

`D_` means the policy is still fundamentally built around phase-local ordering or phase-local Birkhoff style scheduling, even if the simulator later evaluates three phases together. `U_` means the scheduler explicitly competes all currently released flows from all released phases in one global ready set.

## Repository Semantics Audit

Current mainline `fast.py` and `oracle.py` already implement the following barrier semantics:

- full-duplex ports:
  - each GPU has one send port and one recv port
  - send and recv may overlap on the same GPU
  - two sends on the same GPU may not overlap
  - two recvs on the same GPU may not overlap
- local receive barrier:
  - phase `p+1` outbound from GPU `g` may start only after all phase-`p` inbound traffic to `g` has completed
  - for phase `0 -> 1`, `expert_compute_delay` is added

This is implemented in:

- `src/routesense/scheduler/fast.py::_schedule_phase_orders`
- `src/routesense/scheduler/oracle.py::pairwise_oracle`

No silent semantic change was made to those existing schedulers.

## Hard Phase-2 vs Soft Lookahead

Two explicit modes are supported for the new `U_` schedulers:

- `execution_window`
  - `M0`, `M1`, and `M2` are all real traffic
  - this matches the current offline three-phase oracle style
- `runtime_lookahead`
  - `M0` and `M1` are real
  - `M2` is prediction only
  - phase-2 traffic must not produce real schedule entries
  - prediction only affects weights

The audit layer checks that `runtime_lookahead` never emits a real phase-2 transfer.

## Fluid vs Atomic Semantics

This is the most important caveat.

### Existing `D_` family

Existing `D_` schedulers and the current CP-SAT oracle are **atomic** at the `(phase, src, dst)` chunk level:

- each non-zero matrix entry becomes one chunk
- chunk duration is `size`
- chunks are not split

### New `U_` family

The new `U_` schedulers are **fluid wave schedulers**:

- they operate on residual flow
- one ready-set matching is chosen per wave
- a wave serves a finite amount of volume on all matched edges
- an original flow may therefore be split across multiple waves

This is valid for the current simulator only under the implicit zero-setup-cost assumption:

- `d = beta * volume`
- no extra launch/setup cost per split

Therefore:

- `U_` vs `U_` is fair
- `D_` vs `D_` is fair
- `U_` vs current atomic oracle is **diagnostic only**
- a `U_` result beating the current atomic oracle does **not** mean the oracle is wrong; it means the semantic model differs

## Unified Audit / Replay

New module:

- `src/routesense/scheduler/multiphase_global.py`

Shared validator entry:

- `replay_and_audit_schedule(...)`

Audit fields:

- `scheduler_name`
- `mode`
- `prediction_used`
- `valid`
- `makespan`
- `planning_time_ms`
- `wave_count`
- `replay_makespan`
- `barrier_times`
- `send_busy_time`
- `recv_busy_time`
- `validation_errors`

Validator checks:

1. send-port non-overlap
2. recv-port non-overlap
3. full-duplex overlap is allowed
4. local receive barrier is respected
5. traffic conservation
6. no unexpected flow is served
7. replay makespan matches reported makespan
8. no phase-2 real entries in `runtime_lookahead`

## New `U_` Scheduler Scores

All `U_` schedulers build a global ready set over released flows.

Generic score structure:

`score = a * residual + b * barrier_urgency + c * age + d * prediction_bonus`

Where:

- `residual` keeps large unfinished traffic visible
- `barrier_urgency` approximates how much downstream outbound load is unlocked by completing ingress at the receiver
- `age` avoids starvation
- `prediction_bonus` is only active in `runtime_lookahead`

### `U_gated_greedy_maximal`

- low-overhead baseline
- sorts ready edges by score and builds a maximal matching greedily

### `U_gated_maxweight_matching`

- exact maximum-weight matching over the ready set
- one phase label per physical `(src, dst)` pair survives into the matching

### `U_barrier_criticality_global_matching`

- same exact matching primitive
- stronger weight on barrier unlock urgency

### `U_barrier_price_adaptive_matching`

- bounded adaptive prices `lambda[phase, gpu]`
- prices are updated from downstream pressure and remaining ingress
- this is a heuristic barrier-price method, not a formal Lagrangian relaxation proof

## Complexity

For dense `N x N` ready sets:

- exact matching wave: `O(N^3)` using Hungarian / assignment
- greedy maximal wave: `O(E log E)` with `E = O(N^2)`
- planner cost with bounded waves:
  - exact: `O(max_waves * N^3)`
  - greedy: `O(max_waves * N^2 log N)`

No online MILP, no online CP-SAT, and no `N!` matching enumeration is used.

## Minimal Validated Results

### Synthetic case: `unlock_hotspot`

Run:

```bash
cd /root/autodl-tmp/RouterSense/RS
PYTHONPATH=src python experiments/poc_line1/exp_multiphase_global_matching.py \
  --case unlock_hotspot \
  --mode execution_window \
  --output artifacts/poc_line1/multiphase_global_unlock_hotspot.json
```

Observed diagnostic result during implementation:

- `D_birkhoff`: makespan `41`
- `D_lagrangian`: makespan `41`
- `D_ibbr`: makespan `41`
- `U_gated_greedy_maximal`: makespan `29`
- `U_gated_maxweight_matching`: makespan `29`
- `U_barrier_criticality_global_matching`: makespan `29`
- `U_barrier_price_adaptive_matching`: makespan `29`
- current atomic oracle: `40` (`OPTIMAL`)

Interpretation:

- the new `U_` family is exploiting fluid wave splitting
- the current oracle is atomic and therefore not a directly comparable upper bound for `U_`
- this validates the mechanism but not yet the final apples-to-apples claim

### Synthetic case: `cross_phase_shift`

Observed diagnostic result during implementation:

- `D_birkhoff`: `34`
- `D_barrier_aware_birkhoff`: `34`
- `D_lagrangian`: `34`
- `D_ibbr`: `34`
- `U_gated_greedy_maximal`: `34`
- `U_gated_maxweight_matching`: `38`
- `U_barrier_criticality_global_matching`: `38`
- `U_barrier_price_adaptive_matching`: `38`

Interpretation:

- the new `U_` family does not dominate on every synthetic pattern
- the current score still has clear failure modes

## Current Failure Modes

1. The `U_` family is fluid while the current baseline/oracle family is atomic.
2. Some synthetic cases show exact global matching over-prioritizing unlock value and hurting throughput.
3. Real-trace candidate comparison on CPU-only constrained execution was not completed in this round; one small `N=8` slice attempt was killed by the environment.

## What Was Verified

- unit tests for:
  - full-duplex replay
  - runtime-lookahead phase-2 suppression
  - deterministic zero-confidence behavior
- full repository regression:

```bash
cd /root/autodl-tmp/RouterSense/RS
pytest -q
```

Result during implementation:

- `46 passed, 1 warning`

## What Still Needs the Next Round

The single most valuable next step is:

> build an atomic-chunk version of the global ready-set `U_` scheduler, or a fluid oracle, so that `U_` and oracle share the same semantic model.

Until that is done, current `U_` wins are mechanism-valid but not yet final paper numbers.
