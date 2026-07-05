# RouterSense Scheduling Policy Contract

## Public Interface

All formal scheduling policies live under `rs.scheduling` and implement:

```python
class SchedulingPolicy(Protocol):
    policy_name: str
    policy_version: str
    capabilities: PolicyCapabilities

    def build_logical_plan(
        self,
        problem: MultiPhaseSchedulingProblem,
    ) -> LogicalSchedulePlan:
        ...
```

Phase-local policies that can drive the frozen online executor additionally implement:

```python
def build_plan(
    *,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
) -> PhaseExecutionPlan:
    ...
```

## Information Semantics

- `p0_dispatch`: current executable dispatch demand.
- `p1_return`: route-derived blocked return demand. It is not prediction.
- `p2_next_dispatch_forecast`: forecast-only pressure. It never becomes current executable transport in this round.

## Policy Matrix

### Controls

- `native_passthrough`
  - Offline: unsupported for logical scheduling.
  - Online: use native observe / disabled injection path.

- `phase_barrier_fifo`
  - Fixed phase barrier.
  - Deterministic per-phase FIFO edge order.
  - Online executable.

### Generic baseline

- `greedy_ready_set`
  - Offline: ready-flow greedy multiphase scheduler.
  - Online: phase-local greedy ordering adaptation.
  - Does not use blocked P1 or P2.

- `islip_round_robin`
  - Deterministic iSLIP-style phase-local matching adaptation.
  - Uses stable rotating input/output pointers derived from plan identity.
  - Online executable through the frozen `phase_sync_wave` executor.
  - Not a full switch iSLIP system reproduction.

### Fixed-placement adaptations

- `birkhoff_phase_local`
  - Single-phase matching decomposition adaptation.
  - No placement/routing changes.
  - Online executable.
  - This is not the formal fluid-optimal BvN reference.

- `aurora_order_fixed`
  - Fixed-placement, phase-local communication-ordering adaptation.
  - Uses endpoint pressure only.
  - No P1 dependency, no P2.
  - Online executable.

- `fast_bvn_single_tier`
  - Single-tier, fixed-placement matching-decomposition adaptation.
  - No topology routing layer, no placement optimization.
  - Online executable.
  - This uses residual-demand maximum-weight matching, but it is not the formal fluid-optimal BvN certificate.

### Offline References

- `birkhoff_von_neumann_fluid`
  - Offline-only fluid crossbar reference.
  - Emits a certificate with port-load lower bound, fluid horizon, coverage, and matching checks.
  - Not online executor compatible and not runtime-latency comparable.

- `exact_small_instance_reference`
  - Offline-only certified reference for `discrete_bucket_phase_sync_wave`.
  - Supports `rank_count <= 4` and `bucket_task_count <= 12`.
  - Fails closed with `unsupported_scale` above that bound.

### RouterSense

- `routersense_multiphase_lookahead:p0_only`
- `routersense_multiphase_lookahead:p0_p1`
- `routersense_multiphase_lookahead:p0_p1_p2`

These are offline logical schedulers plus prepared-plan emitters.

They are not online joint executors in the frozen runtime. Online correctness must fail closed until a future `multiphase_pending_window` capability exists.

## Diagnostics

Every logical plan emits `PolicyDiagnostics` and per-wave `WaveDiagnostics` with:

- selected flow ids
- selected edges
- matching weight
- remaining bytes before/after
- ready/blocked counts
- priority components
- selection reason

RouterSense lookahead additionally emits:

- `prepared_window_plan_*.json`
- `forecast_comparison_*.json`

## Explicit Non-Goals

This round does not implement:

- full Aurora system reproduction
- full FAST system reproduction
- online RouterSense joint execution
- P2 calibrated predictor artifact
- performance claims
