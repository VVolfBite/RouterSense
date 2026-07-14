# M1 Runtime Report

## Final status

`M1_RUNTIME_LIFECYCLE_READY`

This pass converged the formal runtime entry, lifecycle event path, target preparation worker ownership, and target-plan store state transitions without changing executor/materializer/NCCL data-path behavior.

## What changed

1. Formal attach now returns a single `RuntimeHandle` with idempotent `close()` / `detach()` and restore callbacks for dispatcher wrappers, model forward hooks, expert attribution hooks, and Megatron monkeypatches.
2. Formal wrapped hooks now emit only standard runtime events:
   - `ForwardBeginEvent`
   - `DispatchReadyEvent`
   - `DispatchCompleteEvent`
   - `CombineReadyEvent`
   - `CombineCompleteEvent`
   - `ForwardEndEvent`
3. `RouterSenseInjectionRuntime.handle(...)` is now the formal lifecycle event entrypoint for the wrapped model path.
4. Formal dispatch/combine wrappers no longer re-enter legacy `on_dispatch(...)` / `on_combine(...)`.
5. `lifecycle.py` no longer imports `runtime.py`; policy/config resolution moved to `config.py`.
6. `TargetLayerPlannerService.submit(...)` is non-blocking and returns explicit submit statuses.
7. Preparation worker no longer publishes target plans directly. Worker builds ready results only; publish/agreement happens on the owning thread via `publish_ready_plan(...)`.
8. Worker task failure is isolated to the task; subsequent tasks still execute.
9. `TargetPlanStore` now has explicit logical/execution lifecycle states instead of immediate `consume_once(...)` on the prepared fast path.
10. Prepared target-plan execution now uses `claim -> bind -> start_execution -> complete/fail`.
11. Beginning or ending a forward cancels queued/ready stale target-layer tasks for that generation.
12. Sync current-window planning and target preplanning now meet at the same formal `PlanningRequest` / `WindowPlan` boundary.

## Runtime owner model

- Formal hook owner: `attach_formal_online_runtime(...)`
- Formal event owner: `RouterSenseInjectionRuntime.handle(...)`
- Target preparation owner: `TargetLayerPlannerService`
- Target-plan lifecycle owner: `TargetPlanStore`
- Prepared target-plan publication owner: lifecycle/main thread pump

## Remaining compatibility surface

Retained deliberately and documented:

- `attach_dispatch_observer(...)`
- lifecycle `on_dispatch(...)` / `on_combine(...)`
- `TargetPlanStore.consume_once(...)`
- `TargetPlanStore.claim_for_reconciliation(...)`
- `TargetPlanStore.close_key_if_unclaimed(...)`

These are compatibility-only and are not used by the formal wrapped runtime path covered by the M1 tests.

## Late suffix

Late suffix source code still exists, but the formal profile disables it:

- policy capabilities force `supports_late_suffix_splice=False`
- formal async target-plan path clears `transport_adapter.late_suffix_provider`
- prepared target-plan fast path now uses explicit target-plan state transitions instead of provisional-then-late execution ownership

## Tests and evidence

Executed:

- `python -m compileall RS/src RS/tests`
- targeted runtime/path tests
- combined runtime suite: `93 passed`
- distributed Gloo target-plan publish gate: `passed`, `world_size=4`

Evidence recorded under `RS/dialogue/m1_runtime/evidence/` covers:

- hook/event counts
- predictor/planner call counts
- non-blocking submit
- generation cancellation
- worker recovery
- worker shutdown
- target-plan transitions
- sync/preplanned formal parity
- detach/restore
- architecture gates
- 4-rank Gloo publish order

## Prompt-external issues found and addressed

1. Formal wrapped root model did not emit forward begin/end events through the same entrypoint; fixed by attaching root forward hooks.
2. Worker-built target plans were still implicitly coupled to immediate publication; fixed by introducing a ready-publication drain and explicit publish step.
3. Prepared target-plan path still looked successful before real execution completion; fixed by using explicit store state transitions.
4. Gloo gate script was still written for direct worker publication and narrower rank coverage; updated to drain ready publications and validate 4-rank agreement.
