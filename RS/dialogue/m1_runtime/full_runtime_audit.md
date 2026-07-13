# M1 Runtime Audit

Starting branch: `convergence/m1-runtime-lifecycle`  
Starting SHA: `8cb7212`

This audit was produced before M1 code changes. It reflects the current runtime/lifecycle/state/preparation surface as found in the repository.

## Scope Classification

`RS/src/rs/runtime/online/megatron_ep/host.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/runtime.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/contracts.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/lifecycle.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/state/runtime_state.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/state/window_runtime_state.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/target_planning/contracts.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/target_planning/predictor.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/target_planning/planner_service.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/target_planning/store.py` — `REFACTOR`  
`RS/src/rs/runtime/online/megatron_ep/target_planning/reconcile.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/pending_window/adapter.py` — `COMPATIBILITY`  
`RS/src/rs/runtime/online/megatron_ep/pending_window/policy_adapter.py` — `COMPATIBILITY`  
`RS/src/rs/runtime/online/megatron_ep/pending_window/release_engine.py` — `MERGE`  
`RS/src/rs/runtime/online/megatron_ep/pending_window/shadow.py` — `COMPATIBILITY`  
`RS/src/rs/runtime/online/megatron_ep/pending_window/window_state.py` — `COMPATIBILITY`  
`RS/src/rs/runtime/online/megatron_ep/planning/window_release_service.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/planning/window_shadow_service.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/phase/contracts.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/phase/context_builder.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/phase/layout_join.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/phase/validation.py` — `KEEP`  
`RS/src/rs/runtime/online/megatron_ep/async_release/*` — `OUT_OF_SCOPE` for internal algorithm changes, `IN_SCOPE` for runtime entry ownership boundaries  
`RS/src/rs/runtime/offline/*` — `OUT_OF_SCOPE` except direct M1 runtime caller checks  
`RS/tests/contract/megatron_ep/test_lifecycle_prediction_adapter.py` — `REFACTOR`  
`RS/tests/contract/megatron_ep/test_target_planner_service.py` — `REFACTOR`  
`RS/tests/contract/megatron_ep/test_target_plan_store.py` — `REFACTOR`  
`RS/tests/contract/megatron_ep/test_target_plan_reconciliation.py` — `KEEP`  
`RS/tests/contract/megatron_ep/test_runtime_host_projection.py` — `COMPATIBILITY`  
`RS/tests/contract/megatron_ep/test_no_legacy_runtime_imports.py` — `REFACTOR`

## 1. Model attach and wrapper graph

Current public attach path is still split.

1. `attach_formal_online_runtime(...)` in `host.py`
2. builds `RouterSenseInjectionConfig`
3. delegates to `attach_dispatch_facade(...)`
4. `attach_dispatch_facade(...)`
5. optionally monkeypatches Megatron `all_to_all`
6. instantiates `RouterSenseInjectionRuntime`
7. scans model modules for `token_dispatcher`
8. wraps `dispatcher.token_dispatch`
9. for selected layer also wraps `dispatcher.token_combine`
10. wrapper calls lifecycle methods:
   - `before_prediction_source_dispatch(...)`
   - `before_token_dispatch(...)`
   - `mark_token_dispatch_committed(...)`
   - native dispatch
   - `capture_phase_transport_output(...)`
   - `after_token_dispatch(...)`
   - `on_dispatch(...)`
   - same pattern for combine

Parallel legacy wrapper still exists in the same file:

1. `attach_dispatch_observer(...)`
2. wraps `dispatch_preprocess`
3. wraps `token_dispatch`
4. wraps `token_combine`
5. wraps `combine_postprocess`
6. writes observer records directly

Observed issue:

- formal runtime wrapper and observer wrapper are both first-class attach surfaces
- wrapping state is tracked by two flags: `_routersense_wrapped` and `_routersense_facade_wrapped`
- there is no single `ModelRuntimeAdapter` owner yet
- detach/restore is not centralized in a handle object

## 2. dispatch/combine full event order

Current formal dispatch chain in `host.py`:

1. wrapper enter
2. `runtime.before_token_dispatch(...)`
3. `runtime.mark_token_dispatch_committed(...)`
4. native dispatch
5. `runtime.capture_phase_transport_output(...)`
6. `runtime.after_token_dispatch(...)`
7. `runtime.on_dispatch(...)`

Current formal combine chain:

1. wrapper enter
2. `runtime.before_token_combine(...)`
3. native combine
4. `runtime.capture_phase_transport_output(...)`
5. `runtime.after_token_combine(...)`
6. `runtime.on_combine(...)`

Current runtime therefore still has both:

- before/after lifecycle hooks
- extra `on_dispatch/on_combine` shadow-style callbacks

That is the double-event surface M1 needs to collapse.

## 3. begin_forward/end_forward call graph

`lifecycle.begin_forward(...)`

1. increments or sets `_forward_epoch`
2. clears `_current_plan_build_keys`
3. clears selected/expert timing maps
4. clears `_pending_p0/_pending_p1`
5. clears active transport
6. clears active prediction and prediction consumption records
7. clears consumed plan digests and prepared priority cache
8. clears global joint plan wire/agreement/window plan
9. writes `forward_start_ns`
10. clears reconciled target-plan key set
11. calls `target_plan_store.cleanup_epoch(...)` if store exists

`lifecycle.end_forward(...)`

1. records `forward_end_ns`
2. clears `_pending_p0/_pending_p1`
3. clears active transport
4. clears active prediction and prediction consumption records
5. clears prepared priority cache and global joint plan wire/agreement/window plan
6. calls `target_plan_store.cleanup_epoch(...)`
7. returns summary dict

Observed issue:

- generation/cancellation exists only implicitly via `_forward_epoch`
- no typed preparation task generation or cancellation token exists
- cleanup only acts on store tombstoning, not a worker queue/inflight task graph

## 4. Current Plan generate/save/consume graph

Current selected-layer planning path in lifecycle:

1. `before_token_dispatch(...)`
2. build phase context from dispatcher
3. capture pretransport observation
4. gather actual P0 row matrix
5. optionally record prediction for next layer
6. build P2 hint
7. build runtime observation
8. record window state and release state
9. if target-prepared plan exists:
   - may reconcile target plan
   - may activate transport from prepared/target plan
10. otherwise:
   - run `run_phase_plan_agreement(...)` or pending-window adapter
11. activate transport

Current plan ownership is split across:

- `_runtime_state.prepared_plan`
- `_runtime_state.global_joint_window_plan`
- `_runtime_state.global_joint_plan_wire`
- `_window_states`
- `TargetPlanStore`
- pending-window adapter cache

There is no single typed `CurrentPlanState` owner yet.

## 5. Target Plan submit/predict/plan/publish/claim/execute graph

Current path:

1. `lifecycle._record_prediction_for_dispatch(...)`
2. if target-layer preplanning enabled and next layer selected:
3. `lifecycle._ensure_target_planner_runtime()`
4. `target_planner_service.enqueue(TargetLayerPlanningRequest)`
5. service thread `_worker()`
6. `_build_target_plan(...)`
7. `SharedTwoHorizonPredictor.predict_two_horizon(...)`
8. build `PlanningRequest`
9. build raw planner
10. plan raw U
11. maybe build paired B
12. maybe select with `PlannerSelector`
13. publish agreement payload
14. `TargetPlanStore.put(...)`

Consume path:

1. selected-layer dispatch later calls `_record_plan_arrival(...)` / target-plan lookup path
2. store `claim_for_reconciliation(...)`
3. `reconcile_once(...)`
4. may `consume_once(...)`
5. may proceed to provisional + late suffix splice path

Observed issues:

- worker owns queue and planning thread directly
- service uses blocking `queue.put(...)`
- no explicit submit result
- no stale replacement logic per `(forward,target-layer)`
- no structured cancellation before/after predict/plan/publish/store-write
- consume path still allows `late_suffix` mutation after provisional execution

## 6. worker start/shutdown/restart graph

Current worker owner: `TargetLayerPlannerService`

Start:

1. `TargetLayerPlannerService.start()`
2. if `_thread is not None`, return
3. create daemon thread
4. thread runs `_worker()`

Shutdown:

1. set `_stop`
2. `put_nowait(None)` sentinel
3. `join(timeout=5.0)`

Observed issues:

- close state is not explicit
- queue full on shutdown is silently ignored
- thread liveness after join is not verified
- service cannot distinguish task failure vs service failure
- first exception stops worker permanently
- no restart contract

## 7. all collective call sites

Observed in M1 surface:

`host.py`

- `_discover_all_ep_group_tuples()` → `dist.all_gather_object`
- dedicated group warmup → `dist.all_reduce`
- `stage_barrier()` → `dist.all_gather_object`
- `gather_rank_payloads()` → `dist.all_gather_object`
- optional `dist.new_group(...)`

`lifecycle.py`

- `_agree_target_plan_payload()` → `dist.all_gather` twice on CPU tensors
- actual P0 matrix gather paths → `dist.all_gather`
- payload agreement helpers → `dist.all_gather`
- late suffix helper path → uses group agreement helpers indirectly
- phase agreement path delegates to `run_phase_plan_agreement(...)`

Observed issue:

- lifecycle directly owns target-plan agreement collective
- background planner service calls `agreement_fn`, so the worker can indirectly trigger collectives
- this violates the intended M1 ownership boundary

## 8. all thread call sites

Observed thread creation:

- `TargetLayerPlannerService.start()` → `threading.Thread(..., daemon=True)`

Observed thread-safe state holders:

- `TargetPlanStore` uses `threading.RLock`

No typed preparation scheduler abstraction exists yet. Background execution is direct queue + thread.

## 9. all RuntimeState writers

Major writers:

- `RouterSenseInjectionRuntime.__post_init__`
- `begin_forward()`
- `end_forward()`
- `before_token_dispatch()`
- `after_token_dispatch()`
- `before_token_combine()`
- `after_token_combine()`
- `_record_prediction_for_dispatch()`
- `_record_window_state()`
- `_record_release_update()`
- `_record_pending_window_driver()`
- `_record_prepared_phase_plan_shadow()`
- `_record_compilation_metrics()` and related compile paths
- target-plan arrival / reconciliation / consume paths

Observed issue:

- `PreparedWindowRuntimeState` remains a large hybrid bag with many control booleans and digests
- metrics and control flags are mixed in one object
- `extras` still exists as compatibility escape hatch

## 10. all extras control fields

Current state contract still exposes `extras: dict[str, Any]`.

When invariant mode is diagnostic:

- unknown reads fall back to `extras`
- unknown writes are accepted into `extras`
- unknown pops/removes are accepted

This is still a control-plane leak. M1 needs typed state fields for runtime control semantics and must leave `extras` only for non-control compatibility metadata.

## 11. all fallback branches

Observed fallback families:

- layer role `none` returns without scheduling
- prediction-source layer bypass
- native passthrough when no policy applies
- pending-window fast-path fallback to phase policy
- copy-current-dispatch prediction fallback
- prepared target plan fallback to current sync planning
- provisional execution fallback before late suffix
- shadow-only passthrough in `on_dispatch/on_combine`

Critical overlap:

- old runtime-safe policy pairing still exists in lifecycle
- `PlannerSelector.COMPARE` exists in target planner service
- responsibility is still split between legacy runtime safe logic and formal planner selection

## 12. all shadow/pending/late suffix branches

Observed branches:

- `pending_window/adapter.py`
- `pending_window/policy_adapter.py`
- `pending_window/shadow.py`
- `planning/window_shadow_service.py`
- lifecycle `maybe_build_window_shadow(...)`
- lifecycle `_record_prepared_phase_plan_shadow(...)`
- lifecycle `_late_suffix_provider(...)`
- lifecycle `_residualize_suffix_tasks(...)`
- lifecycle `_agree_late_suffix(...)`
- lifecycle provisional then late suffix consume path

Observed issue:

- formal profile can still enter late suffix code path
- pending-window adapter is still a first-class execution path
- shadow analysis and formal scheduling remain adjacent in the lifecycle

## 13. all old Planning API calls

Confirmed legacy planning leakage:

`lifecycle.py`

- imports `rs.planning._legacy_runtime.build_runtime_policy`
- imports `rs.planning._legacy_runtime.build_runtime_request_from_problem`
- imports `rs.planning.runtime_compat.resolve_phase_policy`
- imports `rs.scheduling.contracts.PreparedWindowPlan`
- imports `rs.scheduling.bucketizer.*`
- imports `rs.scheduling.validation.stable_hash`

`runtime.py`

- imports `rs.planning.runtime_compat.ResolvedAlgorithmId`
- imports `resolve_algorithm_id`
- imports `resolve_phase_policy`

`pending_window/policy_adapter.py`

- imports `rs.planning.runtime_compat.resolve_phase_policy`
- imports `supported_phase_policies`

Observed issue:

- lifecycle current-layer sync path still builds legacy runtime request/policy directly
- runtime public API still exposes legacy phase-policy resolution surface

## 14. all monkeypatch and restore paths

Observed monkeypatch path:

- `host.attach_dispatch_facade(...)`
- imports `megatron.core.transformer.moe.token_dispatcher`
- saves `original_all_to_all`
- replaces module-level `all_to_all` with wrapped transport adapter
- stores `runtime.original_all_to_all`

Observed missing restoration owner:

- no dedicated runtime handle in `host.py`
- no audited detach path in this surface
- wrapper replacement and transport monkeypatch restoration are not centralized under one close/detach API

## 15. each test’s real coverage

`RS/tests/contract/megatron_ep/test_lifecycle_prediction_adapter.py`

- covers `_record_prediction_for_dispatch(...)`
- verifies typed compatibility object is serialized into runtime state shape
- does not cover full attach/hook/detach chain

`RS/tests/contract/megatron_ep/test_target_planner_service.py`

- covers `_build_target_plan(...)`
- covers paired-B build toggling by `safe_projection_mode`
- covers `_select_candidate_plans(...)`
- covers planner call counts in isolated service path
- does not cover lifecycle submit path or worker cancellation/shutdown

`RS/tests/contract/megatron_ep/test_target_plan_store.py`

- covers `put`, `peek`, `claim_for_reconciliation`, `consume_once`, `cancel`
- current semantics still assert terminal `CONSUMED`
- does not cover publish/claim/bind/start/complete/failed state machine required by M1

`RS/tests/contract/megatron_ep/test_target_plan_reconciliation.py`

- covers exact / repaired / rejected reconciliation

`RS/tests/contract/megatron_ep/test_runtime_host_projection.py`

- covers legacy host projection helper behavior
- this is compatibility coverage, not formal M1 runtime entry coverage

`RS/tests/contract/megatron_ep/test_no_legacy_runtime_imports.py`

- currently forbids only a small set of older runtime imports
- does not forbid `rs.planning.runtime_compat`
- does not forbid legacy planning helper imports inside runtime lifecycle

## Prompt-external findings

1. `host.py` still exports `attach_dispatch_observer(...)` alongside formal runtime attach. M1 needs one owner.
2. `runtime.py` lazily re-exports `RouterSenseInjectionRuntime` and still acts as a public mixed facade rather than a narrow adapter surface.
3. `PreparedWindowRuntimeState` remains a flat mutable bag with control booleans, digests, metrics and legacy compatibility fields intermixed.
4. `TargetPlanStore` current terminal model is `AVAILABLE/CLAIMED/CONSUMED/REJECTED/EXPIRED/CANCELLED`; it does not model `LOGICAL_READY/BOUND/EXECUTING/COMPLETED/FAILED`.
5. `TargetLayerPlannerService` still runs collectives indirectly through `agreement_fn` from the background worker.
6. Current target-plan enqueue path is still coupled to `_record_prediction_for_dispatch(...)`, not a dedicated non-blocking preparation scheduler.
7. `begin_forward/end_forward` cleanup is store-only; there is no inflight task invalidation primitive.
8. Late suffix splice remains reachable from the formal runtime path.

## Immediate M1 implementation targets

1. Freeze this audit and preserve it as pre-change evidence.
2. Introduce a single model runtime adapter and runtime handle.
3. Collapse dispatch/combine event chain to one formal lifecycle entry path.
4. Break `runtime.py` / `lifecycle.py` public-type coupling.
5. Replace direct planner-service queueing with typed non-blocking preparation scheduling.
6. Move target-plan publish agreement out of background worker ownership.
7. Introduce typed runtime state ownership for current/target/preparation state.
8. Retire formal late-suffix path from deployable profile.
9. Remove current-layer direct legacy planning construction from formal runtime path.
