# 2026-07-11 Pre-GPU Runtime Gate Progress

Current base:

- commit start: `b954987b0daea42f088de4665566bf639ccfe83c`
- environment: CPU-only in this session
- GPU visibility: `torch.cuda.is_available() == False`, `device_count == 0`

This round only advanced code and CPU contracts. No GPU run was attempted.

## Landed

### P2 runtime lifecycle

- `PredictedTrafficMatrix` now carries `valid` / `error`.
- Added `ActiveNextDispatchPrediction`.
- `RouterSenseInjectionRuntime._record_prediction_for_dispatch()` now stores an active next-layer prediction immediately after P0 observation.
- `CalibratedArtifactP2HintProvider` now prefers active prediction state when the requested layer matches the prediction target.
- Active prediction metadata records:
  - `prediction_created_stage`
  - `prediction_first_consumed_stage`
  - `consumer_layer`
  - `consumer_phase`
  - `consumed_before_p1`
- `extract_prepared_plan_priority()` now maps `PreparedWindowPlan.forecast_matrix` directly into next-layer P0 advisory priorities, while still preserving the legacy logical-P2 compatibility path.

### Predictor validation

- `zero_hint`, `copy_current_dispatch`, and `history_ema` now validate matrix shape against `world_size`.
- Shape mismatch no longer silently truncates rows/cols.

### Explain / safe-U correctness

- `SafeJointPolicy.evaluate_components()` now accepts prebuilt raw-U / paired-B plans.
- `policy_explain.py` reuses those plans rather than rebuilding them inside the explain path.
- This removes one source of explain/runtime drift for safe-U diagnostics.

### Async-release CPU contracts

- `gather_and_validate_async_release_schedule()` now performs header-first tensor agreement with variable payload length support.
- Agreement validation now checks `schema_version`.
- `AsyncReleaseP2PExecutor` now supports `local_rank` filtering:
  - local src rank emits send only
  - local dst rank emits recv only
  - unrelated ranks emit no op

### Runtime host/runtime-model cleanup

- Added `RuntimeHostFeasibilityProjector`.
- Host projection now explicitly assumes rank-level phase completion only:
  - no per-bucket compute overlap
  - no per-expert compute-complete event
  - P1 release waits for full local P0 completion
- Added `GlobalJointPlanWire` agreement and split validation:
  - global abstract plan must agree
  - local executable schedules may differ in length
  - pairwise send/recv contracts are validated separately
- `AsyncReleaseP2PExecutor` now distinguishes:
  - `global_rank`
  - `local_rank`
  - `ep_group_ranks`
- Added minimal `begin_forward()` / `end_forward()` lifecycle isolation so stale active prediction state does not silently leak across forwards.

## CPU Gate

Passed:

- `tests/contract/test_policy_explain.py`
- `tests/contract/test_routersense_joint_bridge.py`
- `tests/contract/megatron_ep/test_async_release_agreement.py`
- `tests/contract/megatron_ep/test_async_release_p2p_executor.py`
- `tests/test_runtime_fast_path_guards.py`
- `tests/test_architecture_dependencies.py`

Aggregate result:

- targeted tests passed: `30`
- `git diff --check`: passed

## Not done in this session

- GPU `Run B2 / C2 / A2`
- AR1 real collective validation
- corrected transport timing collection on GPU
- final common-core scheduler replacement
- final edge-aware P2 candidate evaluation

## Current status

- `gpu_not_run=true`
- `faithful_fate_not_validated=true`
- `async_release_real_collectives_not_validated=true`
- `expert_trace_collection_ready`: code path improved, not validated in this session due no visible GPUs
- `runtime_host_projection_validated=true`
- `local_schedules_allowed_to_differ=true`
- `global_rank_namespace_validated=true`
- `dedicated_p2p_group_initialized=false`
- `nccl_tag_dependency=false`
- `per_peer_sequence_validated=false`
- `forward_epoch_isolation_validated=true`
- `birkhoff_same_executor_control_ready=false`
- `per_bucket_compute_overlap_claimed=false`
