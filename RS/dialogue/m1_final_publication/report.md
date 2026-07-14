M1 runtime lifecycle closure status

Implemented on `convergence/m1-final-publication` after `d1a6f61`:

- control process group ownership moved out of lifecycle lazy paths and into runtime attach ownership via `ControlGroupHandle`
- lifecycle publication now uses deterministic `PublicationSlot` registry and safe-point retry semantics
- terminal lane results now clean store/service state on every rank, even without a local ready candidate
- generation cleanup now cancels every older generation across service, lane, and store
- formal target publication payload now carries `WindowPlan` as the authoritative logical plan
- `TargetLayerPreparedJointPlan.validate()` now binds `window_plan.semantic_digest()` to the formal published digest
- formal 4-rank Gloo lifecycle gate added at `RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py`

Gate coverage now includes:

- all ranks ready
- first poll `NOT_READY`, second safe point `READY`
- remote planner failure
- cancelled generation
- slot/token mismatch
- internal published-plan digest mismatch
- subgroup `(2, 3)` with root global rank `2`
- older-generation cleanup before generation `3`
- stale inflight replacement publishing only the newest version

Late-suffix status:

- formal gate evidence records `late_suffix_call_count = 0`
- formal gate evidence records `late_suffix_provider_present = false`
- formal gate evidence records `formal_target_commit_after_miss = false`

Executed commands:

- `python -m compileall RS/src RS/tests RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py`
- `PYTHONPATH='RS/src;RS;.' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q RS/tests/contract/megatron_ep/test_communication_lane.py RS/tests/contract/megatron_ep/test_lifecycle_prediction_adapter.py RS/tests/contract/megatron_ep/test_target_plan_contracts.py RS/tests/contract/megatron_ep/test_target_plan_reconciliation.py RS/tests/contract/megatron_ep/test_target_plan_store.py RS/tests/contract/megatron_ep/test_target_planner_service.py RS/tests/test_architecture_dependencies.py`
- `PYTHONPATH='RS/src;RS;.' python RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py`

Observed results:

- compileall passed
- targeted M1 suite passed: `54 passed`
- formal 4-rank lifecycle gate passed on CPU/Gloo

Current status for this branch: `M1_RUNTIME_LIFECYCLE_READY`
