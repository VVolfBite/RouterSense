M1 runtime lifecycle closure status

Branch: `convergence/m1-final-publication`
Starting SHA for this closure pass: `1ff916f2ad317eb21d92a0e25290af89f0fddce1`

Implemented in this pass:

- added `ControlGroupRegistry` in runtime host bootstrap so all world ranks create all EP control groups in the same deterministic order before lifecycle publication begins
- updated `ControlGroupHandle.close()` so registry-owned groups are not double-destroyed
- converted the formal Gloo lifecycle gate at `RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py` to drive publication through `runtime.handle(...)` events instead of private submit/pump helpers for the main lifecycle path
- added dynamic spies for `_agree_late_suffix`, `_late_suffix_provider`, `consume_once(execution_origin="provisional_then_late_suffix")`, and post-target-commit publication
- fixed `TargetPlanStore` generation cleanup and shutdown so older generations and unfinished executing entries close deterministically instead of hitting illegal `EXECUTING -> CANCELLED` transitions
- exercised subgroup control-lane publication with runtime bootstrap-created Gloo groups and root global rank `2`
- exercised mismatch and stale-replacement scenarios against the actual publication path

Formal gate coverage now includes:

- all-ready publication
- first safe point `NOT_READY`, second safe point `READY`
- remote planner failure
- cancelled generation
- slot/token mismatch
- canonical plan digest mismatch
- subgroup `(2, 3)` with root global rank `2`
- generation cleanup before generation `3`
- stale inflight replacement publishing only the newest version

Dynamic no-late-suffix evidence:

- `late_suffix_call_count = 0`
- `late_suffix_provider_present = false`
- `formal_target_commit_after_miss = false`
- `post_commit_publication_count = 0`

Executed commands:

- `python -m compileall RS/src RS/tests RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py`
- `PYTHONPATH='RS/src;RS;.' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q RS/tests/contract/megatron_ep/test_communication_lane.py RS/tests/contract/megatron_ep/test_lifecycle_prediction_adapter.py RS/tests/contract/megatron_ep/test_target_plan_contracts.py RS/tests/contract/megatron_ep/test_target_plan_reconciliation.py RS/tests/contract/megatron_ep/test_target_plan_store.py RS/tests/contract/megatron_ep/test_target_planner_service.py RS/tests/test_architecture_dependencies.py`
- `PYTHONPATH='RS/src;RS;.' python RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py`

Observed results:

- compileall passed
- targeted M1 suite passed: `54 passed`
- formal 4-rank lifecycle gate passed on CPU/Gloo

Current branch status: `M1_RUNTIME_LIFECYCLE_READY`
