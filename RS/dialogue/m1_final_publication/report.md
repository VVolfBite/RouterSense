M1 runtime lifecycle closure

Branch: `convergence/m1-final-publication`
Starting SHA for this closure pass: `02e74e856126079580ba71f1124dc4ca9e4d1eee`

Current status: `M1_RUNTIME_LIFECYCLE_READY`

Closed in this pass:

- `TargetLayerPlannerService`, `TargetPlanStore`, and `GlooControlCommunicationLane` now maintain persistent generation floors, so stale generations are rejected even if no earlier task was ever submitted.
- `cleanup_before_generation()` now rejects later direct token registration and direct publish paths for old generations instead of only cleaning currently visible tasks.
- formal lifecycle gate now has centralized pass/fail validation, explicit `target_commit_miss_then_worker_ready`, and successful paths that reach terminal `COMPLETED`.
- dynamic no-late-suffix counters now assert zero late-suffix provider calls, zero late-suffix consume paths, and zero post-commit publication.
- `attach_formal_online_runtime()` now runs collective preflight before control-group bootstrap and rolls back owner/wrappers on mid-install failures.
- a dedicated 4-rank attach/control-group gate now verifies deterministic two-group bootstrap and collective preflight failure handling.

Verification completed in this pass:

- `python -m compileall RS/src RS/tests RS/experiments/distributed/run_m1_formal_lifecycle_publication_gloo.py RS/experiments/distributed/run_m1_formal_attach_control_group_gloo.py`
- narrow M1 regression: `40 passed`
- target M1 suite: `68 passed`
- M0 regression subset covering prediction/planning parity and digests: `72 passed`
- formal 4-rank lifecycle publication Gloo gate: `passed`
- formal 4-rank attach/control-group Gloo gate: `passed`
