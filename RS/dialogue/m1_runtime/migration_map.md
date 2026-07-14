# M1 Migration Map

## Completed in current pass

- `host.py`
  - returned `RuntimeHandle`
  - centralized restore callbacks
  - formal hooks now emit `RuntimeEvent`
  - root model now emits `ForwardBeginEvent` / `ForwardEndEvent`
- `lifecycle.py`
  - added `handle(...)`
  - pumps target-plan ready publications on main thread
  - prepared target-plan path uses `bind/start_execution/complete`
  - current-window sync planning now builds formal `PlanningRequest` directly
- `config.py`
  - moved policy-resolution helpers out of `runtime.py`
  - removed `lifecycle.py -> runtime.py` import edge
- `target_planning/planner_service.py`
  - non-blocking submit contract
  - ready-publication split from worker
  - per-task failure isolation
  - close/restart support
- `target_planning/store.py`
  - explicit state records and transitions

## Still retained

- `attach_dispatch_observer(...)`
- shadow-only `on_dispatch(...)` / `on_combine(...)`
- compatibility tombstone helpers in `TargetPlanStore`
