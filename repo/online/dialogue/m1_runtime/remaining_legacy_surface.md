# Remaining Legacy Surface

## Still present after current pass

- `attach_dispatch_observer(...)`
  - legacy observer attach path
- lifecycle `on_dispatch(...)` / `on_combine(...)`
  - retained for shadow/legacy paths only
- `TargetPlanStore.consume_once(...)`
  - compatibility shortcut
- `TargetPlanStore.close_key_if_unclaimed(...)`
  - late/tombstone helper
- pending-window adapters still depend on legacy scheduling contracts
  - outside the current partial M1 closure

## Not yet retired in this pass

- shadow and pending-window compatibility layers
- late suffix implementation source files
- full sync/preplanned parity proof against all runtime entrypoints
