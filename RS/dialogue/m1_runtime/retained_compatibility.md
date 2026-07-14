# Retained Compatibility

- `attach_dispatch_observer(...)`
  - kept for observer-only scripts and legacy smoke collection
- `on_dispatch(...)` / `on_combine(...)`
  - retained for shadow / legacy internal paths
- `TargetPlanStore.consume_once(...)`
  - compatibility-only terminal shortcut
- `TargetPlanStore.claim_for_reconciliation(...)`
  - thin helper over `claim(...)`
- `TargetPlanStore.close_key_if_unclaimed(...)`
  - retained for late-no-effect tombstones
