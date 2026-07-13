# M1 State Ownership

## Current owners

- hook installation / restore:
  - `RuntimeHandle`
- lifecycle progression:
  - `RouterSenseInjectionRuntime`
- target logical plan lifecycle:
  - `TargetPlanStore`
- preparation queueing and background build:
  - `TargetLayerPlannerService`

## Key runtime-owned fields

- active prediction:
  - `active_next_dispatch_prediction`
- prepared/current plan:
  - `prepared_plan`
  - `global_joint_window_plan`
- target-plan execution origin:
  - `execution_origin`
- target-plan digests:
  - `prepared_target_logical_plan_digest`
  - `stored_p1_plan_digest`

## Still retained compatibility surface

- `attach_dispatch_observer(...)` remains as observer-only compatibility entrypoint.
- `on_dispatch(...)` / `on_combine(...)` remain in lifecycle for legacy/shadow paths, but formal hook wrappers no longer invoke them.
