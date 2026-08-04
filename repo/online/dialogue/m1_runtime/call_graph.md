# M1 Call Graph

## Formal attach path

1. `attach_formal_online_runtime(...)`
2. `attach_dispatch_facade(...)`
3. build `RouterSenseInjectionRuntime`
4. wrap selected `token_dispatch` / `token_combine`
5. wrapped methods emit `RuntimeEvent`
6. `RouterSenseInjectionRuntime.handle(...)`
7. lifecycle stage methods:
   - `before_prediction_source_dispatch(...)`
   - `before_token_dispatch(...)`
   - `mark_token_dispatch_committed(...)`
   - `capture_phase_transport_output(...)`
   - `after_token_dispatch(...)`
   - `before_token_combine(...)`
   - `after_token_combine(...)`

## Target preplanning path

1. `_record_prediction_for_dispatch(...)`
2. `TargetLayerPlannerService.submit(...)`
3. worker builds prediction + formal planning request
4. worker builds `TargetLayerPreparedJointPlan`
5. worker appends ready publication only
6. main thread `_pump_target_planner_publications()`
7. `publish_ready_plan(...)`
8. `publish_agreed_plan(...)`
9. `TargetPlanStore.publish_logical(...)`
10. consumer path:
   - `peek(...)`
   - `claim(...)`
   - `bind(...)`
   - `start_execution(...)`
   - `complete(...)`
