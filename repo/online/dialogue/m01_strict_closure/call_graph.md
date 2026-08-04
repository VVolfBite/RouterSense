Key call graph updates in this pass:
- lifecycle._record_prediction_for_dispatch(): actual dispatch audit -> submit(TargetLayerPlanningRequest) only; no main-thread predictor call.
- TargetLayerPlannerService worker: predict_two_horizon once -> build planning request -> plan once or local/joint compare once -> ready result.
- lifecycle._pump_target_planner_publications(): publish_ready_plan() -> store worker H1/H2 predictions into runtime state -> timeline.
- TargetPlanStore publication: register_expected_publication(token) -> publish_if_current(token, plan).
