"""Split lifecycle responsibility: LifecycleJointPlanningMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


def _reconcile_once_compat(**kwargs):
    from rs.runtime.online.megatron_ep import lifecycle as lifecycle_module
    return lifecycle_module.reconcile_once(**kwargs)


class LifecycleJointPlanningMixin:
    def _store_runtime_joint_plan_from_p0(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation_p0: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
        plan_origin: str = "current_joint_candidate",
    ) -> None:
        from rs.runtime.online.megatron_ep.async_release.runtime_projection import host_project_safe_selection

        self._assert_bucket_mode_consistency()
        self._register_current_plan_build(layer_name=layer_name, phase="P0", plan_origin=plan_origin)
        layer_id = parse_layer_id(layer_name)
        dispatch_matrix_full = tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
        dispatch_matrix = canonicalize_remote_matrix(dispatch_matrix_full)
        if not dispatch_matrix_full:
            return
        num_peers = len(dispatch_matrix_full)
        inferred_p1 = tuple(
            tuple(int(dispatch_matrix_full[col_idx][row_idx]) for col_idx in range(num_peers))
            for row_idx in range(num_peers)
        )
        remote_dispatch_matrix = canonicalize_remote_matrix(dispatch_matrix_full)
        num_peers = len(remote_dispatch_matrix)
        inferred_p1_remote = tuple(
            tuple(int(remote_dispatch_matrix[col_idx][row_idx]) for col_idx in range(num_peers))
            for row_idx in range(num_peers)
        )
        active_prediction = dict(self._runtime_state.read("active_next_dispatch_prediction") or {})
        forecast_matrix = tuple(
            tuple(int(value) for value in row)
            for row in active_prediction.get("forecast_matrix", ())
        ) if active_prediction and bool(active_prediction.get("valid", False)) else tuple(tuple(0 for _ in range(num_peers)) for _ in range(num_peers))
        predictor_name = str(active_prediction.get("predictor_name", "")) if active_prediction else ""
        prediction_digest = str(active_prediction.get("matrix_digest", "")) if active_prediction else ""
        prediction_confidence = float(active_prediction.get("confidence", 0.0) or 0.0) if active_prediction else 0.0
        next_layer_id = self._next_layer_id(layer_name)
        forecast_digest = stable_hash(
            {
                "forecast_matrix": [list(row) for row in forecast_matrix],
                "source_layer": str(layer_id),
                "target_layer": str(next_layer_id),
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
            }
        )
        effective_policy = str(self._current_window_planner_id() or "")
        phase_local_async_policies = {
            "fifo_bucket",
            "greedy_bucket",
            "birkhoff_bucket_phase_local",
            "fifo_bucket",
        }
        policy_options = PlannerPolicyConfig(
            p0_weight=float(self.config.p0_weight),
            p1_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
            p3_return_weight=float(getattr(self.config, "p3_return_weight", 0.0)),
            residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
            barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
            age_weight=float(getattr(self.config, "age_weight", 0.15)),
            prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
        )
        formal_request = self._build_formal_planning_request(
            request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:current_window",
            source_layer_id=str(layer_id),
            target_layer_id=str(next_layer_id),
            p0_dispatch_rows=remote_dispatch_matrix,
            p1_return_rows=inferred_p1_remote,
            p2_hint_rows=forecast_matrix,
            predictor_name=str(predictor_name or "zero"),
            prediction_confidence=float(prediction_confidence),
            information_mode=self._planning_information_mode(),
            max_waves=int(getattr(self.config, "max_waves", 256)),
        )
        formal_cost_model = PlanningCostModel(
            expert_compute_delay=float(formal_request.constraints.expert_compute_delay),
            full_duplex=bool(formal_request.topology.full_duplex),
            max_outgoing_per_rank_per_wave=int(formal_request.topology.max_outgoing_per_rank_per_wave),
            max_incoming_per_rank_per_wave=int(formal_request.topology.max_incoming_per_rank_per_wave),
        )
        effective_axes = parse_planner_axes(effective_policy) if is_axes_planner_id(effective_policy) else None
        if effective_policy in phase_local_async_policies or (effective_axes is not None and effective_axes.scope == "local"):
            joint_planner_name = effective_policy
            local_planner_name = effective_policy
            joint_start_ns = time.monotonic_ns()
            planner_config = dict(getattr(self.config, "planner_config", {}) or {})
            joint_window_plan = PlannerRegistry.create(joint_planner_name, planner_config, usage="runtime").plan(formal_request)
            joint_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="joint_build",
                start_ns=joint_start_ns,
                end_ns=joint_end_ns,
                policy_name=joint_planner_name,
            )
            self._increment_state_counter_map("joint_build_count_by_layer", str(layer_id))
            joint_plan = _compat_logical_plan_from_window_plan(joint_window_plan)
            local_start_ns = time.monotonic_ns()
            local_window_plan = joint_window_plan
            local_plan = joint_plan
            local_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="local_build",
                start_ns=local_start_ns,
                end_ns=local_end_ns,
                policy_name=local_planner_name,
            )
            self._increment_state_counter_map("local_build_count_by_layer", str(layer_id))
            selected_window_plan = joint_window_plan
            selected_plan = joint_plan
        else:
            joint_planner_name, local_planner_name = self._runtime_safe_scope_pair(effective_policy)
            safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
            joint_start_ns = time.monotonic_ns()
            planner_config = dict(getattr(self.config, "planner_config", {}) or {})
            joint_window_plan = PlannerRegistry.create(joint_planner_name, planner_config, usage="runtime").plan(formal_request)
            joint_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="joint_build",
                start_ns=joint_start_ns,
                end_ns=joint_end_ns,
                policy_name=joint_planner_name,
            )
            self._increment_state_counter_map("joint_build_count_by_layer", str(layer_id))
            joint_plan = _compat_logical_plan_from_window_plan(joint_window_plan)
            local_start_ns = time.monotonic_ns()
            if safe_projection_mode == "disabled":
                local_window_plan = joint_window_plan
                local_plan = joint_plan
                local_end_ns = local_start_ns
            else:
                local_window_plan = PlannerRegistry.create(local_planner_name, planner_config, usage="runtime").plan(formal_request)
                local_plan = _compat_logical_plan_from_window_plan(local_window_plan)
                local_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="local_build",
                start_ns=local_start_ns,
                end_ns=local_end_ns,
                policy_name=local_planner_name,
                skipped=bool(safe_projection_mode == "disabled"),
            )
            if safe_projection_mode != "disabled":
                self._increment_state_counter_map("local_build_count_by_layer", str(layer_id))
            selector = PlannerSelector(
                local_planner=PlannerRegistry.create(local_planner_name, planner_config, usage="runtime"),
                joint_planner=PlannerRegistry.create(joint_planner_name, planner_config, usage="runtime"),
                estimator=CommonCorePlanEstimator(),
                cost_model=formal_cost_model,
            )
            if safe_projection_mode == "disabled":
                selected_window_plan = joint_window_plan
                selected_plan = joint_plan
            else:
                selected = selector.select_prebuilt(
                    request=formal_request,
                    local_plan=local_window_plan,
                    joint_plan=joint_window_plan,
                    mode=PlannerSelectionMode.COMPARE,
                )
                selected_window_plan = selected.selected_plan
                selected_plan = _compat_logical_plan_from_window_plan(selected_window_plan)
        safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
        if safe_projection_mode == "disabled":
            joint_candidate_score = CommonCorePlanEstimator().estimate(joint_window_plan, formal_request, formal_cost_model)
            host_projection_start_ns = time.monotonic_ns()
            host_projection_end_ns = host_projection_start_ns
            safe_projection = {
                "ideal_joint_candidate_estimated_makespan": float(joint_candidate_score.estimated_makespan),
                "host_projected_joint_candidate_estimated_makespan": float(joint_candidate_score.estimated_makespan),
                "ideal_local_fallback_estimated_makespan": float(joint_candidate_score.estimated_makespan),
                "host_projected_local_fallback_estimated_makespan": float(joint_candidate_score.estimated_makespan),
                "host_projected_safe_selection": str(joint_plan.policy_name),
                "projection_mode": "disabled",
            }
        else:
            host_projection_start_ns = time.monotonic_ns()
            safe_projection = host_project_safe_selection(
                joint_plan=joint_plan,
                local_plan=local_plan,
            )
            host_projection_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="host_projection",
            start_ns=host_projection_start_ns,
            end_ns=host_projection_end_ns,
            safe_projection_mode=safe_projection_mode,
        )
        actual_p0_row_matrix = [[int(value) for value in row] for row in remote_dispatch_matrix]
        actual_p0_full_row_matrix_list = [[int(value) for value in row] for row in dispatch_matrix_full]
        inferred_p1_row_matrix = [[int(value) for value in row] for row in inferred_p1]
        inferred_p1_remote_row_matrix = [[int(value) for value in row] for row in inferred_p1_remote]
        safe_selection_start_ns = time.monotonic_ns()
        safe_selection_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="safe_selection",
            start_ns=safe_selection_start_ns,
            end_ns=safe_selection_end_ns,
            selected_policy=str(selected_plan.policy_name),
            safe_projection_mode=safe_projection_mode,
        )
        prepared = PreparedWindowPlan(
            window_key=stable_hash(
                {
                    "runtime_safe_scope": bool(safe_projection_mode != "disabled"),
                    "safe_projection_mode": safe_projection_mode,
                    "joint_policy": joint_planner_name,
                    "local_policy": local_planner_name,
                    "selected_policy": str(selected_plan.policy_name),
                    "created_at_layer_id": str(layer_id),
                    "applies_from_layer_id": str(next_layer_id),
                    "forecast_digest": forecast_digest,
                }
            )[:16],
            forecast_digest=forecast_digest,
            logical_plan=selected_plan,
            created_at_layer_id=str(layer_id),
            applies_from_layer_id=str(next_layer_id),
            execution_capability_required="joint_window_async_p2p",
            forecast_matrix=forecast_matrix,
        )
        self._runtime_state.write("prepared_plan", prepared)
        self._runtime_state.write("plan_created_at_us", int(time.time() * 1e6))
        self._runtime_state.write("plan_source_layer", layer_name)
        stored_logical_digest = stable_hash(selected_plan.to_dict())
        stored_compile_input_digest = stable_hash(
            {
                "phase": "P1",
                "layer_name": str(layer_name),
                "forward_epoch": int(self._forward_epoch),
                "matrix": [list(row) for row in inferred_p1],
            }
        )
        self._runtime_state.write("stored_p1_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_logical_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_compile_input_digest", stored_compile_input_digest)
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.write("consumed_p1_logical_plan_digest", "")
        self._runtime_state.write("consumed_p1_compile_input_digest", "")
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.write("requested_bucket_mode", str(self._requested_bucket_mode()))
        self._runtime_state.write("effective_bucket_mode", str(self._effective_bucket_mode()))
        self._runtime_state.write("requested_bucket_rows", int(self.config.bucket_rows))
        self._runtime_state.write("effective_bucket_rows", int(self.config.bucket_rows))
        self._runtime_state.write("predicted_row_sums", [int(sum(row)) for row in forecast_matrix])
        self._runtime_state.write(
            "predicted_col_sums",
            [
            int(sum(forecast_matrix[row_idx][col_idx] for row_idx in range(len(forecast_matrix))))
            for col_idx in range(len(forecast_matrix[0]) if forecast_matrix else 0)
            ],
        )
        self._runtime_state.write("p2_matrix_source", "active_next_dispatch_prediction" if active_prediction else "zero_hint")
        self._runtime_state.write("p2_matrix_total_bytes", int(sum(sum(int(v) for v in row) for row in forecast_matrix)))
        self._runtime_state.write("p1_inferred_from_p0", [list(row) for row in inferred_p1])
        self._runtime_state.write(
            "global_joint_window_plan",
            {
            "window_key": str(prepared.window_key),
            "source_layer_id": str(layer_id),
            "target_layer_id": str(next_layer_id),
            "predictor_name": predictor_name,
            "prediction_digest": prediction_digest,
            "prediction_confidence": float(prediction_confidence),
            "actual_p0_matrix": [list(row) for row in remote_dispatch_matrix],
            "actual_p0_row_matrix": actual_p0_row_matrix,
            "actual_p0_full_matrix": [list(row) for row in dispatch_matrix_full],
            "actual_p0_full_row_matrix": actual_p0_full_row_matrix_list,
            "inferred_p1_matrix": [list(row) for row in inferred_p1],
            "inferred_p1_row_matrix": inferred_p1_row_matrix,
            "inferred_p1_remote_matrix": [list(row) for row in inferred_p1_remote],
            "inferred_p1_remote_row_matrix": inferred_p1_remote_row_matrix,
            "predicted_p2_matrix": [list(row) for row in forecast_matrix],
            "created_stage": "after_p0_observation",
            "planning_traffic_source": str(observation_p0.source),
            "captured_before_transport": bool(observation_p0.captured_before_transport),
            "pre_transport_observation_valid": bool(observation_p0.valid),
            "dispatcher_send_splits": list(observation_p0.send_splits_rows),
            "dispatcher_recv_splits": list(observation_p0.recv_splits_rows),
            "local_p0_row": list(observation_p0.local_p0_row),
            "actual_p0_total_rows": int(sum(sum(int(v) for v in row) for row in dispatch_matrix_full)),
            "p1_is_exact_transpose": bool(tuple(tuple(int(v) for v in row) for row in inferred_p1) == tuple(tuple(int(dispatch_matrix_full[col][row]) for col in range(len(dispatch_matrix_full))) for row in range(len(dispatch_matrix_full)))),
            "joint_planner_name": joint_planner_name,
            "local_planner_name": local_planner_name,
            "safe_projection_mode": safe_projection_mode,
            "requested_bucket_mode": str(self._requested_bucket_mode()),
            "effective_bucket_mode": str(self._effective_bucket_mode()),
            "requested_bucket_rows": int(self.config.bucket_rows),
            "effective_bucket_rows": int(self.config.bucket_rows),
            "default_weights": dict((joint_plan.diagnostics or {}).get("default_weights", {})),
            "requested_weights": dict((joint_plan.diagnostics or {}).get("requested_weights", {})),
            "effective_weights": dict((joint_plan.diagnostics or {}).get("effective_weights", {})),
            "consumed_weights": dict((joint_plan.diagnostics or {}).get("consumed_weights", {})),
            "safe_selected_policy": str(joint_planner_name if stable_hash(selected_plan.to_dict()) == stable_hash(joint_plan.to_dict()) else local_planner_name),
            "safe_selection_margin": float(
                safe_projection["host_projected_local_fallback_estimated_makespan"]
                - safe_projection["host_projected_joint_candidate_estimated_makespan"]
            ),
            "safe_comparison_is_strict_common_core": bool(
                dict((joint_plan.diagnostics or {}).get("common_core", {}))
                == dict((local_plan.diagnostics or {}).get("common_core", {}))
            ),
            "common_core_metadata": dict((joint_plan.diagnostics or {}).get("common_core", {})),
            "joint_plan_policy": str(joint_plan.policy_name),
            "local_plan_policy": str(local_plan.policy_name),
            "joint_candidate_plan_digest": stable_hash(joint_plan.to_dict()),
            "local_plan_digest": stable_hash(local_plan.to_dict()),
            "selected_plan_digest": stable_hash(selected_plan.to_dict()),
            "local_build_count": 0 if safe_projection_mode == "disabled" else 1,
            "host_projection_count": 0 if safe_projection_mode == "disabled" else 1,
            "runtime_policy_equivalent_of": effective_policy,
            "service_demand_model": "rows_from_pre_transport_phase_ready_context",
            "bundle_bytes_per_row": int(self._bundle_bytes_per_row(phase_ctx=phase_ctx)),
            },
        )
        global_joint_window_plan = dict(self._runtime_state.read("global_joint_window_plan") or {})
        global_joint_window_plan["host_projected_safe_selection"] = dict(safe_projection)
        self._runtime_state.write("global_joint_window_plan", global_joint_window_plan)
        self._runtime_state.write("ideal_joint_candidate_makespan", float(safe_projection["ideal_joint_candidate_estimated_makespan"]))
        self._runtime_state.write("ideal_local_fallback_makespan", float(safe_projection["ideal_local_fallback_estimated_makespan"]))
        self._runtime_state.write("host_projected_joint_candidate_makespan", float(safe_projection["host_projected_joint_candidate_estimated_makespan"]))
        self._runtime_state.write("host_projected_local_fallback_makespan", float(safe_projection["host_projected_local_fallback_estimated_makespan"]))
        self._runtime_state.write("joint_candidate_plan_digest", stable_hash(joint_plan.to_dict()))
        self._runtime_state.write("local_plan_digest", stable_hash(local_plan.to_dict()))
        self._runtime_state.write("selected_plan_digest", stable_hash(selected_plan.to_dict()))
        self._runtime_state.write("local_build_count", 0 if safe_projection_mode == "disabled" else 1)
        self._runtime_state.write("host_projection_count", 0 if safe_projection_mode == "disabled" else 1)
        self._runtime_state.write(
            "prediction_consumption_records",
            [
                {
                "prediction_first_consumed_stage": "during_p0_joint_planning",
                "consumer_layer": str(layer_id),
                "consumer_phase": "P1",
                "consumed_before_p1": True,
                "source_layer_id": str(active_prediction.get("source_layer_id", "")) if active_prediction else str(layer_id),
                "target_layer_id": str(active_prediction.get("target_layer_id", "")) if active_prediction else str(next_layer_id),
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
                "prediction_confidence": float(prediction_confidence),
                "prediction_matrix_total": int(sum(sum(int(v) for v in row) for row in forecast_matrix)),
                "consumed_during_p0_joint_planning": True,
                }
            ],
        )
        self._runtime_state.write(
            "host_projected_estimated_makespan",
            float(
            safe_projection["host_projected_local_fallback_estimated_makespan"]
            if str(selected_plan.policy_name) == str(local_plan.policy_name)
            else safe_projection["host_projected_joint_candidate_estimated_makespan"]
            ),
        )
        self._runtime_state.write(
            "ideal_estimated_makespan",
            float(
            safe_projection["ideal_local_fallback_estimated_makespan"]
            if str(selected_plan.policy_name) == str(local_plan.policy_name)
            else safe_projection["ideal_joint_candidate_estimated_makespan"]
            ),
        )
        self._runtime_state.remove("prepared_priority_cache", None)
        global_joint_window_plan = dict(self._runtime_state.read("global_joint_window_plan") or {})
        self._timeline(
            "runtime_joint_window_plan_stored",
            layer_name=layer_name,
            source_layer_id=str(layer_id),
            target_layer_id=str(next_layer_id),
            planning_traffic_source=str(observation_p0.source),
            captured_before_transport=bool(observation_p0.captured_before_transport),
            pre_transport_observation_valid=bool(observation_p0.valid),
            actual_p0_total_rows=int(sum(sum(int(v) for v in row) for row in dispatch_matrix_full)),
            actual_p0_matrix_unit="rows",
            p1_is_exact_transpose=bool(global_joint_window_plan.get("p1_is_exact_transpose", False)),
            prediction_digest=prediction_digest,
            prediction_confidence=float(prediction_confidence),
            predictor_name=predictor_name or "zero_hint",
            prediction_matrix_total=int(sum(sum(int(v) for v in row) for row in forecast_matrix)),
            stored_p1_plan_digest=str(self._runtime_state.read("stored_p1_plan_digest", "")),
            consumed_during_p0_joint_planning=True,
            ideal_joint_candidate_makespan=float(safe_projection["ideal_joint_candidate_estimated_makespan"]),
            ideal_local_fallback_makespan=float(safe_projection["ideal_local_fallback_estimated_makespan"]),
            host_projected_joint_candidate_makespan=float(safe_projection["host_projected_joint_candidate_estimated_makespan"]),
            host_projected_local_fallback_makespan=float(safe_projection["host_projected_local_fallback_estimated_makespan"]),
            host_projected_estimated_makespan=float(self._runtime_state.read("host_projected_estimated_makespan", 0.0)),
            ideal_estimated_makespan=float(self._runtime_state.read("ideal_estimated_makespan", 0.0)),
            safe_selected_policy=str(joint_planner_name if stable_hash(selected_plan.to_dict()) == stable_hash(joint_plan.to_dict()) else local_planner_name),
            joint_planner_name=joint_planner_name,
            local_planner_name=local_planner_name,
        )
    def _try_prepared_target_plan_for_p0(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
    ) -> PhaseExecutionPlan | None:
        if not self._policy_supports_target_layer_preplanning() or self.target_plan_store is None:
            return None
        key = self._target_plan_key(layer_name=layer_name)
        peeked = self.target_plan_store.peek(key)
        if peeked is None:
            self._runtime_state.write("prepared_plan_found", False)
            return None
        prepared_plan = self.target_plan_store.claim_for_reconciliation(key)
        reconcile_key = (key.run_id, key.forward_epoch, key.microbatch_id, key.target_layer_id)
        if reconcile_key in self._target_plan_reconciled_keys:
            raise RuntimeError(f"target plan reconcile_once double invocation for {reconcile_key}")
        outcome = _reconcile_once_compat(
            prepared_plan=prepared_plan,
            actual_p0_rows=canonicalize_remote_matrix(actual_p0_full_row_matrix),
        )
        inferred_p1_rows = tuple(
            tuple(int(actual_p0_full_row_matrix[col_idx][row_idx]) for col_idx in range(len(actual_p0_full_row_matrix)))
            for row_idx in range(len(actual_p0_full_row_matrix))
        )
        self._target_plan_reconciled_keys.add(reconcile_key)
        self._runtime_state.write("prepared_plan_found", True)
        self._runtime_state.write("reconciliation_count", 1)
        self._runtime_state.write("full_u_replan_count", 0)
        self._runtime_state.write("prepared_target_selected_variant", str(getattr(prepared_plan, "selected_variant", "")))
        self._runtime_state.write(
            "prepared_target_safe_projection_mode",
            str(getattr(prepared_plan, "safe_projection_mode", "disabled") or "disabled"),
        )
        if outcome.status == "rejected" or outcome.logical_plan is None:
            self.target_plan_store.fail(key, execution_origin="prepared_rejected")
            self.evidence_counters.preparation_miss_count += 1
            self.evidence_counters.fallback_count += 1
            self._runtime_state.write("execution_origin", "prepared_rejected")
            return None
        target_matrix = (
            tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
            if str(phase_ctx.phase) == "P0"
            else tuple(tuple(int(value) for value in row) for row in (self._runtime_state.read("p1_inferred_from_p0") or []))
        )
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=outcome.logical_plan,
            layer_name=layer_name,
            phase=str(phase_ctx.phase),
            local_context=phase_ctx,
            matrix=target_matrix,
            plan_origin="prepared_exact" if outcome.status == "exact" else "prepared_repaired",
            plan_version=1,
        )
        synthetic_prepared = PreparedWindowPlan(
            window_key=stable_hash({"target_layer": str(layer_name), "origin": str(outcome.status)})[:16],
            forecast_digest=str(prepared_plan.h1_prediction_digest),
            logical_plan=outcome.logical_plan,
            created_at_layer_id=str(prepared_plan.source_layer_id),
            applies_from_layer_id=str(prepared_plan.target_layer_id),
            execution_capability_required="joint_window_async_p2p",
            forecast_matrix=tuple(tuple(int(value) for value in row) for row in prepared_plan.h1_rows),
        )
        stored_logical_digest = str(outcome.logical_plan_digest or stable_hash(outcome.logical_plan.to_dict()))
        stored_compile_input_digest = stable_hash(
            {
                "phase": "P1",
                "layer_name": str(layer_name),
                "forward_epoch": int(self._forward_epoch),
                "matrix": [list(row) for row in inferred_p1_rows],
            }
        )
        self._runtime_state.write("prepared_plan", synthetic_prepared)
        self._runtime_state.write("stored_p1_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_logical_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_compile_input_digest", stored_compile_input_digest)
        self._runtime_state.write("p1_inferred_from_p0", [list(row) for row in inferred_p1_rows])
        execution_origin = "prepared_exact" if outcome.status == "exact" else "prepared_repaired"
        self.target_plan_store.bind(key, bound_owner="prepared_reconcile")
        self.target_plan_store.start_execution(key, execution_origin=execution_origin, claim_owner="prepared_reconcile")
        self._runtime_state.write("execution_origin", execution_origin)
        self._runtime_state.write("prepared_target_logical_plan_digest", str(outcome.logical_plan_digest or ""))
        published_execution_plan = self._execution_plan_cache().get(self.target_plan_store._key(key))
        execution_pipeline = getattr(self, "execution_pipeline", None)
        if published_execution_plan is not None and execution_pipeline is not None:
            prepare_start_ns = time.monotonic_ns()
            prepared_execution = execution_pipeline.prepare(
                published_execution_plan,
                self._actual_phase_context_from_ready_context(phase_ctx=phase_ctx),
            )
            prepare_end_ns = time.monotonic_ns()
            self._record_instrumentation_measurement(
                event_type="materialization",
                layer_id=str(phase_ctx.layer_id),
                phase=str(phase_ctx.phase),
                started_at_ns=prepare_start_ns,
                ended_at_ns=prepare_end_ns,
                details={"valid": bool(prepared_execution.validation.valid)},
            )
            if not prepared_execution.validation.valid:
                self.target_plan_store.fail(key, execution_origin="materialization_invalid")
                self.evidence_counters.materialization_failure_count += 1
                self.evidence_counters.fallback_count += 1
                self._runtime_state.write("execution_origin", "materialization_invalid")
                return None
            self._prepared_execution_cache()[self.target_plan_store._key(key)] = prepared_execution
        return compiled
