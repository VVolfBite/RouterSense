"""Split lifecycle responsibility: LifecycleCombineHooksMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleCombineHooksMixin:
    def before_token_combine(self, *, layer_name: str, dispatcher: Any, packed_hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "selected":
            self._runtime_state.metrics.selected_p1_hook_count = int(self._runtime_state.metrics.selected_p1_hook_count) + 1
        self._timeline("before_token_combine_enter", layer_name=layer_name, phase_name="P1")
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        if layer_role != "selected":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="layer_role_not_selected",
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        self._pump_target_planner_publications()
        self._runtime_state.write("expert_compute_end_ns", int(hook_start_ns))
        observation_start_ns = time.monotonic_ns()
        observation = build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
            dispatcher=dispatcher,
            phase="P1",
            hidden_states=packed_hidden_states,
        )
        observation_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="build_runtime_observation",
            start_ns=observation_start_ns,
            end_ns=observation_end_ns,
            remote_rows=int(observation.remote_rows),
            remote_bytes=int(sum(int(v) for v in observation.per_peer_bytes)),
        )
        self._pending_p1[layer_name] = observation
        self._record_window_state(layer_name=layer_name, p1_observation=observation)
        self._record_release_update(layer_name=layer_name, event="p1_return_materialized")
        self._record_plan_arrival(layer_name=layer_name, phase="P1")
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P1")
        phase_ctx_start_ns = time.monotonic_ns()
        phase_ctx = build_phase_ready_context(
            PhaseContextBuildRequest(
                plan_key=self._plan_key(layer_name, "P1"),
                runtime_identity=RuntimeIdentity(
                    run_id=self.run_id,
                    forward_epoch=int(self._forward_epoch),
                    layer_id=parse_layer_id(layer_name),
                    layer_name=layer_name,
                    global_rank=self.rank,
                    local_rank=self.local_rank,
                    ep_group_ranks=self.ep_group_ranks,
                    ep_group_root_rank=self.ep_group_root_global_rank,
                ),
                topology=observation.topology.to_dict(),
                dispatcher_snapshot=DispatcherSnapshot(
                    dispatcher_class=type(dispatcher).__name__,
                    dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
                    expert_placement_hash=observation.expert_placement_hash,
                    input_splits=observation.input_splits,
                    output_splits=observation.output_splits,
                ),
                payload_contract=PhasePayloadContract(
                    phase="P1",
                    payload_roles=("hidden_states",),
                    atomic_submit=False,
                ),
                packed_tensors=(packed_hidden_states,) if isinstance(packed_hidden_states, torch.Tensor) else (),
                control_mode=self.config.control_mode,
                release_state="ready",
                demand_known_at="router_ready",
                payload_exists=True,
                p2_hint=p2_hint,
            )
        )
        phase_ctx_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="build_phase_ready_context",
            start_ns=phase_ctx_start_ns,
            end_ns=phase_ctx_end_ns,
            remote_rows=int(observation.remote_rows),
            hint_mode=str(p2_hint.hint_mode),
        )
        if self.observation_recorder is not None:
            self.observation_recorder.record_phase_context(
                phase_context_artifact(context=phase_ctx, perf_profile=self._is_perf_profile())
            )
            for bundle in phase_ctx.transport_bundles:
                self.observation_recorder.record_transport_bundle(
                    transport_bundle_artifact(bundle=bundle, perf_profile=self._is_perf_profile())
                )
        self._record_prepared_phase_plan_shadow(
            layer_name=layer_name,
            phase="P1",
            local_context=phase_ctx,
            global_contexts=(
                reconstruct_global_phase_contexts_from_byte_matrix(
                    local_context=phase_ctx,
                    matrix=tuple(
                        tuple(int(value) for value in row)
                        for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix")) or [])
                    ),
                    matrix_unit="rows",
                )
                if self._is_joint_window_async_mode()
                and ((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix"))
                else (phase_ctx,)
            ),
        )
        self._timeline(
            "p1_pre_transport_observation_ready",
            layer_name=layer_name,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P1")
        if self._should_schedule_phase(layer_name=layer_name, phase="P1"):
            if self._is_joint_window_async_mode():
                binding = self._current_prepared_plan_binding(layer_name=layer_name)
                stored_digest = str(self._runtime_state.read("stored_p1_plan_digest", "") or "")
                stored_logical_digest = str(self._runtime_state.read("stored_p1_logical_plan_digest", "") or "")
                stored_compile_input_digest = str(self._runtime_state.read("stored_p1_compile_input_digest", "") or "")
                if stored_digest:
                    self._runtime_state.write("consumed_p1_plan_digest", stored_digest)
                    self._runtime_state.write("consumed_p1_logical_plan_digest", stored_logical_digest or stored_digest)
                    self._runtime_state.write("consumed_p1_compile_input_digest", stored_compile_input_digest)
                elif binding is not None:
                    self._runtime_state.write("consumed_p1_plan_digest", str(binding.source_logical_plan_hash))
                    self._runtime_state.write("consumed_p1_logical_plan_digest", str(binding.source_logical_plan_hash))
                self._timeline(
                    "prepared_p1_plan_consumed",
                    layer_name=layer_name,
                    stored_p1_plan_digest=str(stored_digest),
                    consumed_p1_plan_digest=str(self._runtime_state.read("consumed_p1_plan_digest", "") or ""),
                    p1_plan_source_window=str(binding.window_key) if binding is not None else "",
                    p1_plan_consumed_once=True,
                )
                inferred_p1 = tuple(
                    tuple(int(value) for value in row)
                    for row in (
                        ((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix"))
                        or self._runtime_state.read("p1_inferred_from_p0")
                        or []
                    )
                )
                expected_send = tuple(int(value) for value in phase_ctx.send_splits)
                expected_recv = tuple(int(value) for value in phase_ctx.recv_splits)
                if inferred_p1:
                    local_index = tuple(int(v) for v in self.ep_group_ranks).index(int(self.rank))
                    inferred_send = tuple(int(inferred_p1[local_index][dst]) for dst in range(len(expected_send)))
                    inferred_recv = tuple(int(inferred_p1[src][local_index]) for src in range(len(expected_recv)))
                    inferred_total = int(sum(inferred_send) + sum(inferred_recv))
                    expected_total = int(sum(expected_send) + sum(expected_recv))
                    if inferred_total <= 0 and expected_total > 0:
                        self._timeline(
                            "p1_invariant_skipped_zero_inferred",
                            layer_name=layer_name,
                            inferred_send=list(inferred_send),
                            inferred_recv=list(inferred_recv),
                            actual_send=list(expected_send),
                            actual_recv=list(expected_recv),
                        )
                    elif inferred_send != expected_send or inferred_recv != expected_recv:
                        actual_p0_full = tuple(
                            tuple(int(value) for value in row)
                            for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
                        )
                        raise RuntimeError(
                            f"local P1 invariant mismatch for {layer_name}: "
                            f"inferred_send={inferred_send} actual_send={expected_send} "
                            f"inferred_recv={inferred_recv} actual_recv={expected_recv} "
                            f"local_index={local_index} actual_p0_full_row={actual_p0_full[local_index] if actual_p0_full and local_index < len(actual_p0_full) else ()}"
                        )
                plan = self._compile_async_local_phase_plan(
                    layer_name=layer_name,
                    phase="P1",
                    local_context=phase_ctx,
                )
                if self.target_plan_store is not None and getattr(self, "execution_pipeline", None) is not None:
                    key = self._target_plan_key(layer_name=layer_name)
                    published_execution_plan = self._execution_plan_cache().get(self.target_plan_store._key(key))
                    if published_execution_plan is not None:
                        prepared_execution = self.execution_pipeline.prepare(
                            published_execution_plan,
                            self._actual_phase_context_from_ready_context(phase_ctx=phase_ctx),
                        )
                        self._record_instrumentation_measurement(
                            event_type="materialization",
                            layer_id=str(phase_ctx.layer_id),
                            phase="P1",
                            started_at_ns=int(time.monotonic_ns()),
                            ended_at_ns=int(time.monotonic_ns()),
                            details={"valid": bool(prepared_execution.validation.valid)},
                        )
                        if not prepared_execution.validation.valid:
                            self.target_plan_store.fail(key, execution_origin="materialization_invalid_p1")
                            self.evidence_counters.materialization_failure_count += 1
                            self.evidence_counters.fallback_count += 1
                            self._runtime_state.write("execution_origin", "materialization_invalid_p1")
                            return
                        self._prepared_execution_cache()[self.target_plan_store._key(key)] = prepared_execution
                self._runtime_state.write("p1_planning_collective_count", 0)
                if self.observation_recorder is not None:
                    self.observation_recorder.record_scheduled_plan(
                        scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                    )
                self._activate_transport(layer_name=layer_name, phase="P1", context=phase_ctx, plan=plan)
                self._runtime_state.write("before_async_p2p_phase_count", int(self._runtime_state.read("before_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("combine_transport_start_ns", int(time.monotonic_ns()))
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_before_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_before_async_p2p_phase_count", 0) or 0) + 1)
                self._timeline(
                    "phase_execution_plan_agreed",
                    layer_name=layer_name,
                    phase_name="P1",
                    plan_hash=plan.plan_hash,
                    wave_count=len(plan.waves),
                    bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                    execution_mode=plan.execution_mode,
                    p1_planning_collective_count=0,
                )
                hook_end_ns = time.monotonic_ns()
                self._record_hook_timing(
                    layer_name=layer_name,
                    phase="P1",
                    hook_name="before_token_combine_total",
                    start_ns=hook_start_ns,
                    end_ns=hook_end_ns,
                    scheduled=True,
                    plan_hash=plan.plan_hash,
                    wave_count=int(len(plan.waves)),
                )
                self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=True, plan_hash=plan.plan_hash)
                return
            agreement_start_ns = time.monotonic_ns()
            policy = self._phase_policy()
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=policy, group=self.ep_process_group)
            agreement_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P1",
                stage="run_phase_plan_agreement",
                start_ns=agreement_start_ns,
                end_ns=agreement_end_ns,
                wave_count=int(len(plan.waves)),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
            )
            if self.observation_recorder is not None:
                self.observation_recorder.record_scheduled_plan(
                    scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                )
            self._record_control_replay_trace(phase_ctx=phase_ctx, plan=plan)
            self._activate_transport(layer_name=layer_name, phase="P1", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P1",
                plan_hash=plan.plan_hash,
                wave_count=len(plan.waves),
                bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                execution_mode=plan.execution_mode,
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
                total_agreement_time_us=float(plan.metrics.get("total_agreement_time_us", 0.0) or 0.0),
            )
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P1", plan_hash=plan.plan_hash)
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=True,
                plan_hash=plan.plan_hash,
                wave_count=int(len(plan.waves)),
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=True, plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="phase_policy_not_selected",
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P1",
            hook_name="before_token_combine_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="native_passthrough_identity",
        )
        self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
    def after_token_combine(self, *, layer_name: str) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("after_token_combine_enter", layer_name=layer_name, phase_name="P1")
        layer_id = parse_layer_id(layer_name)
        if str(layer_id).isdigit():
            next_layer_id = str(int(layer_id) + 1)
            self._pump_target_planner_publications()
            self._poll_target_plan_slot(target_layer_id=next_layer_id, safe_point="source_combine_complete")
        if self.layer_role_for_name(layer_name) == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="after_token_combine_total",
                start_ns=hook_start_ns,
            )
            self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        active_transport = self.current_transport()
        if self.layer_role_for_name(layer_name) == "selected" and active_transport is not None and str(active_transport.get("layer_name")) == str(layer_name) and str(active_transport.get("phase")) == "P1":
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P1")
            if self._is_joint_window_async_mode():
                self._runtime_state.write("after_async_p2p_phase_count", int(self._runtime_state.read("after_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("combine_transport_end_ns", int(time.monotonic_ns()))
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_after_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_after_async_p2p_phase_count", 0) or 0) + 1)
            clear_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P1",
                stage="clear_transport",
                start_ns=clear_start_ns,
                end_ns=clear_end_ns,
            )
            observation_p1 = self._pending_p1.pop(layer_name, None)
            if self._is_joint_window_async_mode():
                self._runtime_state.write("prepared_plan", None)
                self._runtime_state.remove("prepared_priority_cache", None)
            if observation_p1 is not None:
                self._record_window_state(layer_name=layer_name, p1_observation=observation_p1)
            if self._is_joint_window_async_mode() and self.target_plan_store is not None:
                execution_origin = str(self._runtime_state.read("execution_origin", "") or "")
                if execution_origin in {"prepared_exact", "prepared_repaired"}:
                    try:
                        self.target_plan_store.complete(
                            self._target_plan_key(layer_name=layer_name),
                            execution_origin=execution_origin,
                        )
                    except Exception as exc:
                        try:
                            self.target_plan_store.fail(
                                self._target_plan_key(layer_name=layer_name),
                                execution_origin="complete_failed",
                            )
                        except Exception:
                            pass
                        raise RuntimeError(f"prepared target completion failed for {layer_name}") from exc
            self._record_release_update(layer_name=layer_name, event="p1_return_completed")
            if self._should_stop_after_layer(layer_name=layer_name, phase="P1"):
                raise SelectedLayerStop(f"Stopped after selected P1 layer {layer_name}")
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="after_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
            )
            self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1")
            return
        if self.config.scheduler_mode == "native_passthrough_identity":
            self._timeline("native_p1_observed", layer_name=layer_name)
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P1",
            hook_name="after_token_combine_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
        )
        self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1")
    def on_dispatch(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="on_dispatch_total",
                start_ns=hook_start_ns,
            )
            return
        try:
            hook_mode = self._hook_execution_mode(layer_name=layer_name)
            if hook_mode in {"DISABLED", "OBSERVATION_ONLY"} or layer_role != "selected":
                return
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._finalize_dispatch_observation(
                    layer_name=layer_name,
                    dispatcher=dispatcher,
                    hidden_states=hidden_states,
                )
                return
            self._runtime_state.metrics.shadow_dispatch_execution_count = int(
                self._runtime_state.metrics.shadow_dispatch_execution_count
            ) + 1
            observation = build_runtime_observation(
                run_id=self.run_id,
                step_id=self.step_id,
                microbatch_id=self.microbatch_id,
                model_revision_hash=self.model_revision_hash,
                request_table_hash=self.request_table_hash,
                hostname=self.hostname,
                layer_name=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
                ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
                dispatcher=dispatcher,
                phase="P0",
                hidden_states=hidden_states,
            )
            self._pending_p0[layer_name] = observation
        finally:
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="on_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                mode=self._hook_execution_mode(layer_name=layer_name),
            )
    def on_combine(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="on_combine_total",
                start_ns=hook_start_ns,
            )
            return
        try:
            hook_mode = self._hook_execution_mode(layer_name=layer_name)
            if hook_mode in {"DISABLED", "OBSERVATION_ONLY"} or layer_role != "selected":
                return
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._finalize_combine_observation(
                    layer_name=layer_name,
                    dispatcher=dispatcher,
                    hidden_states=hidden_states,
                )
                return
            if layer_name not in self._pending_p0:
                return
            self._runtime_state.metrics.shadow_combine_execution_count = int(
                self._runtime_state.metrics.shadow_combine_execution_count
            ) + 1
            p0_observation = self._pending_p0.pop(layer_name)
            p1_observation = build_runtime_observation(
                run_id=self.run_id,
                step_id=self.step_id,
                microbatch_id=self.microbatch_id,
                model_revision_hash=self.model_revision_hash,
                request_table_hash=self.request_table_hash,
                hostname=self.hostname,
                layer_name=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
                ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
                dispatcher=dispatcher,
                phase="P1",
                hidden_states=hidden_states,
            )
            context = replace(self._context(layer_name), expert_placement_hash=p0_observation.expert_placement_hash)
            local_observations = (p0_observation, p1_observation)
            local_observation_bundle = ObservationBundle(
                run_id=str(self.run_id),
                forward_generation=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
                layer_id=str(parse_layer_id(layer_name)),
                ep_group_ranks=tuple(int(v) for v in self.ep_group_ranks),
                observations_by_phase={
                    "P0": p0_observation,
                    "P1": p1_observation,
                },
            )
            policy = self._phase_policy()
            self._runtime_state.metrics.shadow_plan_build_count = int(
                self._runtime_state.metrics.shadow_plan_build_count
            ) + 1
            self._runtime_state.metrics.shadow_policy_agreement_count = int(
                self._runtime_state.metrics.shadow_policy_agreement_count
            ) + 1
            self._runtime_state.metrics.shadow_control_collective_count = int(
                self._runtime_state.metrics.shadow_control_collective_count
            ) + 1
            plan, agreement = run_policy_agreement(
                local_observation=local_observation_bundle,
                context=context,
                policy=policy,
                device=torch.device(f"cuda:{self.local_rank}"),
                group=self.ep_process_group,
            )
            decision = InjectionDecision(
                accepted=True,
                fallback="native",
                plan_hash=plan.plan_hash,
                reason="native_order_passthrough" if plan.policy_name == "native_order" else "shadow_only_passthrough",
                policy_name=plan.policy_name,
                control_mode=self.config.control_mode,
            )
            self.completed.append(
                PolicyRuntimeRecord(
                    layer_name=layer_name,
                    context=context,
                    local_observations=local_observations,
                    plan=plan,
                    agreement=agreement,
                    decision=decision,
                )
            )
            self._record_observer(
                phase="policy_plan",
                layer=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                policy_name=plan.policy_name,
                scheduler_mode=self.config.scheduler_mode,
                control_mode=self.config.control_mode,
                plan_hash=plan.plan_hash,
                execution_mode=plan.execution_mode,
                wave_count=len(plan.waves),
                ready_wave_count=len(plan.ready_waves),
                blocked_future_wave_count=len(plan.blocked_future_waves),
                agreement=agreement.to_dict(),
                decision=decision.to_dict(),
            )
        finally:
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="on_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                mode=self._hook_execution_mode(layer_name=layer_name),
            )
