"""Split lifecycle responsibility: LifecycleDispatchHooksMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleDispatchHooksMixin:
    def before_token_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        hook_mode = self._hook_execution_mode(layer_name=layer_name)
        if layer_role == "selected":
            self._runtime_state.metrics.selected_p0_hook_count = int(self._runtime_state.metrics.selected_p0_hook_count) + 1
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._runtime_state.metrics.real_p0_execution_count = int(self._runtime_state.metrics.real_p0_execution_count) + 1
        self._timeline("before_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        if layer_role == "prediction_source":
            self.before_prediction_source_dispatch(
                layer_name=layer_name,
                dispatcher=dispatcher,
                packed_hidden_states=packed_hidden_states,
                packed_probs=packed_probs,
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        self._pump_target_planner_publications()
        self._poll_target_plan_slot(target_layer_id=str(parse_layer_id(layer_name)), safe_point="target_dispatch_ready")
        self._synchronize_dispatcher_tokens_or_raise(
            dispatcher=dispatcher,
            callsite_id="DTOH_P0_DISPATCHER_SYNC",
        )
        phase_ctx_start_ns = time.monotonic_ns()
        phase_ctx = self._build_phase_ready_context_from_dispatcher(
            layer_name=layer_name,
            phase="P0",
            dispatcher=dispatcher,
            packed_tensors=tuple(
                tensor for tensor in (packed_hidden_states, packed_probs) if isinstance(tensor, torch.Tensor)
            ),
        )
        phase_ctx_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="build_phase_ready_context",
            start_ns=phase_ctx_start_ns,
            end_ns=phase_ctx_end_ns,
            remote_rows=int(sum(int(v) for idx, v in enumerate(phase_ctx.send_splits) if idx != int(self._runtime_topology_dict()["ep_group_rank"]))),
            hint_mode="none",
        )
        pretransport = self._capture_pretransport_traffic_observation(phase_ctx=phase_ctx)
        matrix_device = self._matrix_device(packed_hidden_states)
        actual_p0_full_row_matrix = self._gather_actual_p0_full_row_matrix(
            layer_name=layer_name,
            observation=pretransport,
            device=matrix_device,
        )
        if self._should_generate_runtime_prediction():
            self._record_prediction_for_dispatch(
                layer_name=layer_name,
                phase_ctx=phase_ctx,
                observation=pretransport,
                actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                device=matrix_device,
            )
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P0")
        phase_ctx = replace(phase_ctx, p2_hint=p2_hint)
        observation_start_ns = time.monotonic_ns()
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
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
            ep_group_hash=ep_group_hash,
            dispatcher=dispatcher,
            phase="P0",
            hidden_states=packed_hidden_states,
        )
        observation_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="build_runtime_observation",
            start_ns=observation_start_ns,
            end_ns=observation_end_ns,
            remote_rows=int(observation.remote_rows),
            remote_bytes=int(sum(int(v) for v in observation.per_peer_bytes)),
        )
        if self.observation_recorder is not None and bool(getattr(self.config, "capture_expert_trace", False)):
            bytes_per_token = 1
            if isinstance(packed_hidden_states, torch.Tensor) and packed_hidden_states.ndim >= 1:
                bytes_per_token = int(packed_hidden_states.shape[-1]) * int(packed_hidden_states.element_size())
            maybe_capture_expert_route_trace(
                recorder=self.observation_recorder,
                layer_id=int(parse_layer_id(layer_name)) if str(parse_layer_id(layer_name)).isdigit() else 0,
                rank=int(self.rank),
                source_rank=int(self.rank),
                dispatcher=dispatcher,
                selected_experts=getattr(getattr(dispatcher, "_comm_manager", None), "token_indices", None),
                routing_weights=getattr(getattr(dispatcher, "_comm_manager", None), "token_probs", None),
                top_k=int(getattr(dispatcher, "router_topk", getattr(getattr(dispatcher, "_comm_manager", None), "router_topk", 1)) or 1),
                token_count=int(packed_hidden_states.shape[0]) if isinstance(packed_hidden_states, torch.Tensor) and packed_hidden_states.ndim >= 1 else 0,
                hidden_shape=tuple(int(v) for v in packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
                bytes_per_token=bytes_per_token,
                per_peer_bytes=tuple(int(v) for v in observation.per_peer_bytes),
                ep_group_ranks=tuple(int(v) for v in self.ep_group_ranks),
                enabled=True,
            )
        self._record_plan_arrival(layer_name=layer_name, phase="P0")
        self._pending_p0[layer_name] = observation
        self._record_window_state(layer_name=layer_name, p0_observation=observation)
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
                phase="P0",
                local_context=phase_ctx,
                global_contexts=(
                    reconstruct_global_phase_contexts_from_byte_matrix(
                        local_context=phase_ctx,
                        matrix=tuple(
                            tuple(int(value) for value in row)
                            for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
                        ),
                        matrix_unit="rows",
                    )
                    if self._is_joint_window_async_mode()
                    and ((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix"))
                    else (phase_ctx,)
                ),
            )
        pre_input_splits = tuple(int(v) for v in phase_ctx.input_splits)
        pre_output_splits = tuple(int(v) for v in phase_ctx.output_splits)
        hidden_ptr = int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1
        probs_ptr = int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1
        self._timeline(
            "p0_pre_transport_observation_ready",
            layer_name=layer_name,
            input_splits=list(pre_input_splits),
            output_splits=list(pre_output_splits),
            planning_traffic_source="pre_transport_phase_ready_context",
            pre_transport_observation_valid=bool(pretransport.valid),
            local_p0_row=list(pretransport.local_p0_row),
            actual_p0_total_rows=int(sum(sum(int(v) for v in row) for row in actual_p0_full_row_matrix)),
            hidden_shape=list(packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
            probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P0")
        if self._should_schedule_phase(layer_name=layer_name, phase="P0"):
            if self._is_joint_window_async_mode():
                target_plan = self._try_prepared_target_plan_for_p0(
                    layer_name=layer_name,
                    phase_ctx=phase_ctx,
                    actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                )
                if target_plan is None:
                    plan = self._build_provisional_async_plan(
                        layer_name=layer_name,
                        phase_ctx=phase_ctx,
                        observation_p0=pretransport,
                        actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                    )
                else:
                    plan = target_plan
                if self.observation_recorder is not None:
                    self.observation_recorder.record_scheduled_plan(
                        scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                    )
                self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
                adapter = getattr(self, "transport_adapter", None)
                if adapter is not None:
                    setattr(adapter, "late_suffix_provider", None)
                self._runtime_state.write("before_async_p2p_phase_count", int(self._runtime_state.read("before_async_p2p_phase_count", 0) or 0) + 1)
                self._timeline(
                    "phase_execution_plan_agreed",
                    layer_name=layer_name,
                    phase_name="P0",
                    plan_hash=plan.plan_hash,
                    wave_count=len(plan.waves),
                    bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                    execution_mode=plan.execution_mode,
                    execution_origin=str(self._runtime_state.read("execution_origin", "")),
                )
                hook_end_ns = time.monotonic_ns()
                self._record_hook_timing(
                    layer_name=layer_name,
                    phase="P0",
                    hook_name="before_token_dispatch_total",
                    start_ns=hook_start_ns,
                    end_ns=hook_end_ns,
                    scheduled=True,
                    plan_hash=plan.plan_hash,
                    wave_count=int(len(plan.waves)),
                )
                self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=True, plan_hash=plan.plan_hash)
                return
            agreement_start_ns = time.monotonic_ns()
            policy = self._phase_policy()
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=policy, group=self.ep_process_group)
            agreement_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
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
            self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P0",
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
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P0", plan_hash=plan.plan_hash)
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=True,
                plan_hash=plan.plan_hash,
                wave_count=int(len(plan.waves)),
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=True, plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="phase_policy_not_selected",
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        context = replace(self._context(layer_name), expert_placement_hash=observation.expert_placement_hash)
        local_observations = (observation,)
        local_observation_bundle = ObservationBundle(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
            layer_id=str(parse_layer_id(layer_name)),
            ep_group_ranks=tuple(int(v) for v in self.ep_group_ranks),
            observations_by_phase={"P0": observation},
        )
        plan, agreement = run_policy_agreement(
            local_observation=local_observation_bundle,
            context=context,
            policy=self._phase_policy(),
            device=torch.device(f"cuda:{self.local_rank}"),
            group=self.ep_process_group,
        )
        post_input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        post_output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        self.assertion_state["native_splits_unchanged"] = pre_input_splits == post_input_splits and pre_output_splits == post_output_splits
        self.assertion_state["native_buffers_unchanged"] = (
            hidden_ptr == (int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1)
            and probs_ptr == (int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1)
        )
        current_version = self._active_plan_versions.get(layer_name, 0)
        self._active_plan_versions[layer_name] = current_version
        self._active_plan_hashes[layer_name] = plan.plan_hash
        decision = InjectionDecision(
            accepted=True,
            fallback="native",
            plan_hash=plan.plan_hash,
            reason="identity_pre_transport_passthrough",
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
        self._timeline("root_plan_broadcast_received", layer_name=layer_name, root_wire_hash=agreement.root_wire_hash)
        self._timeline(
            "root_plan_decoded",
            layer_name=layer_name,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "plan_agreement_verified",
            layer_name=layer_name,
            agreement_status=agreement.agreement_status,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "identity_plan_agreed",
            layer_name=layer_name,
            root_wire_hash=agreement.root_wire_hash,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
            agreement_status=agreement.agreement_status,
            version=current_version,
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
            agreement=agreement.to_dict(),
            decision=decision.to_dict(),
        )
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "before_commit":
            self._active_plan_versions[layer_name] = current_version + 1
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current_version,
                    "new_version": current_version + 1,
                    "status": "applied",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_replaced_active",
                layer_name=layer_name,
                old_version=current_version,
                new_version=current_version + 1,
                transport_mutation=False,
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="before_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="native_passthrough_identity",
        )
        self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
    def mark_token_dispatch_committed(self, *, layer_name: str) -> None:
        if self.config.scheduler_mode != "native_passthrough_identity" and not bool(self._effective_phase_policy_name()):
            return
        self._timeline(
            "p0_native_dispatch_committed",
            layer_name=layer_name,
            active_version=self._active_plan_versions.get(layer_name, 0),
        )
    def after_token_dispatch(self, *, layer_name: str) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("after_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
        if self.layer_role_for_name(layer_name) == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        active_transport = self.current_transport()
        if self.layer_role_for_name(layer_name) == "selected" and active_transport is not None and str(active_transport.get("layer_name")) == str(layer_name) and str(active_transport.get("phase")) == "P0":
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P0")
            if self._is_joint_window_async_mode() and self.target_plan_store is not None:
                key = self._target_plan_key(layer_name=layer_name)
                self.target_plan_store.close_key_if_unclaimed(
                    key,
                    final_status="EXPIRED",
                    execution_origin="too_late_no_effect",
                )
            if self._is_joint_window_async_mode():
                self._runtime_state.write("after_async_p2p_phase_count", int(self._runtime_state.read("after_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_after_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_after_async_p2p_phase_count", 0) or 0) + 1)
            clear_end_ns = time.monotonic_ns()
            self._runtime_state.write("dispatch_transport_end_ns", int(clear_end_ns))
            self._runtime_state.write("rank_release_ns", int(clear_end_ns))
            self._runtime_state.write("expert_compute_start_ns", int(clear_end_ns))
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="clear_transport",
                start_ns=clear_start_ns,
                end_ns=clear_end_ns,
            )
            self._record_release_update(layer_name=layer_name, event="p0_dispatch_completed")
            if str(self.config.schedule_phase_selector).lower() == "p0" and self._should_stop_after_layer(layer_name=layer_name, phase="P0"):
                raise SelectedLayerStop(f"Stopped after selected P0 layer {layer_name}")
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0")
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                skipped=True,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0", skipped=True)
            return
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "after_commit":
            current = self._active_plan_versions.get(layer_name, 0)
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current,
                    "new_version": current + 1,
                    "status": "expired_late",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_expired_late",
                layer_name=layer_name,
                old_version=current,
                attempted_version=current + 1,
                transport_mutation=False,
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="after_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
        )
        self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0")
