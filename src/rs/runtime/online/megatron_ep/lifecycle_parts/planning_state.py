"""Split lifecycle responsibility: LifecyclePlanningStateMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecyclePlanningStateMixin:
    def _build_p2_hint(self, *, layer_name: str, phase: str):
        start_ns = time.monotonic_ns()
        if self.config.p2_hint_mode == "calibrated_artifact":
            if self._p2_hint_provider is None:
                self._p2_hint_provider = build_p2_hint_provider(
                    self.config.p2_hint_mode,
                    shared_state=self._runtime_state,
                )
            provider = self._p2_hint_provider
        else:
            provider = build_p2_hint_provider(self.config.p2_hint_mode)
        hint = provider.build_hint(
            P2HintRequest(
                plan_key=self._plan_key(layer_name, phase),
                layer_id=parse_layer_id(layer_name),
                phase=phase,
                global_rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
            )
        )
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="build_p2_hint",
            start_ns=start_ns,
            end_ns=end_ns,
            hint_mode=str(hint.hint_mode),
            hint_source=str(hint.hint_source),
        )
        return hint
    def _record_plan_arrival(self, *, layer_name: str, phase: str) -> None:
        now_us = int(time.time() * 1e6)
        plan = self._runtime_state.read("prepared_plan")
        plan_created_at = int(self._runtime_state.read("plan_created_at_us", 0) or 0)
        source_layer = str(self._runtime_state.read("plan_source_layer", ""))
        if plan is None:
            arrival_status = "none"
            plan_age_us = 0
        else:
            plan_age_us = max(0, now_us - plan_created_at)
            if self.config.control_mode == "sync_before_phase":
                arrival_status = "before_commit"
            else:
                arrival_status = "before_commit" if plan_age_us > 100 else "in_flight"
        record = {
            "ts_us": now_us,
            "layer_name": layer_name,
            "phase": phase,
            "arrival_status": arrival_status,
            "plan_age_us": plan_age_us,
            "source_layer": source_layer,
            "control_mode": self.config.control_mode,
            "has_prepared_plan": plan is not None,
            "window_key": str(getattr(plan, "window_key", "")) if plan is not None else "",
            "forecast_digest": str(getattr(plan, "forecast_digest", "")) if plan is not None else "",
        }
        self.plan_arrival_records.append(record)
        self._timeline(
            "shadow_plan_arrival",
            layer_name=layer_name,
            phase_name=phase,
            arrival_status=arrival_status,
            plan_age_us=plan_age_us,
            source_layer=source_layer,
            has_prepared_plan=plan is not None,
        )
    def _current_prepared_plan_binding(self, *, layer_name: str) -> PreparedPlanBinding | None:
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            return None
        source_logical_plan_hash = ""
        logical_plan = getattr(prepared_plan, "logical_plan", None)
        if logical_plan is not None:
            source_logical_plan_hash = stable_hash(logical_plan.to_dict())
        return bind_prepared_plan(
            layer_name=layer_name,
            prepared_plan=prepared_plan,
            source_layer_name=str(self._runtime_state.read("plan_source_layer", "")),
            source_logical_plan_hash=source_logical_plan_hash,
        )
    def _record_window_state(
        self,
        *,
        layer_name: str,
        p0_observation: RuntimeObservation | None = None,
        p1_observation: RuntimeObservation | None = None,
    ) -> None:
        start_ns = time.monotonic_ns()
        existing = self._window_states.get(layer_name)
        release_state = WindowReleaseState() if existing is None else existing.release_state
        state, record = build_window_state_record(
            layer_name=layer_name,
            ep_group_ranks=self.ep_group_ranks,
            local_rank=self.local_rank,
            p0_observation=p0_observation if p0_observation is not None else (None if existing is None else existing.p0_observation),
            p1_observation=p1_observation if p1_observation is not None else (None if existing is None else existing.p1_observation),
            prepared_plan=self._runtime_state.read("prepared_plan"),
            prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
            release_state=release_state,
        )
        self._window_states[layer_name] = state
        self.window_state_records.append(record)
        self._increment_state_counter_map("window_state_count_by_layer", str(parse_layer_id(layer_name)))
        if state.prepared_plan_binding is not None:
            self.prepared_plan_bindings.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    **state.prepared_plan_binding.to_dict(),
                }
            )
        shadow = maybe_build_window_shadow(
            enabled=self._allow_shadow_artifacts(),
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        if shadow is not None:
            self.window_schedule_shadows.append(shadow)
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="control",
            stage="record_window_state",
            start_ns=start_ns,
            end_ns=end_ns,
            has_p0=bool(state.p0_observation is not None),
            has_p1=bool(state.p1_observation is not None),
            has_prepared_plan=bool(state.prepared_plan_binding is not None),
        )
    def _record_release_update(self, *, layer_name: str, event: str) -> None:
        state = self._window_states.get(layer_name)
        if state is None:
            state, _ = build_window_state_record(
                layer_name=layer_name,
                ep_group_ranks=self.ep_group_ranks,
                local_rank=self.local_rank,
                p0_observation=None,
                p1_observation=None,
                prepared_plan=self._runtime_state.read("prepared_plan"),
                prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
                release_state=WindowReleaseState(),
            )
        state, record, state_record = advance_window_release(state=state, event=event, rank=self.rank, layer_name=layer_name)
        self._window_states[layer_name] = state
        self.release_events.append(record)
        self.window_state_records.append(state_record)
        shadow = maybe_build_window_shadow(
            enabled=self._allow_shadow_artifacts(),
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        if shadow is not None:
            self.window_schedule_shadows.append(shadow)
    def _record_prepared_phase_plan_shadow(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> None:
        if not self._allow_shadow_artifacts():
            return
        start_ns = time.monotonic_ns()
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_prepared_plan",
            )
            return
        binding = self._current_prepared_plan_binding(layer_name=layer_name)
        if binding is None:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_binding",
            )
            return
        phase_policy_name = self._effective_phase_policy_name()
        if not phase_policy_name:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_policy",
            )
            return
        try:
            compilation = compile_schedule(
                PlanCompilationRequest(
                    logical_plan=getattr(prepared_plan, "logical_plan"),
                    local_context=local_context,
                    global_contexts=global_contexts,
                    canonical_tasks=(),
                    phase=str(phase),
                    tensor_role="shadow",
                    rank_context={
                        "global_rank": int(local_context.global_rank),
                        "local_rank": int(local_context.local_rank),
                    },
                    compilation_options=CompilationOptions(
                        bucket_rows=int(self.config.bucket_rows),
                        p0_weight=float(self.config.p0_weight),
                        p1_reservation_weight=float(self.config.p1_reservation_weight),
                        p2_hint_weight=float(self.config.p2_hint_weight),
                        debug_trace=not self._is_perf_profile(),
                        invariant_mode="diagnostic",
                        diagnostic_compiler_fallback=True,
                    ),
                    prepared_plan=prepared_plan,
                    prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                    phase_policy_name=str(phase_policy_name),
                )
            )
            compiled = compilation.execution_plan
        except Exception as exc:  # pragma: no cover
            self.prepared_phase_plan_shadows.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    "phase": phase,
                    "prepared_window_key": binding.window_key,
                    "compile_status": "failed",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="failed",
                exception=f"{type(exc).__name__}: {exc}",
            )
            return
        self.prepared_phase_plan_shadows.append(
            {
                "ts_us": int(time.time() * 1e6),
                "layer_name": layer_name,
                "phase": phase,
                "prepared_window_key": binding.window_key,
                "compile_status": "ok",
                "source_layer_name": binding.source_layer_name,
                "source_logical_plan_hash": binding.source_logical_plan_hash,
                "compiled_plan_hash": compiled.plan_hash,
                "compiled_wave_count": len(compiled.waves),
                "compiled_bucket_order": list(compiled.metrics.get("bucket_order", [])),
                "prepared_plan_order_preserved": bool(compiled.metrics.get("prepared_plan_order_preserved", False)),
                "hint_edges_consumed": int(compiled.metrics.get("hint_edges_consumed", 0) or 0),
                "hint_match_rate": float(compiled.metrics.get("hint_match_rate", 0.0) or 0.0),
            }
        )
        self._increment_state_counter_map("shadow_plan_count_by_layer", str(parse_layer_id(layer_name)))
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="prepared_phase_plan_shadow",
            start_ns=start_ns,
            end_ns=end_ns,
            status="ok",
            wave_count=int(len(compiled.waves)),
            hint_edges_consumed=int(compiled.metrics.get("hint_edges_consumed", 0) or 0),
        )
    def _build_global_joint_plan_wire(self, *, prepared_plan: Any) -> GlobalJointPlanWire:
        logical_plan = getattr(prepared_plan, "logical_plan")
        canonical_edge_order: list[tuple[str, int, int]] = []
        wave_metadata: list[tuple[int, tuple[tuple[str, int, int], ...]]] = []
        per_peer_sequence_rows: list[str] = []
        for wave in getattr(logical_plan, "waves", ()):
            wave_edges: list[tuple[str, int, int]] = []
            for flow in getattr(wave, "flows", ()):
                edge = (str(flow.phase), int(flow.src_rank), int(flow.dst_rank))
                wave_edges.append(edge)
                canonical_edge_order.append(edge)
                per_peer_sequence_rows.append(
                    f"{getattr(prepared_plan, 'created_at_layer_id', '')}:{getattr(prepared_plan, 'applies_from_layer_id', '')}:"
                    f"{str(flow.phase)}:{int(flow.src_rank)}:{int(flow.dst_rank)}:{int(getattr(wave, 'wave_id', 0))}"
                )
            wave_metadata.append((int(getattr(wave, "wave_id", 0)), tuple(wave_edges)))
        per_peer_sequence_digest = stable_hash(per_peer_sequence_rows)
        return GlobalJointPlanWire(
            window_key=str(getattr(prepared_plan, "window_key", "")),
            policy_name=str(getattr(logical_plan, "policy_name", "")),
            safe_selected_policy=str(getattr(logical_plan, "policy_name", "")),
            prediction_digest=str(getattr(prepared_plan, "forecast_digest", "")),
            canonical_edge_order=tuple(canonical_edge_order),
            wave_metadata=tuple(wave_metadata),
            per_peer_sequence_digest=str(per_peer_sequence_digest),
        )
    def _agree_joint_plan_digest(self, *, layer_name: str, phase: str, prepared_plan: Any) -> dict[str, Any]:
        wire = self._build_global_joint_plan_wire(prepared_plan=prepared_plan)
        digest = str(wire.global_plan_digest)
        device = torch.device("cuda", self.local_rank) if (torch.cuda.is_available() and self.ep_process_group is not None) else torch.device("cpu")
        digest_value = int(digest[:16], 16)
        if digest_value >= (1 << 63):
            digest_value -= 1 << 64
        local = torch.tensor([digest_value], dtype=torch.long, device=device)
        gathered = [torch.empty_like(local) for _ in range(len(self.ep_group_ranks) or 1)]
        if len(gathered) > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_gather(gathered, local, group=self.ep_process_group)
        else:
            gathered = [local]
        gathered_values = [int(item.item()) for item in gathered]
        valid = len(set(gathered_values)) == 1
        agreement = {
            "valid": bool(valid),
            "global_plan_digest": digest,
            "gathered_plan_digests": [
                f"{int(value) & ((1 << 64) - 1):016x}"
                for value in gathered_values
            ],
            "per_peer_sequence_digest": str(wire.per_peer_sequence_digest),
            "window_key": str(wire.window_key),
            "policy_name": str(wire.policy_name),
        }
        self._runtime_state.write("global_joint_plan_wire", wire)
        self._runtime_state.write("global_joint_plan_agreement", agreement)
        self._timeline(
            "global_joint_plan_digest_agreed" if valid else "global_joint_plan_digest_mismatch",
            layer_name=layer_name,
            phase_name=phase,
            global_plan_digest=digest,
            per_peer_sequence_digest=str(wire.per_peer_sequence_digest),
        )
        return agreement
    def _build_formal_planning_request(
        self,
        *,
        request_id: str,
        source_layer_id: str,
        target_layer_id: str,
        p0_dispatch_rows: tuple[tuple[int, ...], ...],
        p1_return_rows: tuple[tuple[int, ...], ...],
        p2_hint_rows: tuple[tuple[int, ...], ...],
        predictor_name: str,
        prediction_confidence: float,
        information_mode: str = "p0_p1_p2",
        max_waves: int | None = None,
        planning_track: str = "runtime_lookahead",
        p2_semantics: str | None = None,
    ) -> PlanningRequest:
        effective_max_waves = int(max_waves if max_waves is not None else getattr(self.config, "max_waves", 256))
        effective_p2_semantics = (
            str(p2_semantics)
            if p2_semantics is not None
            else (
                "absent"
                if str(information_mode) in {"p0_only", "p0_p1"}
                else "advisory_hint"
                if str(planning_track) == "runtime_lookahead"
                else "executable_actual"
            )
        )
        return build_window_planning_request(
            identity=PlanningIdentity(
                request_id=str(request_id),
                run_id=str(self.run_id),
                forward_id=str(self._forward_epoch),
                window_id=f"{self._forward_epoch}:{self.microbatch_id}:{source_layer_id}",
                source_layer_id=str(source_layer_id),
                target_layer_id=str(target_layer_id),
            ),
            p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p0_dispatch_rows),
            p1_return_rows=tuple(tuple(int(v) for v in row) for row in p1_return_rows),
            p2_hint_rows=tuple(tuple(int(v) for v in row) for row in p2_hint_rows),
            predictor_id=str(predictor_name or "zero"),
            confidence=float(prediction_confidence),
            topology=PlanningTopology(world_size=int(len(p0_dispatch_rows)), full_duplex=True),
            constraints=PlanningConstraints(
                bucket_rows=int(self.config.bucket_rows),
                max_waves=int(effective_max_waves),
                expert_compute_delay=float(getattr(self.config, "expert_compute_delay", 0.0) or 0.0),
                phase_release_model="p1_return",
            ),
            weights=PlanningWeights(
                p0_weight=float(self.config.p0_weight),
                p1_weight=float(self.config.p1_reservation_weight),
                p2_weight=float(self.config.p2_hint_weight),
                p3_return_weight=float(getattr(self.config, "p3_return_weight", 0.0)),
                residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
                barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
                age_weight=float(getattr(self.config, "age_weight", 0.15)),
                prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
            ),
            information_mode=str(information_mode),
            planning_track=str(planning_track),
            p2_semantics=str(effective_p2_semantics),
            hint_type=(
                "perfect_trace_hint"
                if str(predictor_name) == "perfect_trace_hint"
                else "copy_current_dispatch"
                if str(predictor_name) == "copy_current_dispatch"
                else "learned_prediction"
            ),
            oracle=bool(str(predictor_name) == "perfect_trace_hint"),
        )
