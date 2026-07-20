"""Lifecycle Configuration stage methods."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleConfigurationMixin:
    def __post_init__(self) -> None:
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
        )
        self._runtime_state.set_invariant_mode(str(getattr(self.config, "invariant_mode", "diagnostic")))
        if self.observation_recorder is None:
            self.observation_recorder = RuntimeObservationRecorder(
                config=RuntimeObservationConfig(
                    profile=str(getattr(self.config, "observation_profile", "minimal")),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    capture_enabled=bool(getattr(self.config, "capture_phase_tensors", False)),
                    capture_expert_trace=bool(getattr(self.config, "capture_expert_trace", False)),
                    capture_layer_selector=str(getattr(self.config, "capture_layer_selector", "")),
                    capture_phase_selector=str(getattr(self.config, "capture_phase_selector", "")),
                    heartbeat_enabled=bool(getattr(self.config, "heartbeat_enabled", False)),
                    per_wave_timing_enabled=bool(getattr(self.config, "per_wave_timing_enabled", False)),
                    replay_trace_enabled=bool(getattr(self.config, "replay_trace_enabled", False)),
                )
            )
        if self.config.p2_hint_mode == "calibrated_artifact":
            self._p2_hint_provider = build_p2_hint_provider(
                self.config.p2_hint_mode,
                shared_state=self._runtime_state,
            )
        self._refresh_policy_caches()
        self._ensure_target_planner_runtime()

    def _refresh_policy_caches(self) -> None:
        resolved = resolve_online_policy_config(self.config)
        requested_planner_id = str(getattr(self.config, "planner_id", "") or "").strip()
        planner_spec = None
        if requested_planner_id:
            try:
                planner_spec = PlannerRegistry.resolve(requested_planner_id)
            except ValueError as exc:
                raise UnsupportedSchedulerMode(f"Unsupported planner_id={requested_planner_id!r}") from exc
            if not bool(planner_spec.deployable) or bool(planner_spec.reference_only):
                raise UnsupportedSchedulerMode(f"planner_id={requested_planner_id!r} is not runtime-deployable")
            requested_planner_id = str(planner_spec.planner_id)
        if resolved is None and planner_spec is None:
            self._effective_phase_policy_name_cache = ""
            self._effective_planner_id_cache = ""
            self._effective_planner_spec_cache = None
            self._resolved_policy_capabilities_cache = None
            self._joint_window_enabled_cache = False
            self._cross_layer_prediction_enabled_cache = False
            self._target_preplanning_enabled_cache = False
            return
        self._effective_phase_policy_name_cache = str(
            resolved.builder_key if resolved is not None else "prepared_priority"
        )
        self._effective_planner_id_cache = str(
            requested_planner_id
            or (resolved.builder_key if resolved is not None else "")
        )
        self._effective_planner_spec_cache = planner_spec
        if planner_spec is not None:
            planner_family = str(planner_spec.planner_family)
            uses_prediction = bool(planner_spec.requires_prediction)
            base = PolicyCapabilities(
                supports_offline=True,
                supports_online_phase_local_execution=True,
                supports_online_multiphase_execution=True,
                uses_current_ready_flows=True,
                uses_blocked_p1_dependency=True,
                uses_p2_forecast=uses_prediction,
                requires_fixed_placement=False,
                evaluation_eligible=True,
            )
        else:
            spec = resolved.spec
            scope = str(spec.scheduling_scope)
            execution_model = str(spec.execution_model)
            planner_family = str(spec.family)
            base = PolicyCapabilities(
                supports_offline=bool(spec.offline_eligible),
                supports_online_phase_local_execution=bool(spec.online_eligible and spec.phase_local_eligible),
                supports_online_multiphase_execution=bool(spec.online_eligible and ("joint" in scope or "multiphase" in scope or "global" in execution_model)),
                uses_current_ready_flows=True,
                uses_blocked_p1_dependency=bool("joint" in scope or "multiphase" in scope),
                uses_p2_forecast=bool(spec.supports_p2_hint),
                requires_fixed_placement=False,
                evaluation_eligible=bool(spec.offline_eligible),
            )
        predictor_name = self._online_p2_predictor_name()
        has_prediction = predictor_name not in {"none", "zero_hint"}
        is_joint_window = str(self.config.execution_mode) == "joint_window_async_p2p"
        safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
        planning_timing = self._planning_timing_mode()
        supports_target_preplanning = bool(
            has_prediction
            and is_joint_window
            and base.uses_p2_forecast
            and (safe_projection_mode == "disabled" or safe_projection_mode == "host_select")
            and planning_timing in {"legacy_auto", "previous_layer"}
        )
        self._resolved_policy_capabilities_cache = base.with_runtime_flags(
            supports_current_window_joint_planning=bool(
                is_joint_window and base.supports_online_multiphase_execution
            ),
            supports_cross_layer_prediction=bool(has_prediction and base.uses_p2_forecast),
            supports_two_horizon_prediction=bool(has_prediction and base.uses_p2_forecast),
            supports_target_layer_preplanning=bool(supports_target_preplanning),
            supports_p1_plan_reuse=bool(
                is_joint_window and base.supports_online_multiphase_execution
            ),
            supports_late_suffix_splice=False,
            supports_rank_release_batch=bool(is_joint_window),
        )
        self._joint_window_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_current_window_joint_planning
        )
        self._cross_layer_prediction_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_cross_layer_prediction
        )
        self._target_preplanning_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_target_layer_preplanning
        )
        self._runtime_state.write("effective_policy_name", str(self._effective_phase_policy_name_cache))
        self._runtime_state.write("effective_planner_id", str(self._effective_planner_id_cache))
        self._runtime_state.write("effective_planner_family", str(planner_family))
        self._runtime_state.write("requested_preflight_mode", str(getattr(self.config, "preflight_mode", "full")))
        self._runtime_state.write("effective_preflight_mode", str(getattr(self.config, "preflight_mode", "full")))

    def configure_hook_scope(self, *, available_layer_names: tuple[str, ...]) -> None:
        available_layer_ids: list[str] = []
        for layer_name in available_layer_names:
            layer_id = str(parse_layer_id(layer_name))
            if layer_id not in available_layer_ids:
                available_layer_ids.append(layer_id)
        self._available_moe_layer_ids = tuple(available_layer_ids)
        resolved = resolve_layer_selector(
            str(self.config.schedule_layer_selector),
            selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
            available_layer_ids=self._available_moe_layer_ids,
            invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
        )
        self._resolved_schedule_selector = resolved
        if resolved.matches_all:
            selected = frozenset(self._available_moe_layer_ids)
        else:
            selected = frozenset(str(item) for item in resolved.resolved_layer_ids)
        prediction_source: set[str] = set()
        if self._target_preplanning_enabled_cache:
            for layer_id in selected:
                if str(layer_id).isdigit() and int(layer_id) > 0:
                    candidate = str(int(layer_id) - 1)
                    if candidate not in selected:
                        prediction_source.add(candidate)
        self._selected_layer_id_set = selected
        self._prediction_source_layer_id_set = frozenset(prediction_source)
        self._none_layer_id_set = frozenset(
            layer_id
            for layer_id in self._available_moe_layer_ids
            if layer_id not in self._selected_layer_id_set and layer_id not in self._prediction_source_layer_id_set
        )
        self._runtime_state.write("total_model_moe_layers", int(len(self._available_moe_layer_ids)))
        self._runtime_state.write("selected_layer_ids", stable_layer_ids(self._selected_layer_id_set))
        self._runtime_state.write("prediction_source_layer_ids", stable_layer_ids(self._prediction_source_layer_id_set))
        self._runtime_state.write("none_layer_ids", stable_layer_ids(self._none_layer_id_set))
        self._runtime_state.write("wrapped_selected_layer_ids", stable_layer_ids(self._selected_layer_id_set))
        self._runtime_state.write("wrapped_prediction_source_layer_ids", stable_layer_ids(self._prediction_source_layer_id_set))
        self._runtime_state.write("unwrapped_none_layer_ids", stable_layer_ids(self._none_layer_id_set))

    def layer_role_for_name(self, layer_name: str) -> str:
        layer_id = str(parse_layer_id(layer_name))
        if self._resolved_schedule_selector is None:
            fallback_selector = resolve_layer_selector(
                str(self.config.schedule_layer_selector),
                selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
                invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
            )
            if fallback_selector.matches_all or layer_selected(layer_id, selector=fallback_selector):
                self._selected_layer_matches_seen.add(layer_id)
                self._runtime_state.metrics.selected_layer_match_count = int(len(self._selected_layer_matches_seen))
                return "selected"
            return "none"
        if layer_id in self._selected_layer_id_set:
            self._selected_layer_matches_seen.add(layer_id)
            self._runtime_state.metrics.selected_layer_match_count = int(len(self._selected_layer_matches_seen))
            return "selected"
        if layer_id in self._prediction_source_layer_id_set:
            return "prediction_source"
        return "none"

    def _layer_id_selected(self, layer_id: str) -> bool:
        normalized = str(layer_id)
        if self._resolved_schedule_selector is None:
            fallback_selector = resolve_layer_selector(
                str(self.config.schedule_layer_selector),
                selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
                invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
            )
            return bool(fallback_selector.matches_all or layer_selected(normalized, selector=fallback_selector))
        return normalized in self._selected_layer_id_set

    @property
    def _prepared_plan_state(self) -> PreparedWindowRuntimeState:
        return self._runtime_state

    def _artifact_profile(self) -> str:
        return str(getattr(self.config, "observation_profile", "minimal"))

    def _is_perf_profile(self) -> bool:
        return self._artifact_profile() in {"perf", "timeline_light", "attribution_light"}

    def _is_debug_profile(self) -> bool:
        return self._artifact_profile() == "debug"

    def _allow_shadow_artifacts(self) -> bool:
        return not self._is_perf_profile()

    def _replay_trace_enabled(self) -> bool:
        return bool(getattr(self.config, "replay_trace_enabled", False))

    def _record_control_replay_trace(self, *, phase_ctx: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        if not self._replay_trace_enabled():
            return
        self.control_replay_traces.append(
            control_replay_trace_row(
                run_id=self.run_id,
                ep_group_size=int(len(self.ep_group_ranks) or 1),
                bucket_rows=int(self.config.bucket_rows),
                phase_ctx=phase_ctx,
                plan=plan,
            )
        )

    def _effective_phase_policy_name(self) -> str:
        return str(self._effective_phase_policy_name_cache)

    def _effective_planner_id(self) -> str:
        return str(self._effective_planner_id_cache or self._effective_phase_policy_name_cache)

    def _current_window_planner_id(self) -> str:
        planner_id = self._effective_planner_id()
        if is_axes_planner_id(planner_id):
            axes = parse_planner_axes(planner_id)
            if axes.timing == "future":
                return axes.with_timing("current").canonical_id
        return planner_id

    def _target_window_planner_id(self) -> str:
        return self._effective_planner_id()

    def _phase_policy(self):
        phase_policy_name = self._effective_phase_policy_name()
        if phase_policy_name:
            return resolve_phase_policy(
                policy_name=phase_policy_name,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                p2_hint_artifact=self.config.p2_hint_artifact,
            )
        if self.config.scheduler_mode == "native_passthrough_identity":
            return NativePassthroughIdentityPolicy()
        if self.config.scheduler_mode == "native_order":
            return NativeOrderPolicy()
        if self.config.scheduler_mode == "joint_shadow_p0p1":
            return JointShadowP0P1Policy()
        raise UnsupportedSchedulerMode(f"Unsupported scheduler_mode={self.config.scheduler_mode!r}")

    def _layer_selected(self, layer_name: str) -> bool:
        return self.layer_role_for_name(layer_name) == "selected"

    def _layer_is_prediction_source(self, layer_name: str) -> bool:
        return self.layer_role_for_name(layer_name) == "prediction_source"

    def _phase_selected(self, phase: str) -> bool:
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return True
        return selector == str(phase).lower()

    def _should_schedule_phase(self, *, layer_name: str, phase: str) -> bool:
        return (
            bool(self._effective_phase_policy_name_cache)
            and self.config.execution_mode in {"phase_sync_wave", "joint_window_async_p2p"}
            and self.config.control_mode == "sync_before_phase"
            and self.layer_role_for_name(layer_name) == "selected"
            and self._phase_selected(phase)
        )

    def _is_joint_window_async_mode(self) -> bool:
        return bool(self._joint_window_enabled_cache)

    def _runtime_safe_scope_pair(self, planner_id: str | None = None) -> tuple[str, str]:
        policy_name = str(planner_id or self._effective_planner_id() or self.config.policy or "")
        if is_axes_planner_id(policy_name):
            axes = parse_planner_axes(policy_name)
            if axes.scope == "local":
                return (policy_name, policy_name)
            # Safe pairing changes only scope. Timing, horizon, engine and core
            # remain identical, which is the strict same-core comparison.
            return (policy_name, axes.with_scope("local").canonical_id)
        raise UnsupportedSchedulerMode(
            f"safe pairing requires an explicit orthogonal planner id, got {policy_name!r}"
        )

    def _effective_bucket_mode(self) -> str:
        return bucket_mode_for_rows(int(self.config.bucket_rows))

    def _requested_bucket_mode(self) -> str:
        requested = str(getattr(self.config, "bucket_mode", "") or "").strip()
        if requested:
            return requested
        return self._effective_bucket_mode()

    def _assert_bucket_mode_consistency(self) -> None:
        requested = self._requested_bucket_mode()
        effective = self._effective_bucket_mode()
        if requested != effective:
            raise RuntimeError(
                f"bucket mode mismatch: requested={requested!r} effective={effective!r} "
                f"bucket_rows={int(self.config.bucket_rows)}"
            )

    def _should_stop_after_layer(self, *, layer_name: str, phase: str) -> bool:
        if not (
            self.config.stop_after_selected_layer
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
        ):
            return False
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return str(phase).upper() == "P1"
        return True


__all__ = ["LifecycleConfigurationMixin"]
