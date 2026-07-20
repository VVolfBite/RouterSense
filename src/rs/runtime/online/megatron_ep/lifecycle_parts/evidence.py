"""Lifecycle Evidence stage methods."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleEvidenceMixin:
    def _activate_transport(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        start_ns = time.monotonic_ns()
        effective_preflight_mode = str(getattr(self.config, "preflight_mode", "full") or "full")
        plan_metrics = dict(plan.metrics or {})
        plan_preflight_mode = str(plan_metrics.get("preflight_mode", "") or "")
        if plan_preflight_mode and plan_preflight_mode != effective_preflight_mode:
            raise RuntimeError(
                f"preflight mode mismatch before transport activation: "
                f"plan={plan_preflight_mode!r} effective={effective_preflight_mode!r}"
            )
        if plan_preflight_mode != effective_preflight_mode:
            plan = replace(plan, metrics={**plan_metrics, "preflight_mode": effective_preflight_mode})
        if self._layer_selected(layer_name):
            self._runtime_state.metrics.selected_transport_execution_count = int(
                self._runtime_state.metrics.selected_transport_execution_count
            ) + 1
        prepared_execution = None
        if self._is_joint_window_async_mode():
            prepared_execution = self._prepared_execution_cache().get(self.target_plan_store._key(self._target_plan_key(layer_name=layer_name))) if self.target_plan_store is not None else None
        self._active_transport = {
            "layer_name": layer_name,
            "phase": phase,
            "context": context,
            "plan": plan,
            "prepared_execution": prepared_execution,
        }
        if self._layer_selected(layer_name):
            self.expected_evidence.expect_phase_payload_roles(
                layer_id=str(context.layer_id),
                phase=str(phase),
                payload_roles=self._required_payload_roles_for_phase(str(phase)),
            )
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            if hasattr(adapter, "set_effective_preflight_mode"):
                adapter.set_effective_preflight_mode(effective_preflight_mode)
            adapter.activate(
                layer_name=layer_name,
                phase=phase,
                context=context,
                plan=plan,
                prepared_execution=prepared_execution,
                execution_pipeline=getattr(self, "execution_pipeline", None),
                runtime=self,
            )
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="activate_transport",
            start_ns=start_ns,
            end_ns=end_ns,
            wave_count=int(len(plan.waves)),
            bucket_count=int(sum(len(wave.bucket_tasks) for wave in plan.waves)),
        )

    def current_transport(self) -> dict[str, Any] | None:
        return self._active_transport

    def _execution_plan_cache(self) -> dict[tuple[str, int, str, str], Any]:
        cache = getattr(self, "_published_execution_plans", None)
        if cache is None:
            cache = {}
            setattr(self, "_published_execution_plans", cache)
        return cache

    def _prepared_execution_cache(self) -> dict[tuple[str, int, str, str], Any]:
        cache = getattr(self, "_prepared_executions", None)
        if cache is None:
            cache = {}
            setattr(self, "_prepared_executions", cache)
        return cache

    def _cleanup_execution_caches_for_generation(
        self,
        *,
        before_generation: int | None = None,
        exact_generation: int | None = None,
    ) -> None:
        def _filter(cache: dict[tuple[str, int, str, str], Any]) -> None:
            doomed: list[tuple[str, int, str, str]] = []
            for key in list(cache.keys()):
                generation = int(key[1])
                if before_generation is not None and generation < int(before_generation):
                    doomed.append(key)
                elif exact_generation is not None and generation == int(exact_generation):
                    doomed.append(key)
            for key in doomed:
                cache.pop(key, None)

        _filter(self._execution_plan_cache())
        _filter(self._prepared_execution_cache())

    def _local_group_rank(self) -> int:
        ranks = tuple(int(value) for value in self.ep_group_ranks)
        return int(ranks.index(int(self.rank))) if int(self.rank) in ranks else 0

    def _required_payload_roles_for_phase(self, phase: str) -> tuple[str, ...]:
        if str(phase) == "P0":
            return ("hidden_states", "routing_probs")
        if str(phase) == "P1":
            return ("hidden_states",)
        return ()

    def record_phase_payload_completion(
        self,
        *,
        layer_id: str,
        phase: str,
        payload_role: str,
    ) -> tuple[str, ...]:
        releases = self.release_state_ledger.record_payload_completion(
            layer_id=str(layer_id),
            phase=str(phase),
            local_group_rank=self._local_group_rank(),
            payload_role=str(payload_role),
            required_payload_roles=self._required_payload_roles_for_phase(str(phase)),
        )
        if not releases:
            return ()
        self.release_state_ledger.satisfied_release_ids.update(str(item) for item in releases)
        return tuple(str(item) for item in releases)

    def satisfied_release_dependency_ids_for(
        self,
        *,
        layer_id: str,
        phase: str,
        local_group_rank: int | None = None,
    ) -> tuple[str, ...]:
        layer_id = str(layer_id)
        phase = str(phase)
        rank = self._local_group_rank() if local_group_rank is None else int(local_group_rank)
        if phase == "P1":
            prefix = f"release:{layer_id}:p0_inbound_complete:{rank}"
        elif phase == "P2":
            prefix = f"release:{layer_id}:p1_inbound_complete:{rank}"
        else:
            return ()
        return tuple(
            sorted(
                str(item)
                for item in self.release_state_ledger.satisfied_release_ids
                if str(item) == prefix
            )
        )

    def record_execution_outcome(
        self,
        *,
        layer_id: str,
        phase: str,
        payload_role: str,
        outcome: dict[str, object],
    ) -> None:
        self._latest_execution_outcomes.append(
            {
                "layer_id": str(layer_id),
                "phase": str(phase),
                "payload_role": str(payload_role),
                "outcome": dict(outcome),
                "release_ids": list(sorted(self.release_state_ledger.satisfied_release_ids)),
            }
        )

    def _finalize_result_bundle(self) -> ResultBundle | None:
        instrumentation = getattr(self, "runtime_instrumentation", None)
        if instrumentation is None:
            return None
        measurement_sink = getattr(instrumentation, "measurement_sink", None)
        measurement_snapshot = measurement_sink.snapshot() if measurement_sink is not None and hasattr(measurement_sink, "snapshot") else None
        measurement_complete = bool(getattr(measurement_snapshot, "completeness", None) and measurement_snapshot.completeness.complete)
        mode = str(getattr(self, "_instrumentation_mode", "off") or "off")
        commit_sha = str(getattr(self, "_commit_sha", "") or "")
        git_clean = bool(getattr(self, "_git_clean", False))
        outcomes = list(self._latest_execution_outcomes)
        expected_evidence = self.expected_evidence
        formal_execution_expected = int(expected_evidence.expected_execution_count) > 0
        observed_role_map: dict[tuple[str, str], set[str]] = {}
        observed_layers: set[str] = set()
        for item in outcomes:
            layer_id = str(item.get("layer_id", ""))
            phase = str(item.get("phase", ""))
            payload_role = str(item.get("payload_role", ""))
            if layer_id:
                observed_layers.add(layer_id)
            if layer_id and phase and payload_role:
                observed_role_map.setdefault((layer_id, phase), set()).add(payload_role)
        missing_expected_roles: list[str] = []
        for key, required_roles in expected_evidence.expected_phase_payload_roles.items():
            observed_roles = observed_role_map.get(key, set())
            for payload_role in sorted(required_roles):
                if payload_role not in observed_roles:
                    missing_expected_roles.append(f"{key[0]}:{key[1]}:{payload_role}")
        missing_selected_layers = sorted(
            layer_id for layer_id in expected_evidence.selected_layers if layer_id not in observed_layers
        )
        missing_execution_outcome_count = max(
            0,
            int(expected_evidence.expected_execution_count) - int(len(outcomes)),
        )
        runtime_failure_reason = str(getattr(self, "_runtime_failure_reason", "") or "")
        if runtime_failure_reason:
            all_work_completed = False
            correctness_status = "invalid"
            status = "failure"
            failure_reason = runtime_failure_reason
        elif formal_execution_expected and (
            not outcomes
            or missing_execution_outcome_count > 0
            or missing_expected_roles
            or missing_selected_layers
        ):
            all_work_completed = False
            correctness_status = "invalid"
            status = "failure"
            if not outcomes:
                failure_reason = "missing_execution_outcomes"
            elif missing_expected_roles:
                failure_reason = "missing_expected_payload_roles"
            elif missing_selected_layers:
                failure_reason = "missing_selected_layers"
            else:
                failure_reason = "missing_execution_outcomes"
        elif outcomes:
            all_work_completed = all(
                bool(dict(item.get("outcome", {})).get("success", False))
                and bool(dict(item.get("outcome", {})).get("all_work_completed", False))
                and not tuple(dict(item.get("outcome", {})).get("unresolved_task_ids", ()))
                for item in outcomes
            )
            correctness_status = "valid" if all_work_completed else "invalid"
            status = "success" if all_work_completed else "failure"
            failure_reason = "" if all_work_completed else "execution_incomplete"
        else:
            all_work_completed = True
            correctness_status = "valid"
            status = "success"
            failure_reason = ""
        summary = {
            "formal_execution_expected": bool(formal_execution_expected),
            "expected_execution_count": int(expected_evidence.expected_execution_count),
            "all_work_completed": bool(all_work_completed),
            "fallback_count": int(self.evidence_counters.fallback_count),
            "timeout_count": int(self.evidence_counters.timeout_count),
            "check_failure_count": int(self.evidence_counters.check_failure_count),
            "preparation_miss_count": int(self.evidence_counters.preparation_miss_count),
            "provisional_execution_count": int(self.evidence_counters.provisional_execution_count),
            "materialization_failure_count": int(self.evidence_counters.materialization_failure_count),
            "execution_failure_count": int(self.evidence_counters.execution_failure_count),
            "cleanup_failure_count": int(self.evidence_counters.cleanup_failure_count),
            "execution_outcome_count": int(len(outcomes)),
            "missing_execution_outcome_count": int(missing_execution_outcome_count),
            "missing_expected_payload_role_count": int(len(missing_expected_roles)),
            "missing_selected_layer_count": int(len(missing_selected_layers)),
            "release_id_count": int(len(self.release_state_ledger.satisfied_release_ids)),
            "measurement_event_count": int(getattr(measurement_snapshot, "event_count", 0) or 0) if measurement_snapshot is not None else 0,
            "measurement_unknown_event_count": int(getattr(measurement_snapshot, "unknown_event_count", 0) or 0) if measurement_snapshot is not None else 0,
            "measurement_malformed_event_count": int(getattr(measurement_snapshot, "malformed_event_count", 0) or 0) if measurement_snapshot is not None else 0,
            "measurement_dropped_event_count": int(getattr(measurement_snapshot, "dropped_event_count", 0) or 0) if measurement_snapshot is not None else 0,
        }
        bundle = build_result_bundle(
            ResultBundleDraft(
                run_identity=RunIdentity(
                    run_id=str(self.run_id),
                    pipeline="online",
                    claim_scope="formal",
                    trace_origin="runtime",
                    future_information_mode=str(getattr(self.config, "future_hint_mode", "runtime")),
                ),
                status=status,
                correctness_status=correctness_status,
                performance_status="unknown",
                commit_sha=commit_sha or "unknown",
                git_clean=bool(git_clean),
                instrumentation_mode=mode,
                audit_evidence_level="summary_only",
                measurement_complete=bool(measurement_complete),
                summary=summary,
                details={
                    "run_kind": "GLOO_FUNCTIONAL",
                    "latest_execution_outcomes": outcomes,
                    "measurement_summary": {} if measurement_snapshot is None else dict(measurement_snapshot.summary),
                    "measurement_capability": None if measurement_snapshot is None else measurement_snapshot.capability.to_dict(),
                    "measurement_completeness": None if measurement_snapshot is None else measurement_snapshot.completeness.to_dict(),
                    "failure_reason": str(failure_reason),
                    "expected_evidence": {
                        "claim_scope": str(expected_evidence.claim_scope),
                        "selected_layers": list(sorted(expected_evidence.selected_layers)),
                        "expected_execution_count": int(expected_evidence.expected_execution_count),
                        "expected_phase_payload_roles": {
                            f"{layer_id}:{phase}": list(sorted(payload_roles))
                            for (layer_id, phase), payload_roles in sorted(expected_evidence.expected_phase_payload_roles.items())
                        },
                        "measurement_required": bool(expected_evidence.measurement_required),
                    },
                    "missing_expected_payload_roles": list(missing_expected_roles),
                    "missing_selected_layers": list(missing_selected_layers),
                    "commit_sha_source": str(getattr(self, "_commit_sha_source", "") or ""),
                },
                extensions={},
            )
        )
        self._latest_result_bundle = bundle
        instrumentation.record_result(bundle)
        return bundle

    def _record_instrumentation_measurement(
        self,
        *,
        event_type: str,
        layer_id: str | None,
        phase: str | None,
        started_at_ns: int,
        ended_at_ns: int,
        details: dict[str, object] | None = None,
    ) -> None:
        instrumentation = getattr(self, "runtime_instrumentation", None)
        if instrumentation is None:
            return
        instrumentation.record_measurement(
            MeasurementEvent(
                run_id=str(self.run_id),
                rank=int(getattr(self, "rank", 0)),
                forward_generation=int(getattr(self, "_current_forward_epoch", 0) or 0),
                microbatch_id=str(getattr(self, "microbatch_id", "global")),
                event_type=str(event_type),
                started_at_ns=int(started_at_ns),
                ended_at_ns=int(ended_at_ns),
                layer_id=None if layer_id is None else str(layer_id),
                phase=None if phase is None else str(phase),
                details=dict(details or {}),
            )
        )

    def _actual_phase_context_from_ready_context(self, *, phase_ctx: PhaseReadyContext) -> ActualPhaseContext:
        return ActualPhaseContext(
            layer_id=str(phase_ctx.layer_id),
            phase=str(phase_ctx.phase),
            world_size=int(len(phase_ctx.ep_group_ranks)),
            rank_space="global",
            layout_digest=str(phase_ctx.canonical_receive_layout_id),
            metadata={"phase_ready_context": phase_ctx.to_dict()},
        )

    def clear_transport(self, *, layer_name: str, phase: str) -> None:
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            adapter.deactivate(layer_name=layer_name, phase=phase)
        if self._active_transport is None:
            return
        if self._active_transport.get("layer_name") == layer_name and self._active_transport.get("phase") == phase:
            self._active_transport = None

    def record_transport_execution(self, payload: dict[str, Any]) -> None:
        if self.observation_recorder is not None:
            self.observation_recorder.record_transport_execution(dict(payload))

    def _append_heartbeat(self, payload: dict[str, Any]) -> None:
        if self._is_perf_profile():
            return
        if not self.config.executor_heartbeat_path:
            return
        heartbeat_dir = Path(self.config.executor_heartbeat_path)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        target = heartbeat_dir / f"heartbeat-rank{self.rank}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()

    def _timeline(self, event: str, *, layer_name: str, **detail: Any) -> None:
        if self._is_perf_profile():
            return
        row = {
            "ts_us": int(time.time() * 1e6),
            "monotonic_ns": time.monotonic_ns(),
            "event_seq": len(self.control_timeline) + 1,
            "event": event,
            "run_id": self.run_id,
            "forward_epoch": int(self._forward_epoch),
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": parse_layer_id(layer_name),
            "phase": "P0" if ("p0" in event or "dispatch" in event) else "P1" if ("p1" in event or "combine" in event) else "control",
            "layer": layer_name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "control_mode": self.config.control_mode,
            "scheduler_mode": self.config.scheduler_mode,
            **detail,
        }
        self.control_timeline.append(row)
        if event in {
            "before_phase_plan",
            "after_phase_plan",
            "before_wave",
            "after_wave",
            "before_payload_collective",
            "after_payload_collective",
            "after_phase",
            "p0_pre_transport_observation_ready",
            "p1_pre_transport_observation_ready",
            "p0_native_dispatch_committed",
        }:
            self._append_heartbeat(row)
            if self.observation_recorder is not None:
                self.observation_recorder.record_heartbeat(row)

    def _record_planning_timing(
        self,
        *,
        layer_name: str,
        phase: str,
        stage: str,
        start_ns: int,
        end_ns: int,
        **detail: Any,
    ) -> float:
        duration_us = max(0.0, float(end_ns - start_ns) / 1000.0)
        if str(stage).startswith("materialize"):
            measurement_event_type = "materialization"
        elif str(stage).startswith("validate"):
            measurement_event_type = "validation"
        elif "publish" in str(stage):
            measurement_event_type = "publish"
        elif "executor" in str(stage) or "transport" in str(stage):
            measurement_event_type = "active_transport"
        else:
            measurement_event_type = "planning"
        self._record_instrumentation_measurement(
            event_type=measurement_event_type,
            layer_id=parse_layer_id(layer_name),
            phase=str(phase),
            started_at_ns=int(start_ns),
            ended_at_ns=int(end_ns),
            details={"stage": str(stage), **detail},
        )
        if self._is_perf_profile():
            counter = self.perf_counters.setdefault(
                str(stage),
                {"count": 0.0, "total_us": 0.0, "max_us": 0.0},
            )
            counter["count"] += 1.0
            counter["total_us"] += float(duration_us)
            counter["max_us"] = max(float(counter["max_us"]), float(duration_us))
            return duration_us
        record = {
            "ts_us": int(time.time() * 1e6),
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "stage": stage,
            "duration_us": duration_us,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "execution_mode": self.config.execution_mode,
            "control_mode": self.config.control_mode,
            **detail,
        }
        self.planning_timing_records.append(record)
        self._timeline(
            "planning_stage_timing",
            layer_name=layer_name,
            phase_name=phase,
            stage=stage,
            duration_us=duration_us,
            **detail,
        )
        return duration_us

    def _record_hook_timing(
        self,
        *,
        layer_name: str,
        phase: str,
        hook_name: str,
        start_ns: int,
        end_ns: int,
        **detail: Any,
    ) -> float:
        return self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage=f"hook_{hook_name}",
            start_ns=start_ns,
            end_ns=end_ns,
            **detail,
        )

    def _increment_state_counter_map(self, key: str, item: str) -> None:
        payload = dict(self._runtime_state.read(key, {}) or {})
        payload[str(item)] = int(payload.get(str(item), 0) or 0) + 1
        self._runtime_state.write(key, payload)

    def _register_current_plan_build(self, *, layer_name: str, phase: str, plan_origin: str) -> None:
        build_key = (
            int(self.rank),
            int(self._forward_epoch),
            str(parse_layer_id(layer_name)),
            str(phase),
            str(plan_origin),
        )
        if build_key in self._current_plan_build_keys:
            raise RuntimeError(f"duplicate current plan build detected for {build_key}")
        self._current_plan_build_keys.add(build_key)

    def _record_none_heavy_hook(self, *, layer_name: str, phase: str, hook_name: str, start_ns: int) -> None:
        self._runtime_state.metrics.none_heavy_hook_count = int(self._runtime_state.metrics.none_heavy_hook_count) + 1
        end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase=phase,
            hook_name=hook_name,
            start_ns=start_ns,
            end_ns=end_ns,
            scheduled=False,
            reason="layer_role_none_defensive_entry",
        )
        self._timeline(
            f"{hook_name}_none_heavy_defensive_exit",
            layer_name=layer_name,
            phase_name=phase,
            scheduled=False,
        )

    def _hook_execution_mode(self, *, layer_name: str) -> HookExecutionMode:
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            return "DISABLED"
        if layer_role == "prediction_source":
            return "OBSERVATION_ONLY"
        if layer_role != "selected":
            return "DISABLED"
        if self.config.scheduler_mode in {"native_passthrough_identity", "native_order", "joint_shadow_p0p1"}:
            return "LEGACY_SHADOW"
        if str(self.config.execution_mode) in {"joint_window_async_p2p", "phase_sync_wave"}:
            return "REAL_EXECUTION_WITH_OBSERVATION"
        if self.config.scheduler_mode == "disabled":
            return "OBSERVATION_ONLY"
        return "OBSERVATION_ONLY"

    def _record_dtoh_callsite(
        self,
        *,
        callsite_id: str,
        start_ns: int,
        end_ns: int,
        bytes_if_known: int | None = None,
    ) -> None:
        count_map = dict(self._runtime_state.read("dtoh_callsite_count", {}) or {})
        wall_map = dict(self._runtime_state.read("dtoh_callsite_wall_us", {}) or {})
        byte_map = dict(self._runtime_state.read("dtoh_callsite_bytes", {}) or {})
        count_map[str(callsite_id)] = int(count_map.get(str(callsite_id), 0) or 0) + 1
        wall_map[str(callsite_id)] = float(wall_map.get(str(callsite_id), 0.0) or 0.0) + max(
            0.0, float(end_ns - start_ns) / 1000.0
        )
        if bytes_if_known is not None:
            byte_map[str(callsite_id)] = int(byte_map.get(str(callsite_id), 0) or 0) + int(bytes_if_known)
        self._runtime_state.write("dtoh_callsite_count", count_map)
        self._runtime_state.write("dtoh_callsite_wall_us", wall_map)
        self._runtime_state.write("dtoh_callsite_bytes", byte_map)

    def _synchronize_dispatcher_tokens_or_raise(
        self,
        *,
        dispatcher: Any,
        callsite_id: str,
    ) -> None:
        sync_fn = getattr(dispatcher, "_maybe_dtoh_and_synchronize", None)
        if not callable(sync_fn):
            return
        tokens_per_expert = getattr(dispatcher, "tokens_per_expert", None)
        dtoh_start_ns = time.monotonic_ns()
        try:
            synchronized = sync_fn("before_ep_alltoall", tokens_per_expert)
        except Exception as exc:
            dtoh_end_ns = time.monotonic_ns()
            self._record_dtoh_callsite(
                callsite_id=str(callsite_id),
                start_ns=dtoh_start_ns,
                end_ns=dtoh_end_ns,
            )
            self.evidence_counters.check_failure_count += 1
            raise DispatcherSynchronizationError(
                f"dispatcher synchronization failed at {callsite_id}: {type(exc).__name__}: {exc}"
            ) from exc
        dtoh_end_ns = time.monotonic_ns()
        self._record_dtoh_callsite(
            callsite_id=str(callsite_id),
            start_ns=dtoh_start_ns,
            end_ns=dtoh_end_ns,
        )
        if synchronized is not None:
            dispatcher.tokens_per_expert = synchronized

    def _finalize_dispatch_observation(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        self._runtime_state.metrics.observation_finalize_dispatch_count = int(
            self._runtime_state.metrics.observation_finalize_dispatch_count
        ) + 1
        self._runtime_state.write(
            "dispatch_finalize_shape",
            list(hidden_states.shape) if isinstance(hidden_states, torch.Tensor) else None,
        )
        self._runtime_state.write(
            "dispatch_finalize_dispatcher",
            str(type(dispatcher).__name__) if dispatcher is not None else "",
        )

    def _finalize_combine_observation(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        self._runtime_state.metrics.observation_finalize_combine_count = int(
            self._runtime_state.metrics.observation_finalize_combine_count
        ) + 1
        self._runtime_state.write(
            "combine_finalize_shape",
            list(hidden_states.shape) if isinstance(hidden_states, torch.Tensor) else None,
        )
        self._runtime_state.write(
            "combine_finalize_dispatcher",
            str(type(dispatcher).__name__) if dispatcher is not None else "",
        )

    def before_prediction_source_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        hook_start_ns = time.monotonic_ns()
        if self.layer_role_for_name(layer_name) != "prediction_source":
            return
        self._runtime_state.metrics.prediction_source_p0_hook_count = int(
            self._runtime_state.metrics.prediction_source_p0_hook_count
        ) + 1
        self._synchronize_dispatcher_tokens_or_raise(
            dispatcher=dispatcher,
            callsite_id="DTOH_P0_DISPATCHER_SYNC",
        )
        phase_ctx = self._build_phase_ready_context_from_dispatcher(
            layer_name=layer_name,
            phase="P0",
            dispatcher=dispatcher,
            packed_tensors=tuple(
                tensor for tensor in (packed_hidden_states, packed_probs) if isinstance(tensor, torch.Tensor)
            ),
        )
        pretransport = self._capture_pretransport_traffic_observation(phase_ctx=phase_ctx)
        actual_p0_full_row_matrix = self._gather_actual_p0_full_row_matrix(
            layer_name=layer_name,
            observation=pretransport,
            device=self._matrix_device(packed_hidden_states),
        )
        if self._should_generate_runtime_prediction():
            self._record_prediction_for_dispatch(
                layer_name=layer_name,
                phase_ctx=phase_ctx,
                observation=pretransport,
                actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                device=self._matrix_device(packed_hidden_states),
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="before_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="prediction_source_only",
        )

    def _matrix_device(self, candidate: Any) -> torch.device:
        if isinstance(candidate, torch.Tensor):
            return candidate.device
        return torch.device("cpu")

    def _runtime_topology_dict(self) -> dict[str, Any]:
        return {
            "global_rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "node_index": -1,
            "hostname_digest": digest_text(self.hostname),
            "device_index": int(self.local_rank),
            "ep_group_rank": int(tuple(int(v) for v in self.ep_group_ranks).index(int(self.rank))) if int(self.rank) in tuple(int(v) for v in self.ep_group_ranks) else 0,
        }

    def _dispatcher_expert_placement_hash(self, dispatcher: Any) -> str:
        return digest_text(
            stable_hash(
                {
                    "placement_mode": "megatron_native_ep",
                    "ep_group_ranks": list(int(v) for v in self.ep_group_ranks),
                    "ep_group_size": len(self.ep_group_ranks),
                    "dispatcher_class": type(dispatcher).__name__,
                }
            )
        )

    def _build_phase_ready_context_from_dispatcher(
        self,
        *,
        layer_name: str,
        phase: str,
        dispatcher: Any,
        packed_tensors: tuple[torch.Tensor, ...],
        p2_hint: Any | None = None,
    ) -> PhaseReadyContext:
        return build_phase_ready_context(
            PhaseContextBuildRequest(
                plan_key=self._plan_key(layer_name, phase),
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
                topology=self._runtime_topology_dict(),
                dispatcher_snapshot=DispatcherSnapshot(
                    dispatcher_class=type(dispatcher).__name__,
                    dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
                    expert_placement_hash=self._dispatcher_expert_placement_hash(dispatcher),
                    input_splits=tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None))[: len(self.ep_group_ranks)]),
                    output_splits=tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None))[: len(self.ep_group_ranks)]),
                ),
                payload_contract=PhasePayloadContract(
                    phase=phase,
                    payload_roles=("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",),
                    atomic_submit=(phase == "P0"),
                ),
                packed_tensors=packed_tensors,
                control_mode=self.config.control_mode,
                release_state="ready",
                demand_known_at="router_ready",
                payload_exists=True,
                p2_hint=p2_hint,
            )
        )

    def _capture_pretransport_traffic_observation(
        self,
        *,
        phase_ctx: PhaseReadyContext,
    ) -> PreTransportTrafficObservation:
        group_ranks = tuple(int(v) for v in phase_ctx.ep_group_ranks)
        group_rank = group_ranks.index(int(phase_ctx.global_rank)) if int(phase_ctx.global_rank) in group_ranks else 0
        send_splits_rows = tuple(int(v) for v in phase_ctx.send_splits)
        recv_splits_rows = tuple(int(v) for v in phase_ctx.recv_splits)
        valid = str(phase_ctx.phase) == "P0" and len(send_splits_rows) == len(group_ranks) and len(recv_splits_rows) == len(group_ranks)
        error = None if valid else "invalid_phase_or_split_shape"
        return PreTransportTrafficObservation(
            run_id=str(self.run_id),
            forward_epoch=int(phase_ctx.forward_epoch),
            microbatch_id=str(self.microbatch_id),
            layer_id=int(parse_layer_id(phase_ctx.layer_name)) if str(parse_layer_id(phase_ctx.layer_name)).isdigit() else -1,
            phase=str(phase_ctx.phase),
            global_rank=int(phase_ctx.global_rank),
            group_rank=int(group_rank),
            group_global_ranks=group_ranks,
            send_splits_rows=send_splits_rows,
            recv_splits_rows=recv_splits_rows,
            local_p0_row=send_splits_rows,
            local_send_rows=int(sum(send_splits_rows)),
            local_recv_rows=int(sum(recv_splits_rows)),
            source="phase_ready_context_dispatcher_splits",
            captured_before_transport=True,
            valid=bool(valid),
            error=error,
        )

    def _bundle_bytes_per_row(self, *, phase_ctx: PhaseReadyContext) -> int:
        max_row_count = max((int(bundle.outgoing_segment.row_count) for bundle in phase_ctx.transport_bundles if int(bundle.outgoing_segment.row_count) > 0), default=0)
        if max_row_count <= 0:
            return 1
        for bundle in phase_ctx.transport_bundles:
            row_count = int(bundle.outgoing_segment.row_count)
            if row_count <= 0:
                continue
            total_bytes = int(sum(int(payload.payload_byte_count) for payload in bundle.payload_slices))
            if total_bytes > 0:
                return max(1, int(round(total_bytes / row_count)))
        return 1

    def _gather_actual_p0_full_row_matrix(
        self,
        *,
        layer_name: str,
        observation: PreTransportTrafficObservation,
        device: torch.device,
    ) -> tuple[tuple[int, ...], ...]:
        local_prepare_start_ns = time.monotonic_ns()
        local_row = tuple(int(v) for v in observation.local_p0_row)
        local_total = int(sum(local_row))
        if local_total != int(sum(observation.send_splits_rows)):
            raise RuntimeError(f"pre-transport local send mismatch for {layer_name}: local_row={local_row} send_splits={observation.send_splits_rows}")
        row_tensor = torch.tensor(local_row, dtype=torch.int64, device=device)
        local_prepare_end_ns = time.monotonic_ns()
        if len(local_row) <= 1:
            matrix = (local_row,)
            gather_count = 0
            collective_start_ns = local_prepare_end_ns
            collective_end_ns = local_prepare_end_ns
            dtoh_decode_start_ns = local_prepare_end_ns
            dtoh_decode_end_ns = local_prepare_end_ns
        elif dist.is_available() and dist.is_initialized():
            collective_start_ns = time.monotonic_ns()
            gathered = [torch.empty_like(row_tensor) for _ in range(len(local_row))]
            dist.all_gather(gathered, row_tensor, group=self.ep_process_group)
            collective_end_ns = time.monotonic_ns()
            dtoh_decode_start_ns = time.monotonic_ns()
            matrix = tuple(tuple(int(v) for v in item.detach().cpu().tolist()) for item in gathered)
            dtoh_decode_end_ns = time.monotonic_ns()
            gather_count = 1
        else:
            matrix = tuple(local_row for _ in range(len(local_row)))
            gather_count = 0
            collective_start_ns = local_prepare_end_ns
            collective_end_ns = local_prepare_end_ns
            dtoh_decode_start_ns = local_prepare_end_ns
            dtoh_decode_end_ns = local_prepare_end_ns
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_local_prepare",
            start_ns=local_prepare_start_ns,
            end_ns=local_prepare_end_ns,
        )
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_collective",
            start_ns=collective_start_ns,
            end_ns=collective_end_ns,
            collective_count=int(gather_count),
        )
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_dtoh_decode",
            start_ns=dtoh_decode_start_ns,
            end_ns=dtoh_decode_end_ns,
        )
        if dtoh_decode_end_ns > dtoh_decode_start_ns:
            self._record_dtoh_callsite(
                callsite_id="DTOH_P0_MATRIX_DECODE",
                start_ns=dtoh_decode_start_ns,
                end_ns=dtoh_decode_end_ns,
                bytes_if_known=int(row_tensor.numel() * row_tensor.element_size() * max(1, len(local_row))),
            )
        matrix_total = int(sum(sum(int(v) for v in row) for row in matrix))
        self._runtime_state.write("planning_traffic_source", "pre_transport_phase_ready_context")
        self._runtime_state.write("pre_transport_observation_valid", bool(observation.valid))
        self._runtime_state.write("captured_before_transport", bool(observation.captured_before_transport))
        self._runtime_state.write("dispatcher_send_splits", tuple(int(v) for v in observation.send_splits_rows))
        self._runtime_state.write("dispatcher_recv_splits", tuple(int(v) for v in observation.recv_splits_rows))
        self._runtime_state.write("local_p0_row", local_row)
        self._runtime_state.write("actual_p0_total_rows", int(matrix_total))
        self._runtime_state.write("p0_traffic_matrix_gather_count", int(gather_count))
        self._runtime_state.write("prediction_extra_collective_count", 0)
        if (int(sum(observation.send_splits_rows)) > 0 or int(sum(observation.recv_splits_rows)) > 0) and matrix_total <= 0:
            self._write_traffic_source_mismatch(
                layer_name=layer_name,
                observation=observation,
                global_matrix=matrix,
                transport_started=False,
            )
            raise RuntimeError(f"traffic_source_mismatch for {layer_name}: nonzero dispatcher splits but zero actual_p0_full_row_matrix")
        local_col_total = int(sum(int(matrix[src][observation.group_rank]) for src in range(len(matrix)))) if matrix else 0
        if int(sum(observation.recv_splits_rows)) != local_col_total:
            raise RuntimeError(
                f"pre-transport recv mismatch for {layer_name}: recv_total={sum(observation.recv_splits_rows)} col_total={local_col_total} group_rank={observation.group_rank}"
            )
        return matrix

    def _write_traffic_source_mismatch(
        self,
        *,
        layer_name: str,
        observation: PreTransportTrafficObservation,
        global_matrix: tuple[tuple[int, ...], ...],
        transport_started: bool,
    ) -> None:
        target_dir = Path(self.config.executor_heartbeat_path) if self.config.executor_heartbeat_path else Path("outputs/distributed/runtime_traffic_source_mismatch")
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "forward_epoch": int(self._forward_epoch),
            "microbatch_id": self.microbatch_id,
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "global_rank": int(self.rank),
            "group_rank": int(observation.group_rank),
            "dispatcher_send_splits": list(observation.send_splits_rows),
            "dispatcher_recv_splits": list(observation.recv_splits_rows),
            "phase_ready_context_send_splits": list(observation.send_splits_rows),
            "phase_ready_context_recv_splits": list(observation.recv_splits_rows),
            "local_p0_row": list(observation.local_p0_row),
            "global_p0_matrix": [list(row) for row in global_matrix],
            "runtime_observation_p0": (
                self._pending_p0.get(layer_name).to_dict() if self._pending_p0.get(layer_name) is not None else None
            ),
            "planning_stage": "before_token_dispatch",
            "transport_started": bool(transport_started),
        }
        (target_dir / f"traffic_source_mismatch_rank{self.rank}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _next_layer_id(self, layer_name: str) -> str:
        layer_id = parse_layer_id(layer_name)
        try:
            return str(int(layer_id) + 1)
        except ValueError:
            return layer_id


__all__ = ["LifecycleEvidenceMixin"]
