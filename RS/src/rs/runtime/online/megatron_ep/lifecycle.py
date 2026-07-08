"""P0/P1 runtime lifecycle for formal Megatron EP execution."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from rs.core.contracts.observation import RuntimeObservationConfig
from rs.runtime.online.megatron_ep.contracts import (
    InjectionDecision,
    PlanAgreement,
    PolicyContext,
    RouterSenseInjectionConfig,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.runtime.online.megatron_ep.control.agreement_wire import compute_ep_group_hash, run_policy_agreement
from rs.runtime.online.megatron_ep.control.plan_agreement import run_phase_plan_agreement
from rs.runtime.online.megatron_ep.multiphase_pending_window import build_pending_window_shadow
from rs.runtime.online.megatron_ep.observation import (
    PolicyRuntimeRecord,
    RuntimeObservationRecorder,
    build_runtime_observation,
    digest_text,
    extract_int_tuple,
    parse_layer_id,
)
from rs.runtime.online.megatron_ep.observer import RouterSenseObserver
from rs.runtime.online.megatron_ep.p2_provider import P2HintRequest, build_p2_hint_provider
from rs.runtime.online.megatron_ep.pending_window_adapter import MultiphasePendingWindowAdapter
from rs.runtime.online.megatron_ep.policy_adapter import compile_prepared_window_phase_plan
from rs.runtime.online.megatron_ep.release_engine import record_release_event
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhaseExecutionPlan,
    PhasePayloadContract,
    PhaseReadyContext,
    RuntimeIdentity,
    build_phase_ready_context,
)
from rs.runtime.online.megatron_ep.runtime import SelectedLayerStop, UnsupportedSchedulerMode
from rs.runtime.online.megatron_ep.window_state import PreparedPlanBinding, WindowReleaseState, bind_prepared_plan, build_window_state
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.scheduling.registry import resolve_phase_policy, supported_phase_policies
from rs.scheduling.validation import stable_hash


@dataclass
class RouterSenseInjectionRuntime:
    config: RouterSenseInjectionConfig
    rank: int
    local_rank: int
    run_id: str
    step_id: str
    microbatch_id: str
    model_revision_hash: str
    request_table_hash: str
    hostname: str
    observer: RouterSenseObserver | None = None
    ep_group_ranks: tuple[int, ...] = ()
    ep_group_root_global_rank: int = 0
    ep_process_group: Any | None = None
    completed: list[PolicyRuntimeRecord] = field(default_factory=list)
    _pending_p0: dict[str, RuntimeObservation] = field(default_factory=dict)
    _pending_p1: dict[str, RuntimeObservation] = field(default_factory=dict)
    _prepared_plan_state: dict[str, Any] = field(
        default_factory=lambda: {
            "prepared_plan": None,
            "plan_created_at_us": 0,
            "plan_source_layer": "",
        }
    )
    plan_arrival_records: list[dict[str, Any]] = field(default_factory=list)
    window_state_records: list[dict[str, Any]] = field(default_factory=list)
    prepared_plan_bindings: list[dict[str, Any]] = field(default_factory=list)
    release_events: list[dict[str, Any]] = field(default_factory=list)
    window_schedule_shadows: list[dict[str, Any]] = field(default_factory=list)
    prepared_phase_plan_shadows: list[dict[str, Any]] = field(default_factory=list)
    pending_window_driver_records: list[dict[str, Any]] = field(default_factory=list)
    planning_timing_records: list[dict[str, Any]] = field(default_factory=list)
    control_timeline: list[dict[str, Any]] = field(default_factory=list)
    control_commands: list[dict[str, Any]] = field(default_factory=list)
    assertion_state: dict[str, Any] = field(default_factory=dict)
    _active_plan_versions: dict[str, int] = field(default_factory=dict)
    _active_plan_hashes: dict[str, str] = field(default_factory=dict)
    _window_states: dict[str, Any] = field(default_factory=dict)
    observation_recorder: RuntimeObservationRecorder | None = None
    _active_transport: dict[str, Any] | None = None
    _p2_hint_provider: Any | None = None

    # Configuration and policy selection

    def __post_init__(self) -> None:
        if self.observation_recorder is None:
            self.observation_recorder = RuntimeObservationRecorder(
                config=RuntimeObservationConfig(
                    profile=str(getattr(self.config, "observation_profile", "minimal")),
                    capture_enabled=bool(getattr(self.config, "capture_phase_tensors", False)),
                    capture_layer_selector=str(getattr(self.config, "capture_layer_selector", "")),
                    capture_phase_selector=str(getattr(self.config, "capture_phase_selector", "")),
                    heartbeat_enabled=bool(getattr(self.config, "heartbeat_enabled", False)),
                    per_wave_timing_enabled=bool(getattr(self.config, "per_wave_timing_enabled", False)),
                )
            )
        if self.config.p2_hint_mode == "calibrated_artifact":
            self._p2_hint_provider = build_p2_hint_provider(
                self.config.p2_hint_mode,
                shared_state=self._prepared_plan_state,
            )

    def _effective_phase_policy_name(self) -> str:
        if self.config.policy:
            return str(self.config.policy)
        if self.config.scheduler_mode in set(supported_phase_policies()):
            return str(self.config.scheduler_mode)
        return ""

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

    def _pending_window_adapter(self) -> MultiphasePendingWindowAdapter:
        phase_policy_name = self._effective_phase_policy_name()
        if not phase_policy_name:
            raise UnsupportedSchedulerMode("multiphase_pending_window requires a resolved phase policy name")
        return MultiphasePendingWindowAdapter(
            shared_state=self._prepared_plan_state,
            phase_policy_name=phase_policy_name,
            bucket_rows=self.config.bucket_rows,
            p0_weight=self.config.p0_weight,
            p1_reservation_weight=self.config.p1_reservation_weight,
            p2_hint_weight=self.config.p2_hint_weight,
        )

    def _layer_selected(self, layer_name: str) -> bool:
        selector = str(self.config.schedule_layer_selector)
        if selector in {"", "all"}:
            return True
        selected = {item.strip() for item in selector.split(",") if item.strip()}
        return parse_layer_id(layer_name) in selected

    def _phase_selected(self, phase: str) -> bool:
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return True
        return selector == str(phase).lower()

    def _should_schedule_phase(self, *, layer_name: str, phase: str) -> bool:
        return (
            bool(self._effective_phase_policy_name())
            and self.config.execution_mode in {"phase_sync_wave", "multiphase_pending_window"}
            and self.config.control_mode == "sync_before_phase"
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
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

    # Transport activation and timing

    def _activate_transport(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        start_ns = time.monotonic_ns()
        self._active_transport = {
            "layer_name": layer_name,
            "phase": phase,
            "context": context,
            "plan": plan,
        }
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            adapter.activate(layer_name=layer_name, phase=phase, context=context, plan=plan)
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
        if not self.config.executor_heartbeat_path:
            return
        heartbeat_dir = Path(self.config.executor_heartbeat_path)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        target = heartbeat_dir / f"heartbeat-rank{self.rank}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()

    def _timeline(self, event: str, *, layer_name: str, **detail: Any) -> None:
        row = {
            "ts_us": int(time.time() * 1e6),
            "monotonic_ns": time.monotonic_ns(),
            "event_seq": len(self.control_timeline) + 1,
            "event": event,
            "run_id": self.run_id,
            "forward_epoch": 0,
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

    # Hint, shadow, and pending-window state

    def _build_p2_hint(self, *, layer_name: str, phase: str):
        start_ns = time.monotonic_ns()
        if self.config.p2_hint_mode == "calibrated_artifact":
            if self._p2_hint_provider is None:
                self._p2_hint_provider = build_p2_hint_provider(
                    self.config.p2_hint_mode,
                    shared_state=self._prepared_plan_state,
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
        plan = self._prepared_plan_state.get("prepared_plan")
        plan_created_at = int(self._prepared_plan_state.get("plan_created_at_us", 0) or 0)
        source_layer = str(self._prepared_plan_state.get("plan_source_layer", ""))
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
        prepared_plan = self._prepared_plan_state.get("prepared_plan")
        if prepared_plan is None:
            return None
        source_logical_plan_hash = ""
        logical_plan = getattr(prepared_plan, "logical_plan", None)
        if logical_plan is not None:
            source_logical_plan_hash = stable_hash(logical_plan.to_dict())
        return bind_prepared_plan(
            layer_name=layer_name,
            prepared_plan=prepared_plan,
            source_layer_name=str(self._prepared_plan_state.get("plan_source_layer", "")),
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
        state = build_window_state(
            layer_name=layer_name,
            ep_group_ranks=self.ep_group_ranks,
            local_rank=self.local_rank,
            p0_observation=p0_observation if p0_observation is not None else (None if existing is None else existing.p0_observation),
            p1_observation=p1_observation if p1_observation is not None else (None if existing is None else existing.p1_observation),
            prepared_plan=self._prepared_plan_state.get("prepared_plan"),
            prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
            release_state=release_state,
        )
        self._window_states[layer_name] = state
        self.window_state_records.append(state.to_record())
        if state.prepared_plan_binding is not None:
            self.prepared_plan_bindings.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    **state.prepared_plan_binding.to_dict(),
                }
            )
        shadow = build_pending_window_shadow(
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        first_wave = shadow.get("first_executable_wave") or {}
        shadow.setdefault("shadow_first_wave_flow_ids", list(first_wave.get("selected_flow_ids", []) or []))
        shadow.setdefault("shadow_first_wave_edges", list(first_wave.get("selected_edges", []) or []))
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
            state = build_window_state(
                layer_name=layer_name,
                ep_group_ranks=self.ep_group_ranks,
                local_rank=self.local_rank,
                p0_observation=None,
                p1_observation=None,
                prepared_plan=self._prepared_plan_state.get("prepared_plan"),
                prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
                release_state=WindowReleaseState(),
            )
        state, record = record_release_event(state=state, event=event, rank=self.rank, layer_name=layer_name)
        self._window_states[layer_name] = state
        self.release_events.append(record)
        self.window_state_records.append(state.to_record())
        shadow = build_pending_window_shadow(
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        first_wave = shadow.get("first_executable_wave") or {}
        shadow.setdefault("shadow_first_wave_flow_ids", list(first_wave.get("selected_flow_ids", []) or []))
        shadow.setdefault("shadow_first_wave_edges", list(first_wave.get("selected_edges", []) or []))
        self.window_schedule_shadows.append(shadow)

    def _record_prepared_phase_plan_shadow(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> None:
        start_ns = time.monotonic_ns()
        prepared_plan = self._prepared_plan_state.get("prepared_plan")
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
            compiled = compile_prepared_window_phase_plan(
                prepared_plan=prepared_plan,
                local_context=local_context,
                global_contexts=global_contexts,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                policy_name=phase_policy_name,
            )
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

    def _store_prepared_plan(self, *, layer_name: str, observation_p1: RuntimeObservation) -> None:
        total_start_ns = time.monotonic_ns()
        from rs.scheduling.contracts import (
            FlowWindow,
            ForecastPressure,
            GlobalReadySetOptions,
            LogicalTopology,
            MultiPhaseSchedulingProblem,
            ReleaseConstraint,
        )
        from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
        from rs.scheduling.validation import stable_hash

        per_peer = tuple(int(value) for value in observation_p1.per_peer_bytes)
        num_peers = len(per_peer)
        if num_peers <= 0:
            return
        forecast_matrix = tuple(
            tuple(int(per_peer[j]) if i != j else 0 for j in range(num_peers))
            for i in range(num_peers)
        )
        p0_obs = self._pending_p0.get(layer_name)
        if p0_obs is not None:
            p0_per_peer = tuple(int(value) for value in p0_obs.per_peer_bytes)
            dispatch_matrix = tuple(
                tuple(int(p0_per_peer[j]) if j < len(p0_per_peer) and i != j else 0 for j in range(num_peers))
                for i in range(num_peers)
            )
        else:
            dispatch_matrix = forecast_matrix
        forecast_digest = stable_hash({"per_peer_bytes": list(per_peer), "layer": layer_name})
        problem = MultiPhaseSchedulingProblem(
            flow_window=FlowWindow(ready_flows=(), blocked_flows=(), forecast_pressure=()),
            topology=LogicalTopology(num_gpus=num_peers),
            release_model=ReleaseConstraint(
                phase="p1_return",
                rank=0,
                release_after_phase="p0_dispatch",
                expert_compute_delay=0.0,
            ),
            forecast=ForecastPressure(
                source="online_p1_observation",
                digest=forecast_digest,
                oracle=False,
                evaluation_eligible=True,
                matrix_shape=(num_peers, num_peers),
                matrix_total_bytes=sum(int(value) for value in per_peer),
                matrix=forecast_matrix,
            ),
            options=GlobalReadySetOptions(
                scheduling_mode="runtime_lookahead",
                information_mode="p0_p1_p2",
                prediction_confidence=1.0,
                p0_weight=float(self.config.p0_weight),
                p1_reservation_weight=float(self.config.p1_reservation_weight),
                p2_hint_weight=float(self.config.p2_hint_weight),
                max_waves=256,
            ),
            p0_dispatch_matrix=dispatch_matrix,
            p1_return_matrix=forecast_matrix,
            p2_next_dispatch_forecast_matrix=forecast_matrix,
        )
        policy = RouterSenseMultiphaseLookaheadPolicy(
            information_mode="p0_p1_p2",
            p0_weight=self.config.p0_weight,
            p1_reservation_weight=self.config.p1_reservation_weight,
            p2_hint_weight=self.config.p2_hint_weight,
        )
        layer_id = parse_layer_id(layer_name)
        try:
            applies_from_layer_id = str(int(layer_id) + 1)
        except ValueError:
            applies_from_layer_id = layer_id
        build_start_ns = time.monotonic_ns()
        prepared = policy.build_prepared_window_plan(
            problem=problem,
            created_at_layer_id=str(layer_id),
            applies_from_layer_id=applies_from_layer_id,
        )
        build_end_ns = time.monotonic_ns()
        self._prepared_plan_state["prepared_plan"] = prepared
        self._prepared_plan_state["plan_created_at_us"] = int(time.time() * 1e6)
        self._prepared_plan_state["plan_source_layer"] = layer_name
        self._timeline(
            "prepared_window_plan_stored",
            layer_name=layer_name,
            window_key=prepared.window_key,
            forecast_digest=prepared.forecast_digest,
            applies_from_layer_id=prepared.applies_from_layer_id,
        )
        total_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="store_prepared_plan",
            start_ns=total_start_ns,
            end_ns=total_end_ns,
            prepared_window_key=str(prepared.window_key),
            forecast_digest=str(prepared.forecast_digest),
            policy_name=str(prepared.logical_plan.policy_name),
            logical_build_time_us=max(0.0, float(build_end_ns - build_start_ns) / 1000.0),
        )

    def _record_pending_window_driver(
        self,
        *,
        layer_name: str,
        phase: str,
        plan: PhaseExecutionPlan,
    ) -> None:
        if self.config.execution_mode != "multiphase_pending_window":
            return
        metrics = dict(plan.metrics)
        record = {
            "ts_us": int(time.time() * 1e6),
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "plan_hash": plan.plan_hash,
            "policy_name": plan.policy_name,
            "compiled_from_pending_window": bool(metrics.get("compiled_from_pending_window", False)),
            "pending_window_logical_policy_name": str(metrics.get("pending_window_logical_policy_name", "")),
            "pending_window_plan_hash": str(metrics.get("pending_window_plan_hash", "")),
            "pending_window_information_mode": str(metrics.get("pending_window_information_mode", "")),
            "pending_window_forecast_available": bool(metrics.get("pending_window_forecast_available", False)),
            "pending_window_p0_total_bytes": int(metrics.get("pending_window_p0_total_bytes", 0) or 0),
            "pending_window_p1_total_bytes": int(metrics.get("pending_window_p1_total_bytes", 0) or 0),
            "pending_window_p2_total_bytes": int(metrics.get("pending_window_p2_total_bytes", 0) or 0),
            "pending_window_p1_matrix_source": str(metrics.get("pending_window_p1_matrix_source", "")),
            "pending_window_p2_matrix_source": str(metrics.get("pending_window_p2_matrix_source", "")),
            "prepared_window_key": str(metrics.get("prepared_window_key", "")),
            "source_logical_plan_hash": str(metrics.get("source_logical_plan_hash", "")),
            "wave_count": len(plan.waves),
            "bucket_count": sum(len(wave.bucket_tasks) for wave in plan.waves),
            "hint_edges_consumed": int(metrics.get("hint_edges_consumed", 0) or 0),
            "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
            "prepared_plan_order_preserved": bool(metrics.get("prepared_plan_order_preserved", False)),
            "bucket_order": list(metrics.get("bucket_order", [])),
        }
        self.pending_window_driver_records.append(record)

    def capture_phase_transport_output(
        self,
        *,
        layer_name: str,
        phase: str,
        result: Any,
        dispatcher: Any,
    ) -> None:
        recorder = self.observation_recorder
        if recorder is None:
            return
        layer_id = parse_layer_id(layer_name)
        if not recorder.should_capture_tensor(layer_id=layer_id, phase=phase):
            return
        tensors: list[tuple[str, torch.Tensor]] = []
        if isinstance(result, torch.Tensor):
            tensors.append(("hidden_states", result))
        elif isinstance(result, (list, tuple)):
            roles = ["hidden_states", "routing_probs"]
            for index, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    role = roles[index] if index < len(roles) else f"output_{index}"
                    tensors.append((role, item))
        input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        for role, tensor in tensors:
            checksum = hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
            row_digest = hashlib.sha256(
                tensor.detach().float().cpu().reshape(tensor.shape[0], -1).numpy().tobytes()
            ).hexdigest() if tensor.ndim >= 1 else checksum
            recorder.record_captured_tensor(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "phase": phase,
                    "rank": self.rank,
                    "tensor_role": role,
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "input_splits": list(input_splits),
                    "output_splits": list(output_splits),
                    "row_order_digest": row_digest,
                    "tensor_checksum": checksum,
                    "tensor": tensor.detach().cpu(),
                }
            )

    def _record_observer(self, **payload: Any) -> None:
        if self.observer is None:
            return
        try:
            self.observer.record(**payload)
        except Exception:
            pass

    def _context(self, layer_name: str) -> PolicyContext:
        layer_id = parse_layer_id(layer_name)
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        return PolicyContext(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            layer_id=layer_id,
            run_id_digest=digest_text(self.run_id),
            step_id_digest=digest_text(self.step_id),
            microbatch_id_digest=digest_text(self.microbatch_id),
            request_table_hash=self.request_table_hash,
            model_revision_hash=self.model_revision_hash,
            expert_placement_hash="unknown",
            ep_group_ranks=self.ep_group_ranks,
            ep_group_size=len(self.ep_group_ranks),
            ep_group_hash=ep_group_hash,
            future_hint_mode=self.config.future_hint_mode,
            control_mode=self.config.control_mode,
        )

    def _plan_key(self, layer_name: str, phase: str) -> dict[str, Any]:
        return {
            "run_id_digest": digest_text(self.run_id),
            "forward_epoch": 0,
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "ep_group_hash": compute_ep_group_hash(self.ep_group_ranks),
            "ep_group_epoch": 0,
            "model_revision_hash": self.model_revision_hash,
            "expert_placement_hash": "unknown",
            "request_table_hash": self.request_table_hash,
        }

    # Main lifecycle hooks

    def before_token_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("before_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
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
        self._record_plan_arrival(layer_name=layer_name, phase="P0")
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P0")
        self._pending_p0[layer_name] = observation
        self._record_window_state(layer_name=layer_name, p0_observation=observation)
        phase_ctx_start_ns = time.monotonic_ns()
        phase_ctx = build_phase_ready_context(
            PhaseContextBuildRequest(
                plan_key=self._plan_key(layer_name, "P0"),
                runtime_identity=RuntimeIdentity(
                    run_id=self.run_id,
                    forward_epoch=0,
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
                    phase="P0",
                    payload_roles=("hidden_states", "routing_probs"),
                    atomic_submit=True,
                ),
                packed_tensors=tuple(
                    tensor for tensor in (packed_hidden_states, packed_probs) if isinstance(tensor, torch.Tensor)
                ),
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
            phase="P0",
            stage="build_phase_ready_context",
            start_ns=phase_ctx_start_ns,
            end_ns=phase_ctx_end_ns,
            remote_rows=int(observation.remote_rows),
            hint_mode=str(p2_hint.hint_mode),
        )
        if self.observation_recorder is not None:
            self.observation_recorder.record_phase_context(phase_ctx.to_dict())
            for bundle in phase_ctx.transport_bundles:
                self.observation_recorder.record_transport_bundle(bundle.to_dict())
        self._record_prepared_phase_plan_shadow(
            layer_name=layer_name,
            phase="P0",
            local_context=phase_ctx,
            global_contexts=(phase_ctx,),
        )
        pre_input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        pre_output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        hidden_ptr = int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1
        probs_ptr = int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1
        self._timeline(
            "p0_pre_transport_observation_ready",
            layer_name=layer_name,
            input_splits=list(pre_input_splits),
            output_splits=list(pre_output_splits),
            hidden_shape=list(packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
            probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P0")
        if self._should_schedule_phase(layer_name=layer_name, phase="P0"):
            agreement_start_ns = time.monotonic_ns()
            policy = self._pending_window_adapter() if self.config.execution_mode == "multiphase_pending_window" else self._phase_policy()
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
                self.observation_recorder.record_scheduled_plan(plan.to_dict())
            self._record_pending_window_driver(layer_name=layer_name, phase="P0", plan=plan)
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
        plan, agreement = run_policy_agreement(
            local_observations=local_observations,
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
        if bool(self._effective_phase_policy_name()):
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P0")
            clear_end_ns = time.monotonic_ns()
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

    def before_token_combine(self, *, layer_name: str, dispatcher: Any, packed_hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("before_token_combine_enter", layer_name=layer_name, phase_name="P1")
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
                    forward_epoch=0,
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
            self.observation_recorder.record_phase_context(phase_ctx.to_dict())
            for bundle in phase_ctx.transport_bundles:
                self.observation_recorder.record_transport_bundle(bundle.to_dict())
        self._record_prepared_phase_plan_shadow(
            layer_name=layer_name,
            phase="P1",
            local_context=phase_ctx,
            global_contexts=(phase_ctx,),
        )
        self._timeline(
            "p1_pre_transport_observation_ready",
            layer_name=layer_name,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P1")
        if self._should_schedule_phase(layer_name=layer_name, phase="P1"):
            agreement_start_ns = time.monotonic_ns()
            policy = self._pending_window_adapter() if self.config.execution_mode == "multiphase_pending_window" else self._phase_policy()
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
                self.observation_recorder.record_scheduled_plan(plan.to_dict())
            self._record_pending_window_driver(layer_name=layer_name, phase="P1", plan=plan)
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
        if bool(self._effective_phase_policy_name()):
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P1")
            clear_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P1",
                stage="clear_transport",
                start_ns=clear_start_ns,
                end_ns=clear_end_ns,
            )
            observation_p1 = self._pending_p1.pop(layer_name, None)
            if observation_p1 is not None:
                self._store_prepared_plan(layer_name=layer_name, observation_p1=observation_p1)
                self._record_window_state(layer_name=layer_name, p1_observation=observation_p1)
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

    # Shadow-only native observation hooks

    def on_dispatch(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        if self.config.scheduler_mode in {"disabled", "native_passthrough_identity"} or bool(self._effective_phase_policy_name()):
            return
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

    def on_combine(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        if self.config.scheduler_mode in {"disabled", "native_passthrough_identity"} or bool(self._effective_phase_policy_name()):
            return
        if layer_name not in self._pending_p0:
            return
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
        policy = self._phase_policy()
        plan, agreement = run_policy_agreement(
            local_observations=local_observations,
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

    # Export helpers

    def _export_list(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(rows)

    def _export_observation_rows(self, method_name: str) -> list[dict[str, Any]]:
        if self.observation_recorder is None:
            return []
        export_fn = getattr(self.observation_recorder, method_name)
        return list(export_fn())

    def export_records(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.completed]

    def export_control_timeline(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_timeline)

    def export_control_commands(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_commands)

    def export_plan_arrival_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.plan_arrival_records)

    def export_window_state_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.window_state_records)

    def export_prepared_plan_bindings(self) -> list[dict[str, Any]]:
        return self._export_list(self.prepared_plan_bindings)

    def export_release_events(self) -> list[dict[str, Any]]:
        return self._export_list(self.release_events)

    def export_window_schedule_shadows(self) -> list[dict[str, Any]]:
        return self._export_list(self.window_schedule_shadows)

    def export_prepared_phase_plan_shadows(self) -> list[dict[str, Any]]:
        return self._export_list(self.prepared_phase_plan_shadows)

    def export_pending_window_driver_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.pending_window_driver_records)

    def export_planning_timing_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.planning_timing_records)

    def export_assertions(self) -> dict[str, Any]:
        return dict(self.assertion_state)

    def export_phase_contexts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_phase_contexts")

    def export_transport_bundles(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_bundles")

    def export_scheduled_phase_plans(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_scheduled_phase_plans")

    def export_transport_execution_results(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_execution")

    def export_captured_phase_tensors(self) -> list[dict[str, Any]]:
        rows = self._export_observation_rows("export_captured_phase_tensors")
        return [{key: value for key, value in item.items() if key != "tensor"} for item in rows]

    def export_captured_phase_tensors_with_payload(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_captured_phase_tensors")
