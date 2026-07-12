"""Megatron EP 正式执行链路的 P0/P1 生命周期主线。

这个文件是在线运行时的核心编排器，主要负责：
- before/after token_dispatch
- before/after token_combine
- phase context 构建、计划协商、transport 激活/清理
- prepared plan、release state、pending-window shadow 的记录
如果想看“运行时一层里到底发生了什么”，优先看这里。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rs.core.layer_selection import layer_selected, resolve_layer_selector
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
from rs.runtime.online.megatron_ep.control.p2_matrix import TrafficMatrixBundle, build_traffic_matrix_bundle
from rs.runtime.online.megatron_ep.control.plan_agreement import run_phase_plan_agreement
from rs.runtime.online.megatron_ep.control.p2_contracts import P2HintRequest
from rs.runtime.online.megatron_ep.control.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.observation import (
    PolicyRuntimeRecord,
    RouterSenseObserver,
    RuntimeObservationRecorder,
    build_runtime_observation,
    control_replay_trace_row,
    digest_text,
    extract_int_tuple,
    parse_layer_id,
    phase_context_artifact,
    scheduled_plan_artifact,
    transport_bundle_artifact,
)
from rs.runtime.online.megatron_ep.state.window_runtime_state import (
    PreparedPlanBinding,
    WindowReleaseState,
    bind_prepared_plan,
)
from rs.runtime.online.megatron_ep.async_release.joint_plan_agreement import GlobalJointPlanWire
from rs.runtime.online.megatron_ep.compiler_facade import (
    CompilationOptions,
    PlanCompilationRequest,
    build_phase_canonical_tasks,
    compile_schedule,
)
from rs.runtime.online.megatron_ep.state import PreparedWindowRuntimeState
from rs.runtime.online.megatron_ep.planning.window_shadow_service import (
    advance_window_release,
    build_window_state_record,
    maybe_build_window_shadow,
)
from rs.runtime.online.megatron_ep.observation.runtime_export import build_prepared_plan_summary
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhaseExecutionPlan,
    PhasePayloadContract,
    PhaseReadyContext,
    PreTransportTrafficObservation,
    RuntimeIdentity,
    build_phase_ready_context,
    reconstruct_global_phase_contexts_from_byte_matrix,
)
from rs.runtime.online.megatron_ep.runtime import (
    SelectedLayerStop,
    UnsupportedSchedulerMode,
    resolve_online_policy_config,
)
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.runtime.online.megatron_ep.prediction import (
    ActiveNextDispatchPrediction,
    CopyCurrentDispatchPredictor,
    HistoryEMATrafficPredictor,
    PredictionInput,
    ZeroHintPredictor,
    compare_predicted_to_actual,
    maybe_capture_expert_route_trace,
)
from rs.scheduling.contracts import PreparedWindowPlan
from rs.scheduling.registry import resolve_phase_policy
from rs.scheduling.unified_interface import PolicyOptions, build_policy, build_request_from_problem
from rs.scheduling.bucketizer import (
    BUCKET_MODE_DYNAMIC_CURRENT,
    BUCKET_MODE_FIXED_ROWS,
    bucket_mode_for_rows,
    summarize_bucket_tasks,
)
from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_digest_remote,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
)
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
    _runtime_state: PreparedWindowRuntimeState = field(default_factory=PreparedWindowRuntimeState)
    plan_arrival_records: list[dict[str, Any]] = field(default_factory=list)
    window_state_records: list[dict[str, Any]] = field(default_factory=list)
    prepared_plan_bindings: list[dict[str, Any]] = field(default_factory=list)
    release_events: list[dict[str, Any]] = field(default_factory=list)
    window_schedule_shadows: list[dict[str, Any]] = field(default_factory=list)
    prepared_phase_plan_shadows: list[dict[str, Any]] = field(default_factory=list)
    pending_window_driver_records: list[dict[str, Any]] = field(default_factory=list)
    planning_timing_records: list[dict[str, Any]] = field(default_factory=list)
    control_replay_traces: list[dict[str, Any]] = field(default_factory=list)
    prediction_audits: list[dict[str, Any]] = field(default_factory=list)
    control_timeline: list[dict[str, Any]] = field(default_factory=list)
    control_commands: list[dict[str, Any]] = field(default_factory=list)
    assertion_state: dict[str, Any] = field(default_factory=dict)
    _active_plan_versions: dict[str, int] = field(default_factory=dict)
    _active_plan_hashes: dict[str, str] = field(default_factory=dict)
    _window_states: dict[str, Any] = field(default_factory=dict)
    _selected_layer_matches_seen: set[str] = field(default_factory=set)
    _forward_epoch: int = 0
    observation_recorder: RuntimeObservationRecorder | None = None
    _active_transport: dict[str, Any] | None = None
    _p2_hint_provider: Any | None = None
    _pending_window_adapter_instance: Any | None = None
    perf_counters: dict[str, dict[str, float]] = field(default_factory=dict)

    # Configuration and policy selection

    def __post_init__(self) -> None:
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

    @property
    def _prepared_plan_state(self) -> PreparedWindowRuntimeState:
        return self._runtime_state

    def _artifact_profile(self) -> str:
        return str(getattr(self.config, "observation_profile", "minimal"))

    def _is_perf_profile(self) -> bool:
        return self._artifact_profile() == "perf"

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
        resolved = resolve_online_policy_config(self.config)
        if resolved is None:
            return ""
        return str(resolved.builder_key)

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

    def _pending_window_adapter(self) -> Any:
        from rs.runtime.online.megatron_ep.pending_window import MultiphasePendingWindowAdapter

        phase_policy_name = self._effective_phase_policy_name()
        if not phase_policy_name:
            raise UnsupportedSchedulerMode("multiphase_pending_window requires a resolved phase policy name")
        if self._pending_window_adapter_instance is None:
            self._pending_window_adapter_instance = MultiphasePendingWindowAdapter(
                shared_state=self._runtime_state,
                phase_policy_name=phase_policy_name,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                fast_path_enabled=self._is_perf_profile(),
            )
        return self._pending_window_adapter_instance

    def _layer_selected(self, layer_name: str) -> bool:
        resolved = resolve_layer_selector(
            str(self.config.schedule_layer_selector),
            selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
            invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
        )
        matched = layer_selected(parse_layer_id(layer_name), selector=resolved)
        if matched:
            self._selected_layer_matches_seen.add(parse_layer_id(layer_name))
            self._runtime_state.metrics.selected_layer_match_count = int(len(self._selected_layer_matches_seen))
        return matched

    def _phase_selected(self, phase: str) -> bool:
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return True
        return selector == str(phase).lower()

    def _should_schedule_phase(self, *, layer_name: str, phase: str) -> bool:
        return (
            bool(self._effective_phase_policy_name())
            and self.config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            and self.config.control_mode == "sync_before_phase"
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
        )

    def _is_joint_window_async_mode(self) -> bool:
        return str(self.config.execution_mode) == "joint_window_async_p2p"

    def _runtime_safe_joint_pair(self) -> tuple[str, str]:
        policy_name = str(self.config.policy or "")
        if "gated_greedy" in policy_name:
            return ("U_gated_greedy_maximal", "B_gated_greedy_maximal")
        return ("U_barrier_criticality_global_matching", "B_barrier_criticality_matching")

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

    # Transport activation and timing

    def _activate_transport(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        start_ns = time.monotonic_ns()
        if self._layer_selected(layer_name):
            self._runtime_state.metrics.selected_transport_execution_count = int(
                self._runtime_state.metrics.selected_transport_execution_count
            ) + 1
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
        local_row = tuple(int(v) for v in observation.local_p0_row)
        local_total = int(sum(local_row))
        if local_total != int(sum(observation.send_splits_rows)):
            raise RuntimeError(f"pre-transport local send mismatch for {layer_name}: local_row={local_row} send_splits={observation.send_splits_rows}")
        row_tensor = torch.tensor(local_row, dtype=torch.int64, device=device)
        if len(local_row) <= 1:
            matrix = (local_row,)
            gather_count = 0
        elif dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(row_tensor) for _ in range(len(local_row))]
            dist.all_gather(gathered, row_tensor, group=self.ep_process_group)
            matrix = tuple(tuple(int(v) for v in item.detach().cpu().tolist()) for item in gathered)
            gather_count = 1
        else:
            matrix = tuple(local_row for _ in range(len(local_row)))
            gather_count = 0
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

    def _online_p2_predictor_name(self) -> str:
        return str(getattr(self.config, "online_p2_predictor", "copy_current_dispatch") or "copy_current_dispatch")

    def _build_online_predictor(self):
        name = self._online_p2_predictor_name()
        if name in {"none", "zero_hint"}:
            return ZeroHintPredictor()
        if name == "history_ema":
            return HistoryEMATrafficPredictor(alpha=0.5)
        if name == "copy_current_dispatch":
            return CopyCurrentDispatchPredictor()
        raise ValueError(
            "unsupported online_p2_predictor "
            f"{name!r}; expected one of ('none', 'zero_hint', 'copy_current_dispatch', 'history_ema')"
        )

    def _resolved_online_policy_family(self) -> str:
        resolved = resolve_online_policy_config(self.config)
        if resolved is None:
            return ""
        return str(getattr(resolved.spec, "family", ""))

    def _policy_supports_runtime_prediction(self) -> bool:
        return (
            self._resolved_online_policy_family() in {"joint_u", "runtime_safe"}
            and str(self.config.p2_hint_mode) == "calibrated_artifact"
        )

    def _policy_uses_joint_window_plan(self) -> bool:
        return (
            self._resolved_online_policy_family() in {"joint_u", "runtime_safe"}
            and self._is_joint_window_async_mode()
        )

    def _should_generate_runtime_prediction(self) -> bool:
        return self._policy_supports_runtime_prediction()

    def _record_prediction_for_dispatch(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
        device: torch.device,
    ) -> None:
        stage_start_ns = time.monotonic_ns()
        layer_id = parse_layer_id(layer_name)
        next_layer_id = self._next_layer_id(layer_name)
        world_size = int(len(self.ep_group_ranks) or len(observation.local_p0_row) or 1)
        full_matrix = tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
        remote_matrix = canonicalize_remote_matrix(full_matrix)
        actual_dispatch_by_layer = dict(self._runtime_state.read("actual_dispatch_by_layer", {}) or {})
        actual_dispatch_by_layer[str(layer_id)] = {
            "matrix": [list(row) for row in remote_matrix],
            "full_matrix": [list(row) for row in full_matrix],
            "matrix_digest": matrix_digest_remote(remote_matrix),
            "matrix_source": "pre_transport_phase_ready_context",
            "row_sums": list(matrix_row_sums_remote(remote_matrix)),
            "col_sums": list(matrix_col_sums_remote(remote_matrix)),
            "total_bytes": int(matrix_remote_bytes(remote_matrix)),
            "nonzero_edge_count": int(matrix_nonzero_remote_edge_count(remote_matrix)),
        }
        self._runtime_state.write("actual_dispatch_by_layer", actual_dispatch_by_layer)

        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
        existing_prediction = predicted_dispatch_by_layer.get(str(layer_id))
        audit_start_ns = time.monotonic_ns()
        if isinstance(existing_prediction, dict) and existing_prediction:
            from rs.runtime.online.megatron_ep.prediction.contracts import PredictedTrafficMatrix

            predicted = PredictedTrafficMatrix(
                predictor_name=str(existing_prediction.get("predictor_name", "")),
                predictor_version=str(existing_prediction.get("predictor_version", "")),
                source_layer_id=str(existing_prediction.get("source_layer_id", "")),
                predicted_layer_id=str(existing_prediction.get("predicted_layer_id", "")),
                matrix=tuple(tuple(int(value) for value in row) for row in existing_prediction.get("matrix", [])),
                matrix_digest=str(existing_prediction.get("matrix_digest", "")),
                total_bytes=int(existing_prediction.get("total_bytes", 0) or 0),
                nonzero_edge_count=int(existing_prediction.get("nonzero_edge_count", 0) or 0),
                confidence=float(existing_prediction.get("confidence", 0.0) or 0.0),
                is_oracle=bool(existing_prediction.get("is_oracle", False)),
                evaluation_eligible=bool(existing_prediction.get("evaluation_eligible", False)),
                created_at_phase=str(existing_prediction.get("created_at_phase", "")),
            )
            audit = compare_predicted_to_actual(predicted, remote_matrix)
            self.prediction_audits.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "actual_matrix_source": "pre_transport_phase_ready_context",
                    **audit.to_dict(),
                }
            )
            predicted_dispatch_by_layer.pop(str(layer_id), None)
        audit_end_ns = time.monotonic_ns()

        predictor = self._build_online_predictor()
        prediction_input = PredictionInput(
            run_id_digest=digest_text(self.run_id),
            layer_id=str(layer_id),
            next_layer_id=str(next_layer_id),
            rank=int(self.rank),
            world_size=world_size,
            current_dispatch_matrix_digest=str(matrix_digest_remote(remote_matrix)),
            current_dispatch_total_bytes=int(matrix_remote_bytes(remote_matrix)),
            current_dispatch_nonzero_edges=int(matrix_nonzero_remote_edge_count(remote_matrix)),
            metadata={
                "matrix_source": "pre_transport_phase_ready_context",
                "is_global": True,
                "traffic_units": "rows",
                "dispatcher_send_splits": list(observation.send_splits_rows),
                "dispatcher_recv_splits": list(observation.recv_splits_rows),
                "payload_roles": [descriptor.tensor_role for descriptor in phase_ctx.payload_specs],
                "previous_dispatch_matrix": (
                    actual_dispatch_by_layer.get(str(int(layer_id) - 1), {}).get("matrix")
                    if str(layer_id).isdigit()
                    else None
                ),
            },
        )
        prediction_start_ns = time.monotonic_ns()
        predicted = predictor.predict(prediction_input=prediction_input, current_dispatch_matrix=remote_matrix)
        prediction_end_ns = time.monotonic_ns()
        predicted_dispatch_by_layer[str(next_layer_id)] = predicted.to_dict()
        self._runtime_state.write("predicted_dispatch_by_layer", predicted_dispatch_by_layer)
        active_prediction = ActiveNextDispatchPrediction(
            source_layer_id=str(layer_id),
            target_layer_id=str(next_layer_id),
            forecast_matrix=predicted.matrix,
            matrix_digest=str(predicted.matrix_digest),
            predictor_name=str(predicted.predictor_name),
            predictor_version=str(predicted.predictor_version),
            confidence=float(predicted.confidence),
            evaluation_eligible=bool(predicted.evaluation_eligible),
            is_oracle=bool(predicted.is_oracle),
            created_at_phase=str(predicted.created_at_phase),
            created_at_stage="after_p0_observation",
            prediction_time_us=max(0.0, float(prediction_end_ns - prediction_start_ns) / 1000.0),
            valid=bool(predicted.valid),
            error=str(predicted.error),
        )
        self._runtime_state.write("active_next_dispatch_prediction", active_prediction.to_dict())
        self._runtime_state.write("latest_predictor_name", predicted.predictor_name)
        self._runtime_state.write("latest_prediction_digest", predicted.matrix_digest)
        self._runtime_state.write("latest_prediction_target_layer_id", str(next_layer_id))
        self._runtime_state.write("latest_prediction_matrix_source", "pre_transport_phase_ready_context")
        self._runtime_state.write("latest_prediction_row_sums", [int(sum(row)) for row in predicted.matrix])
        self._runtime_state.write(
            "latest_prediction_col_sums",
            [
            int(sum(predicted.matrix[row_idx][col_idx] for row_idx in range(len(predicted.matrix))))
            for col_idx in range(len(predicted.matrix[0]) if predicted.matrix else 0)
            ],
        )
        stage_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="predict_next_dispatch",
            start_ns=stage_start_ns,
            end_ns=stage_end_ns,
            matrix_source="pre_transport_phase_ready_context",
            matrix_total_bytes=int(matrix_remote_bytes(remote_matrix)),
            matrix_nonzero_edge_count=int(matrix_nonzero_remote_edge_count(remote_matrix)),
            p2_matrix_gather_time_us=0.0,
            p2_matrix_gather_call_count=0,
            predictor_name=str(predicted.predictor_name),
            predicted_layer_id=str(next_layer_id),
            prediction_confidence=float(predicted.confidence),
            prediction_valid=bool(predicted.valid),
            prediction_error=str(predicted.error),
            prediction_time_us=max(0.0, float(prediction_end_ns - prediction_start_ns) / 1000.0),
            audit_time_us=max(0.0, float(audit_end_ns - audit_start_ns) / 1000.0),
            prediction_audit_emitted=bool(existing_prediction is not None),
        )

    # Hint, shadow, and pending-window state

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
                        legacy_compiler_bridge=True,
                    ),
                    prepared_plan=prepared_plan,
                    prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                    legacy_phase_policy_name=str(phase_policy_name),
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

    def _store_runtime_joint_plan_from_p0(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation_p0: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
    ) -> None:
        from rs.scheduling.contracts import (
            FlowWindow,
            ForecastPressure,
            GlobalReadySetOptions,
            LogicalTopology,
            MultiPhaseSchedulingProblem,
            ReleaseConstraint,
        )
        from rs.runtime.online.megatron_ep.async_release.runtime_projection import host_project_safe_selection

        self._assert_bucket_mode_consistency()
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
                source="active_next_dispatch_prediction" if active_prediction else "zero_hint",
                digest=forecast_digest,
                oracle=bool(active_prediction.get("is_oracle", False)) if active_prediction else False,
                evaluation_eligible=bool(active_prediction.get("evaluation_eligible", True)) if active_prediction else True,
                matrix_shape=(num_peers, num_peers),
                matrix_total_bytes=int(sum(sum(int(v) for v in row) for row in forecast_matrix)),
                matrix=forecast_matrix,
                metadata={
                    "predictor_name": predictor_name,
                    "prediction_digest": prediction_digest,
                },
            ),
            options=GlobalReadySetOptions(
                scheduling_mode="runtime_lookahead",
                information_mode="p0_p1_p2",
                prediction_confidence=float(prediction_confidence),
                p0_weight=float(self.config.p0_weight),
                p1_reservation_weight=float(self.config.p1_reservation_weight),
                p2_hint_weight=float(self.config.p2_hint_weight),
                max_waves=256,
            ),
            p0_dispatch_matrix=remote_dispatch_matrix,
            p1_return_matrix=inferred_p1_remote,
            p2_next_dispatch_forecast_matrix=forecast_matrix,
        )
        effective_policy = str(self._effective_phase_policy_name() or "")
        phase_local_async_policies = {"bucketed_fifo", "greedy_ready_set", "birkhoff_phase_local", "phase_barrier_fifo"}
        policy_options = PolicyOptions(
            p0_weight=float(self.config.p0_weight),
            p1_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
            residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
            barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
            age_weight=float(getattr(self.config, "age_weight", 0.15)),
            prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
        )
        if effective_policy in phase_local_async_policies:
            phase_local_request = build_request_from_problem(
                request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:raw_u",
                problem=problem,
                bucket_rows=int(self.config.bucket_rows),
                policy_options=policy_options,
                hint_type=str(getattr(problem.forecast, "source", "none") if problem.forecast is not None else "none"),
                confidence=float(prediction_confidence),
                layer_id=int(layer_id),
            )
            raw_u_name = effective_policy
            paired_b_name = effective_policy
            raw_u_start_ns = time.monotonic_ns()
            raw_u_plan = build_policy(raw_u_name, phase_local_request.policy_options).plan(phase_local_request)
            raw_u_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="raw_u_build",
                start_ns=raw_u_start_ns,
                end_ns=raw_u_end_ns,
                policy_name=raw_u_name,
            )
            consumed_weights = dict((raw_u_plan.diagnostics or {}).get("consumed_weights", {}))
            requested_weights = {
                "residual_weight": float(policy_options.residual_weight),
                "barrier_weight": float(policy_options.barrier_weight),
                "age_weight": float(policy_options.age_weight),
                "prediction_weight": float(policy_options.prediction_weight),
            }
            if raw_u_name.startswith("U_") and consumed_weights != requested_weights:
                raise RuntimeError(
                    f"async joint U weights were not consumed: requested={requested_weights} consumed={consumed_weights}"
                )
            paired_b_start_ns = time.monotonic_ns()
            paired_b_plan = raw_u_plan
            paired_b_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="paired_b_build",
                start_ns=paired_b_start_ns,
                end_ns=paired_b_end_ns,
                policy_name=paired_b_name,
            )
        else:
            raw_u_name, paired_b_name = self._runtime_safe_joint_pair()
            safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
            request = build_request_from_problem(
                request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:safe_joint",
                problem=problem,
                bucket_rows=int(self.config.bucket_rows),
                policy_options=policy_options,
                hint_type=str(getattr(problem.forecast, "source", "none") if problem.forecast is not None else "none"),
                confidence=float(prediction_confidence),
                layer_id=int(layer_id),
            )
            raw_u_start_ns = time.monotonic_ns()
            raw_u_plan = build_policy(raw_u_name, request.policy_options).plan(request)
            raw_u_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="raw_u_build",
                start_ns=raw_u_start_ns,
                end_ns=raw_u_end_ns,
                policy_name=raw_u_name,
            )
            paired_b_start_ns = time.monotonic_ns()
            if safe_projection_mode == "disabled":
                paired_b_plan = raw_u_plan
                paired_b_end_ns = paired_b_start_ns
            else:
                paired_b_plan = build_policy(paired_b_name, request.policy_options).plan(request)
                paired_b_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="paired_b_build",
                start_ns=paired_b_start_ns,
                end_ns=paired_b_end_ns,
                policy_name=paired_b_name,
                skipped=bool(safe_projection_mode == "disabled"),
            )
        safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
        if safe_projection_mode == "disabled":
            host_projection_start_ns = time.monotonic_ns()
            host_projection_end_ns = host_projection_start_ns
            safe_projection = {
                "ideal_raw_u_estimated_makespan": float(raw_u_plan.diagnostics.get("makespan", 0.0) or 0.0),
                "host_projected_raw_u_estimated_makespan": float(raw_u_plan.diagnostics.get("makespan", 0.0) or 0.0),
                "ideal_paired_b_estimated_makespan": float(raw_u_plan.diagnostics.get("makespan", 0.0) or 0.0),
                "host_projected_paired_b_estimated_makespan": float(raw_u_plan.diagnostics.get("makespan", 0.0) or 0.0),
                "host_projected_safe_selection": str(raw_u_plan.policy_name),
                "projection_mode": "disabled",
            }
        else:
            host_projection_start_ns = time.monotonic_ns()
            safe_projection = host_project_safe_selection(
                raw_u_plan=raw_u_plan,
                paired_b_plan=paired_b_plan,
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
        selected_plan = (
            raw_u_plan
            if safe_projection_mode == "disabled"
            else paired_b_plan
            if str(safe_projection["host_projected_safe_selection"]) == str(paired_b_plan.policy_name)
            else raw_u_plan
        )
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
                    "runtime_safe_joint": bool(safe_projection_mode != "disabled"),
                    "safe_projection_mode": safe_projection_mode,
                    "raw_u_policy": raw_u_name,
                    "paired_b_policy": paired_b_name,
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
            execution_capability_required="multiphase_pending_window",
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
            "raw_u_policy_name": raw_u_name,
            "paired_b_policy_name": paired_b_name,
            "safe_projection_mode": safe_projection_mode,
            "requested_bucket_mode": str(self._requested_bucket_mode()),
            "effective_bucket_mode": str(self._effective_bucket_mode()),
            "requested_bucket_rows": int(self.config.bucket_rows),
            "effective_bucket_rows": int(self.config.bucket_rows),
            "default_weights": dict((raw_u_plan.diagnostics or {}).get("default_weights", {})),
            "requested_weights": dict((raw_u_plan.diagnostics or {}).get("requested_weights", {})),
            "effective_weights": dict((raw_u_plan.diagnostics or {}).get("effective_weights", {})),
            "consumed_weights": dict((raw_u_plan.diagnostics or {}).get("consumed_weights", {})),
            "safe_selected_policy": str(selected_plan.policy_name),
            "safe_selection_margin": float(
                safe_projection["host_projected_paired_b_estimated_makespan"]
                - safe_projection["host_projected_raw_u_estimated_makespan"]
            ),
            "safe_comparison_is_strict_common_core": bool(raw_u_name == paired_b_name),
            "raw_u_plan_policy": str(raw_u_plan.policy_name),
            "paired_b_plan_policy": str(paired_b_plan.policy_name),
            "raw_plan_digest": stable_hash(raw_u_plan.to_dict()),
            "paired_b_plan_digest": stable_hash(paired_b_plan.to_dict()),
            "selected_plan_digest": stable_hash(selected_plan.to_dict()),
            "paired_b_build_count": 0 if safe_projection_mode == "disabled" else 1,
            "host_projection_count": 0 if safe_projection_mode == "disabled" else 1,
            "runtime_policy_equivalent_of": effective_policy,
            "service_demand_model": "rows_from_pre_transport_phase_ready_context",
            "bundle_bytes_per_row": int(self._bundle_bytes_per_row(phase_ctx=phase_ctx)),
            },
        )
        global_joint_window_plan = dict(self._runtime_state.read("global_joint_window_plan") or {})
        global_joint_window_plan["host_projected_safe_selection"] = dict(safe_projection)
        self._runtime_state.write("global_joint_window_plan", global_joint_window_plan)
        self._runtime_state.write("ideal_raw_u_makespan", float(safe_projection["ideal_raw_u_estimated_makespan"]))
        self._runtime_state.write("ideal_paired_b_makespan", float(safe_projection["ideal_paired_b_estimated_makespan"]))
        self._runtime_state.write("host_projected_raw_u_makespan", float(safe_projection["host_projected_raw_u_estimated_makespan"]))
        self._runtime_state.write("host_projected_paired_b_makespan", float(safe_projection["host_projected_paired_b_estimated_makespan"]))
        self._runtime_state.write("raw_plan_digest", stable_hash(raw_u_plan.to_dict()))
        self._runtime_state.write("paired_b_plan_digest", stable_hash(paired_b_plan.to_dict()))
        self._runtime_state.write("selected_plan_digest", stable_hash(selected_plan.to_dict()))
        self._runtime_state.write("paired_b_build_count", 0 if safe_projection_mode == "disabled" else 1)
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
            safe_projection["host_projected_paired_b_estimated_makespan"]
            if str(selected_plan.policy_name) == str(paired_b_plan.policy_name)
            else safe_projection["host_projected_raw_u_estimated_makespan"]
            ),
        )
        self._runtime_state.write(
            "ideal_estimated_makespan",
            float(
            safe_projection["ideal_paired_b_estimated_makespan"]
            if str(selected_plan.policy_name) == str(paired_b_plan.policy_name)
            else safe_projection["ideal_raw_u_estimated_makespan"]
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
            ideal_raw_u_makespan=float(safe_projection["ideal_raw_u_estimated_makespan"]),
            ideal_paired_b_makespan=float(safe_projection["ideal_paired_b_estimated_makespan"]),
            host_projected_raw_u_makespan=float(safe_projection["host_projected_raw_u_estimated_makespan"]),
            host_projected_paired_b_makespan=float(safe_projection["host_projected_paired_b_estimated_makespan"]),
            host_projected_estimated_makespan=float(self._runtime_state.read("host_projected_estimated_makespan", 0.0)),
            ideal_estimated_makespan=float(self._runtime_state.read("ideal_estimated_makespan", 0.0)),
            safe_selected_policy=str(selected_plan.policy_name),
            raw_u_policy_name=raw_u_name,
            paired_b_policy_name=paired_b_name,
        )

    def _compile_async_local_phase_plan(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
    ) -> PhaseExecutionPlan:
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            raise RuntimeError(f"missing prepared runtime joint plan for {layer_name} {phase}")
        if str(phase) == "P0":
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
            )
            matrix_unit = "rows"
        else:
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix")) or [])
            )
            matrix_unit = "rows"
            if not matrix:
                matrix = tuple(
                    tuple(int(value) for value in row)
                    for row in (self._runtime_state.read("p1_inferred_from_p0") or [])
                )
        if not matrix:
            raise RuntimeError(f"missing global row matrix for async local materialization {layer_name} {phase}")
        global_contexts = reconstruct_global_phase_contexts_from_byte_matrix(
            local_context=local_context,
            matrix=matrix,
            matrix_unit="rows",
        )
        compiled_local_context = next(
            (context for context in global_contexts if int(context.global_rank) == int(local_context.global_rank)),
            local_context,
        )
        canonical_tasks = build_phase_canonical_tasks(
            phase=str(phase),
            matrix_rows=matrix,
            bucket_rows=int(self.config.bucket_rows),
        )
        bucket_summary = summarize_bucket_tasks(canonical_tasks)
        compilation = compile_schedule(
            PlanCompilationRequest(
                logical_plan=getattr(prepared_plan, "logical_plan"),
                local_context=compiled_local_context,
                global_contexts=global_contexts,
                canonical_tasks=canonical_tasks,
                phase=str(phase),
                tensor_role="hidden_states" if str(phase) == "P1" else "dispatch_bundle",
                rank_context={
                    "global_rank": int(compiled_local_context.global_rank),
                    "local_rank": int(compiled_local_context.local_rank),
                },
                compilation_options=CompilationOptions(
                    bucket_rows=int(self.config.bucket_rows),
                    p0_weight=float(self.config.p0_weight),
                    p1_reservation_weight=float(self.config.p1_reservation_weight),
                    p2_hint_weight=float(self.config.p2_hint_weight),
                    debug_trace=not self._is_perf_profile(),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    legacy_compiler_bridge=bool(getattr(self.config, "legacy_compiler_bridge", False)),
                ),
                prepared_plan=prepared_plan,
                prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                legacy_phase_policy_name=str(self._effective_phase_policy_name() or "routersense_p0p1p2_hint"),
            )
        )
        compiled = compilation.execution_plan
        self._runtime_state.write("compiler_id", str(compilation.audit.compiler_id))
        self._runtime_state.write("logical_plan_digest", str(compilation.audit.logical_plan_digest))
        self._runtime_state.write("compiled_plan_digest", str(compilation.audit.compiled_plan_digest))
        self._runtime_state.write("canonical_task_digest", str(compilation.audit.task_digest))
        self._runtime_state.write("canonical_task_count", int(compilation.audit.task_count))
        self._runtime_state.write("canonical_task_total_rows", int(compilation.audit.total_rows))
        self._runtime_state.write(
            "legacy_secondary_policy_invocation_count",
            int(compilation.audit.metrics.get("legacy_secondary_policy_invocation_count", 0) or 0),
        )
        self._runtime_state.write(
            "legacy_secondary_policy_call_count",
            int(compilation.audit.metrics.get("legacy_secondary_policy_call_count", 0) or 0),
        )
        self._runtime_state.write(
            "direct_compiler_selected_count",
            int(compilation.audit.metrics.get("direct_compiler_selected_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_compare_count",
            int(compilation.audit.metrics.get("compiler_shadow_compare_count", 0) or 0),
        )
        self._runtime_state.write("compiler_shadow_status", str(compilation.audit.metrics.get("shadow_status", "")))
        self._runtime_state.write(
            "compiler_shadow_plan_hash_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_plan_hash_matches_legacy", False)),
        )
        self._runtime_state.write("compiler_shadow_plan_hash", str(compilation.audit.metrics.get("shadow_plan_hash", "")))
        self._runtime_state.write(
            "compiler_shadow_missing_task_count",
            int(compilation.audit.metrics.get("shadow_missing_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_extra_task_count",
            int(compilation.audit.metrics.get("shadow_extra_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_execution_order_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_execution_order_matches_legacy", False)),
        )
        return replace(
            compiled,
            execution_mode="joint_window_async_p2p",
            metrics={
                **compiled.metrics,
                "requested_bucket_mode": str(self._requested_bucket_mode()),
                "effective_bucket_mode": str(self._effective_bucket_mode()),
                "requested_bucket_rows": int(self.config.bucket_rows),
                "effective_bucket_rows": int(self.config.bucket_rows),
                "canonical_bucket_task_summary": bucket_summary,
                "joint_window_async_local_materialization": True,
                "p1_planning_collective_count": 0 if str(phase) == "P1" else int(compiled.metrics.get("p1_planning_collective_count", 0) or 0),
                "prediction_extra_collective_count": 0,
                "preflight_mode": "compact" if self._is_perf_profile() else "full",
                "emit_detailed_task_artifacts": not self._is_perf_profile(),
            },
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
        from rs.runtime.online.megatron_ep.pending_window.policy_adapter import get_or_build_prepared_priority_cache
        from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
        from rs.scheduling.validation import stable_hash

        per_peer = tuple(int(value) for value in observation_p1.per_peer_bytes)
        num_peers = len(per_peer)
        if num_peers <= 0:
            return
        p1_bundle = build_traffic_matrix_bundle(
            per_peer_bytes=per_peer,
            world_size=max(int(len(self.ep_group_ranks) or 0), num_peers),
            device=torch.device(str(getattr(observation_p1, "device", "cpu"))),
            group=self.ep_process_group,
        )
        layer_id = parse_layer_id(layer_name)
        next_layer_id = self._next_layer_id(layer_name)
        actual_dispatch_by_layer = dict(self._runtime_state.read("actual_dispatch_by_layer", {}) or {})
        dispatch_entry = actual_dispatch_by_layer.get(str(layer_id), {})
        dispatch_matrix = tuple(
            tuple(int(value) for value in row)
            for row in dispatch_entry.get("matrix", p1_bundle.matrix)
        )
        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
        prediction_entry = predicted_dispatch_by_layer.get(str(next_layer_id))
        active_prediction = self._runtime_state.read("active_next_dispatch_prediction")
        predictor_name = ""
        prediction_digest = ""
        prediction_confidence = 0.0
        prediction_evaluation_eligible = True
        prediction_is_oracle = False
        if (
            isinstance(active_prediction, dict)
            and active_prediction
            and str(active_prediction.get("target_layer_id", "")) == str(next_layer_id)
        ):
            forecast_matrix = tuple(
                tuple(int(value) for value in row)
                for row in active_prediction.get("forecast_matrix", [])
            )
            p2_matrix_source = "active_next_dispatch_prediction"
            predictor_name = str(active_prediction.get("predictor_name", ""))
            prediction_digest = str(active_prediction.get("matrix_digest", ""))
            prediction_confidence = float(active_prediction.get("confidence", 0.0) or 0.0)
            prediction_evaluation_eligible = bool(active_prediction.get("evaluation_eligible", True))
            prediction_is_oracle = bool(active_prediction.get("is_oracle", False))
        elif isinstance(prediction_entry, dict) and prediction_entry:
            forecast_matrix = tuple(
                tuple(int(value) for value in row)
                for row in prediction_entry.get("matrix", [])
            )
            p2_matrix_source = "predicted_next_dispatch"
            predictor_name = str(prediction_entry.get("predictor_name", ""))
            prediction_digest = str(prediction_entry.get("matrix_digest", ""))
            prediction_confidence = float(prediction_entry.get("confidence", 0.0) or 0.0)
            prediction_evaluation_eligible = bool(prediction_entry.get("evaluation_eligible", True))
            prediction_is_oracle = bool(prediction_entry.get("is_oracle", False))
        else:
            fallback = self._build_online_predictor().predict(
                prediction_input=PredictionInput(
                    run_id_digest=digest_text(self.run_id),
                    layer_id=str(layer_id),
                    next_layer_id=str(next_layer_id),
                    rank=int(self.rank),
                    world_size=num_peers,
                    current_dispatch_matrix_digest=str(dispatch_entry.get("matrix_digest", "")),
                    current_dispatch_total_bytes=int(dispatch_entry.get("total_bytes", 0) or 0),
                    current_dispatch_nonzero_edges=int(dispatch_entry.get("nonzero_edge_count", 0) or 0),
                    metadata={
                        "fallback": True,
                        "previous_dispatch_matrix": (
                            actual_dispatch_by_layer.get(str(int(layer_id) - 1), {}).get("matrix")
                            if str(layer_id).isdigit()
                            else None
                        ),
                    },
                ),
                current_dispatch_matrix=dispatch_matrix,
            )
            forecast_matrix = fallback.matrix
            p2_matrix_source = "copy_current_dispatch_fallback"
            predictor_name = fallback.predictor_name
            prediction_digest = fallback.matrix_digest
            prediction_confidence = float(fallback.confidence)
            prediction_evaluation_eligible = bool(fallback.evaluation_eligible)
            prediction_is_oracle = bool(fallback.is_oracle)
        row_sums = [int(sum(row)) for row in forecast_matrix]
        col_sums = [
            int(sum(forecast_matrix[row_idx][col_idx] for row_idx in range(len(forecast_matrix))))
            for col_idx in range(len(forecast_matrix[0]) if forecast_matrix else 0)
        ]
        forecast_digest = stable_hash(
            {
                "forecast_matrix": [list(row) for row in forecast_matrix],
                "layer": layer_name,
                "source": p2_matrix_source,
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
            }
        )
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
                source=p2_matrix_source,
                digest=forecast_digest,
                oracle=prediction_is_oracle,
                evaluation_eligible=prediction_evaluation_eligible,
                matrix_shape=(num_peers, num_peers),
                matrix_total_bytes=int(sum(row_sums)),
                matrix=forecast_matrix,
                metadata={
                    "p2_matrix_source": p2_matrix_source,
                    "predictor_name": predictor_name,
                    "prediction_digest": prediction_digest,
                    "p2_matrix_is_replicated_local_row": False,
                    "p2_matrix_row_sums": row_sums,
                    "p2_matrix_col_sums": col_sums,
                    "p2_matrix_total_bytes": int(sum(row_sums)),
                },
            ),
            options=GlobalReadySetOptions(
                scheduling_mode="runtime_lookahead",
                information_mode="p0_p1_p2",
                prediction_confidence=float(prediction_confidence),
                p0_weight=float(self.config.p0_weight),
                p1_reservation_weight=float(self.config.p1_reservation_weight),
                p2_hint_weight=float(self.config.p2_hint_weight),
                max_waves=256,
            ),
            p0_dispatch_matrix=dispatch_matrix,
            p1_return_matrix=p1_bundle.matrix,
            p2_next_dispatch_forecast_matrix=forecast_matrix,
        )
        policy = RouterSenseMultiphaseLookaheadPolicy(
            information_mode="p0_p1_p2",
            p0_weight=self.config.p0_weight,
            p1_reservation_weight=self.config.p1_reservation_weight,
            p2_hint_weight=self.config.p2_hint_weight,
        )
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
        self._runtime_state.write("prepared_plan", prepared)
        self._runtime_state.write("plan_created_at_us", int(time.time() * 1e6))
        self._runtime_state.write("plan_source_layer", layer_name)
        self._runtime_state.write("p2_matrix_source", p2_matrix_source)
        self._runtime_state.write("p2_matrix_is_replicated_local_row", False)
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.write("predicted_row_sums", row_sums)
        self._runtime_state.write("predicted_col_sums", col_sums)
        self._runtime_state.write("p2_matrix_source", p2_matrix_source)
        self._runtime_state.write("p2_matrix_total_bytes", int(sum(row_sums)))
        self._runtime_state.write("p2_matrix_row_sums", row_sums)
        self._runtime_state.write("p2_matrix_col_sums", col_sums)
        self._runtime_state.write("p2_matrix_is_replicated_local_row", False)
        self._runtime_state.write("p2_matrix_shape", [num_peers, num_peers])
        self._runtime_state.write("p2_matrix_gather_time_us", float(p1_bundle.gather_time_us))
        self._runtime_state.write("p2_matrix_gather_status", str(p1_bundle.matrix_source))
        self._runtime_state.write("p2_matrix_gather_call_count", int(p1_bundle.gather_call_count))
        self._runtime_state.write("prepared_priority_mode", "mapped_p2_tiebreak")
        self._runtime_state.write("has_real_p1_reservation", False)
        self._runtime_state.write("p1_reservation_row_sums", [])
        self._runtime_state.write("p1_reservation_col_sums", [])
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.remove("prepared_priority_cache", None)
        cache_build_start_ns = time.monotonic_ns()
        _, _, cache_build_time_us = get_or_build_prepared_priority_cache(
            shared_state=self._runtime_state,
            prepared_plan=prepared,
        )
        cache_build_end_ns = time.monotonic_ns()
        self._timeline(
            "prepared_window_plan_stored",
            layer_name=layer_name,
            window_key=prepared.window_key,
            forecast_digest=prepared.forecast_digest,
            applies_from_layer_id=prepared.applies_from_layer_id,
            p2_matrix_source=p2_matrix_source,
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
            prepared_priority_cache_build_time_us=float(cache_build_time_us),
            prepared_priority_cache_total_time_us=(cache_build_end_ns - cache_build_start_ns) / 1000.0,
            p2_matrix_gather_time_us=float(p1_bundle.gather_time_us),
            p2_matrix_gather_status=str(p1_bundle.matrix_source),
            p2_matrix_gather_call_count=int(p1_bundle.gather_call_count),
            predictor_name=predictor_name,
            prediction_digest=prediction_digest,
            prediction_confidence=float(prediction_confidence),
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
            "p2_matrix_source": str(self._runtime_state.read("p2_matrix_source", "")),
            "p2_matrix_total_bytes": int(self._runtime_state.read("p2_matrix_total_bytes", 0) or 0),
            "p2_matrix_row_sums": list(self._runtime_state.read("p2_matrix_row_sums", []) or []),
            "p2_matrix_col_sums": list(self._runtime_state.read("p2_matrix_col_sums", []) or []),
            "p2_matrix_is_replicated_local_row": bool(self._runtime_state.read("p2_matrix_is_replicated_local_row", False)),
            "predictor_name": str(self._runtime_state.read("predictor_name", "")),
            "prediction_digest": str(self._runtime_state.read("prediction_digest", "")),
            "prepared_window_key": str(metrics.get("prepared_window_key", "")),
            "source_logical_plan_hash": str(metrics.get("source_logical_plan_hash", "")),
            "wave_count": len(plan.waves),
            "bucket_count": sum(len(wave.bucket_tasks) for wave in plan.waves),
            "hint_edges_consumed": int(metrics.get("hint_edges_consumed", 0) or 0),
            "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
            "prepared_plan_order_preserved": bool(metrics.get("prepared_plan_order_preserved", False)),
        }
        if not self._is_perf_profile():
            record["bucket_order"] = list(metrics.get("bucket_order", []))
        self.pending_window_driver_records.append(record)

    # Tensor/debug capture and context builders

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
            "forward_epoch": int(self._forward_epoch),
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

    def begin_forward(self, *, forward_epoch: int | None = None) -> None:
        if forward_epoch is None:
            self._forward_epoch += 1
        else:
            self._forward_epoch = int(forward_epoch)
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        self._runtime_state.write("forward_start_ns", int(time.monotonic_ns()))
        self._runtime_state.write("forward_end_ns", 0)

    def end_forward(self) -> dict[str, Any]:
        active_transport = self._active_transport is not None
        has_active_prediction = bool(self._runtime_state.read("active_next_dispatch_prediction"))
        self._runtime_state.write("forward_end_ns", int(time.monotonic_ns()))
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        return {
            "forward_epoch": int(self._forward_epoch),
            "active_transport_cleared": bool(active_transport),
            "stale_prediction_cleared": bool(has_active_prediction),
            "valid": not active_transport,
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
        if self._layer_selected(layer_name):
            self._runtime_state.metrics.selected_p0_hook_count = int(self._runtime_state.metrics.selected_p0_hook_count) + 1
        self._timeline("before_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
        sync_fn = getattr(dispatcher, "_maybe_dtoh_and_synchronize", None)
        if callable(sync_fn):
            try:
                tokens_per_expert = getattr(dispatcher, "tokens_per_expert", None)
                synchronized = sync_fn("before_ep_alltoall", tokens_per_expert)
                if synchronized is not None:
                    dispatcher.tokens_per_expert = synchronized
            except Exception:
                pass
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
        if self._policy_uses_joint_window_plan():
            self._store_runtime_joint_plan_from_p0(
                layer_name=layer_name,
                phase_ctx=phase_ctx,
                observation_p0=pretransport,
                actual_p0_full_row_matrix=actual_p0_full_row_matrix,
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
                agreement_start_ns = time.monotonic_ns()
                agreement = self._agree_joint_plan_digest(
                    layer_name=layer_name,
                    phase="P0",
                    prepared_plan=self._runtime_state.read("prepared_plan"),
                )
                if not bool(agreement.get("valid", False)):
                    raise RuntimeError(f"global joint plan digest mismatch for {layer_name} P0")
                plan = self._compile_async_local_phase_plan(
                    layer_name=layer_name,
                    phase="P0",
                    local_context=phase_ctx,
                )
                agreement_end_ns = time.monotonic_ns()
                self._record_planning_timing(
                    layer_name=layer_name,
                    phase="P0",
                    stage="agree_global_joint_plan_digest",
                    start_ns=agreement_start_ns,
                    end_ns=agreement_end_ns,
                    global_plan_digest=str(agreement.get("global_plan_digest", "")),
                )
                if self.observation_recorder is not None:
                    self.observation_recorder.record_scheduled_plan(
                        scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                    )
                self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
                self._runtime_state.write("before_async_p2p_phase_count", int(self._runtime_state.read("before_async_p2p_phase_count", 0) or 0) + 1)
                self._timeline(
                    "phase_execution_plan_agreed",
                    layer_name=layer_name,
                    phase_name="P0",
                    plan_hash=plan.plan_hash,
                    wave_count=len(plan.waves),
                    bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                    execution_mode=plan.execution_mode,
                    global_plan_digest=str(agreement.get("global_plan_digest", "")),
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
                self.observation_recorder.record_scheduled_plan(
                    scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                )
            self._record_control_replay_trace(phase_ctx=phase_ctx, plan=plan)
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

    def before_token_combine(self, *, layer_name: str, dispatcher: Any, packed_hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        if self._layer_selected(layer_name):
            self._runtime_state.metrics.selected_p1_hook_count = int(self._runtime_state.metrics.selected_p1_hook_count) + 1
        self._timeline("before_token_combine_enter", layer_name=layer_name, phase_name="P1")
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
                self.observation_recorder.record_scheduled_plan(
                    scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                )
            self._record_control_replay_trace(phase_ctx=phase_ctx, plan=plan)
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
            if observation_p1 is not None and self.config.p2_hint_mode == "calibrated_artifact" and not self._is_joint_window_async_mode():
                self._store_prepared_plan(layer_name=layer_name, observation_p1=observation_p1)
            if self._is_joint_window_async_mode():
                self._runtime_state.write("prepared_plan", None)
                self._runtime_state.remove("prepared_priority_cache", None)
            if observation_p1 is not None:
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
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_state_records)

    def export_prepared_plan_bindings(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_plan_bindings)

    def export_release_events(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.release_events)

    def export_window_schedule_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_schedule_shadows)

    def export_prepared_phase_plan_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_phase_plan_shadows)

    def export_pending_window_driver_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.pending_window_driver_records)

    def export_planning_timing_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.planning_timing_records)

    def export_control_replay_traces(self) -> list[dict[str, Any]]:
        if not self._replay_trace_enabled():
            return []
        return self._export_list(self.control_replay_traces)

    def export_prediction_audits(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prediction_audits)

    def export_expert_route_traces(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_route_traces")

    def export_source_expert_counts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_source_expert_counts")

    def export_expert_to_traffic_audits(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_to_traffic_audits")

    def export_expert_trace_warnings(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_trace_warnings")

    def export_assertions(self) -> dict[str, Any]:
        return dict(self.assertion_state)

    def export_prepared_plan_summary(self) -> dict[str, Any]:
        return build_prepared_plan_summary(runtime_state=self._runtime_state)

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
