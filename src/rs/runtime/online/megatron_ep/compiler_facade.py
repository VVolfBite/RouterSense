from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import LogicalSchedulePlan
from rs.scheduling.bucketizer import CanonicalBucketTask, CanonicalBucketizer
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_local.common import build_transfer_layouts_and_tasks, finalize_execution_plan
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan
from rs.scheduling.validation import stable_hash
from rs.runtime.guards import InvariantContext, RouterSenseInvariantError, invariant_mode_allows_diagnostic_bridge, invariant_mode_forbids_diagnostic_bridge, require_invariant


@dataclass(frozen=True)
class CompilationOptions:
    bucket_rows: int
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0
    debug_trace: bool = False
    compiler_id: str = "unified_schedule_compiler"
    invariant_mode: str = "diagnostic"
    diagnostic_compiler_fallback: bool = False


@dataclass(frozen=True)
class PlanCompilationRequest:
    logical_plan: LogicalSchedulePlan
    local_context: PhaseReadyContext
    global_contexts: tuple[PhaseReadyContext, ...]
    canonical_tasks: tuple[CanonicalBucketTask, ...]
    phase: str
    tensor_role: str
    rank_context: dict[str, Any]
    compilation_options: CompilationOptions
    prepared_plan: Any | None = None
    prepared_priority_cache: dict[str, Any] | None = None
    phase_policy_name: str = ""


@dataclass(frozen=True)
class CompilationAudit:
    compiler_id: str
    task_digest: str
    task_count: int
    total_rows: int
    phase: str
    phase_policy_invoked: bool
    logical_plan_digest: str
    compiled_plan_digest: str
    local_send_task_count: int
    local_recv_task_count: int
    local_copy_task_count: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompilationResult:
    execution_plan: PhaseExecutionPlan
    audit: CompilationAudit


class ScheduleCompiler(Protocol):
    def compile(self, request: PlanCompilationRequest) -> CompilationResult:
        ...


@dataclass(frozen=True)
class _PhaseBucketWindowLike:
    p0_truth_rows: tuple[tuple[int, ...], ...]
    p1_truth_rows: tuple[tuple[int, ...], ...]
    p2_truth_rows: tuple[tuple[int, ...], ...]


def build_phase_canonical_tasks(
    *,
    phase: str,
    matrix_rows: tuple[tuple[int, ...], ...],
    bucket_rows: int,
) -> tuple[CanonicalBucketTask, ...]:
    zero_matrix = tuple(tuple(0 for _ in row) for row in matrix_rows)
    phase_name = str(phase)
    window_like = _PhaseBucketWindowLike(
        p0_truth_rows=matrix_rows if phase_name == "P0" else zero_matrix,
        p1_truth_rows=matrix_rows if phase_name == "P1" else zero_matrix,
        p2_truth_rows=matrix_rows if phase_name == "P2" else zero_matrix,
    )
    return CanonicalBucketizer(bucket_rows=int(bucket_rows)).bucketize(window_like)


def _compiled_task_digest(tasks: tuple[Any, ...]) -> str:
    if all(hasattr(task, "to_tuple") for task in tasks):
        return CanonicalBucketizer.digest(tasks)
    return stable_hash(
        [task.to_dict() if hasattr(task, "to_dict") else repr(task) for task in tasks]
    )


def _local_counts(plan: PhaseExecutionPlan, *, global_rank: int) -> tuple[int, int, int]:
    send = recv = local_copy = 0
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            if int(task.src_rank) == int(task.dst_rank):
                local_copy += 1
            elif int(task.src_rank) == int(global_rank):
                send += 1
            elif int(task.dst_rank) == int(global_rank):
                recv += 1
    return send, recv, local_copy


def _phase_flow_name(phase: str) -> str:
    return {
        "P0": "p0_dispatch",
        "P1": "p1_return",
        "P2": "p2_next_dispatch",
    }.get(str(phase), str(phase))


def _task_src_rank(task: Any) -> int:
    if hasattr(task, "src_group_rank"):
        return int(getattr(task, "src_group_rank"))
    return int(getattr(task, "src_rank"))


def _task_dst_rank(task: Any) -> int:
    if hasattr(task, "dst_group_rank"):
        return int(getattr(task, "dst_group_rank"))
    return int(getattr(task, "dst_rank"))


def _task_row_offset(task: Any) -> int:
    if hasattr(task, "row_offset"):
        return int(getattr(task, "row_offset"))
    return int(getattr(task, "sender_offset_rows", 0))


def _payload_bytes(*, row_count: int, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
    elements_per_row = 1
    for dim in shape_suffix:
        elements_per_row *= int(dim)
    return int(row_count) * int(elements_per_row) * int(element_size_bytes)


def _edge_task_template(tasks: list[Any], *, edge: tuple[int, int]) -> tuple[Any, int]:
    ordered = sorted(
        tasks,
        key=lambda item: (
            int(item.sender_offset_rows),
            int(item.receiver_offset_rows),
            int(item.bucket_ordinal),
            str(item.task_id),
        ),
    )
    if not ordered:
        raise ValueError(f"missing physical tasks for edge {edge}")
    first = ordered[0]
    expected_sender = int(first.sender_offset_rows)
    expected_receiver = int(first.receiver_offset_rows)
    total_rows = 0
    common = (
        str(first.bundle_id),
        int(first.segment_ordinal),
        int(first.source_peer_index),
        int(first.destination_peer_index),
        str(first.packed_send_layout_id),
        str(first.canonical_receive_layout_id),
        tuple(
            (
                str(payload.tensor_role),
                str(payload.dtype),
                tuple(int(value) for value in payload.shape_suffix),
                int(payload.element_size_bytes),
                str(payload.packed_layout_id),
            )
            for payload in first.payload_slices
        ),
    )
    for task in ordered:
        task_common = (
            str(task.bundle_id),
            int(task.segment_ordinal),
            int(task.source_peer_index),
            int(task.destination_peer_index),
            str(task.packed_send_layout_id),
            str(task.canonical_receive_layout_id),
            tuple(
                (
                    str(payload.tensor_role),
                    str(payload.dtype),
                    tuple(int(value) for value in payload.shape_suffix),
                    int(payload.element_size_bytes),
                    str(payload.packed_layout_id),
                )
                for payload in task.payload_slices
            ),
        )
        if task_common != common:
            raise ValueError(f"non-uniform physical task metadata for edge {edge}")
        if int(task.sender_offset_rows) != expected_sender:
            raise ValueError(f"non-contiguous sender offsets for edge {edge}")
        if int(task.receiver_offset_rows) != expected_receiver:
            raise ValueError(f"non-contiguous receiver offsets for edge {edge}")
        rows = int(task.row_count)
        if rows <= 0:
            raise ValueError(f"non-positive physical task rows for edge {edge}")
        expected_sender += rows
        expected_receiver += rows
        total_rows += rows
    return first, int(total_rows)


def _logical_flow_task(
    *,
    template: Any,
    phase_name: str,
    edge: tuple[int, int],
    edge_offset_rows: int,
    row_count: int,
    logical_wave_id: int,
    edge_ordinal: int,
) -> Any:
    sender_offset = int(template.sender_offset_rows) + int(edge_offset_rows)
    receiver_offset = int(template.receiver_offset_rows) + int(edge_offset_rows)
    payload_slices = tuple(
        replace(
            payload,
            sender_offset_rows=sender_offset,
            receiver_offset_rows=receiver_offset,
            row_count=int(row_count),
            payload_byte_count=_payload_bytes(
                row_count=int(row_count),
                shape_suffix=tuple(int(value) for value in payload.shape_suffix),
                element_size_bytes=int(payload.element_size_bytes),
            ),
        )
        for payload in template.payload_slices
    )
    return replace(
        template,
        task_id=(
            f"{phase_name}:{int(edge[0])}->{int(edge[1])}:"
            f"logical_wave:{int(logical_wave_id)}:chunk:{int(edge_ordinal)}"
        ),
        phase=str(phase_name),
        bucket_ordinal=int(edge_ordinal),
        sender_offset_rows=sender_offset,
        receiver_offset_rows=receiver_offset,
        row_count=int(row_count),
        byte_count=int(payload_slices[0].payload_byte_count) if payload_slices else 0,
        payload_slices=payload_slices,
    )


def _build_direct_phase_plan(request: PlanCompilationRequest) -> tuple[PhaseExecutionPlan | None, dict[str, Any]]:
    transfer_layouts, physical_tasks, build_stats = build_transfer_layouts_and_tasks(
        local_context=request.local_context,
        global_contexts=request.global_contexts,
        bucket_rows=int(request.compilation_options.bucket_rows),
        return_stats=True,
    )
    phase_name = str(request.phase)
    logical_phase_name = _phase_flow_name(phase_name)

    edge_to_physical: dict[tuple[int, int], list[Any]] = {}
    for task in physical_tasks:
        if str(task.phase) != phase_name:
            continue
        edge_to_physical.setdefault((int(task.src_rank), int(task.dst_rank)), []).append(task)

    edge_templates: dict[tuple[int, int], Any] = {}
    edge_total_rows: dict[tuple[int, int], int] = {}
    for edge, tasks in edge_to_physical.items():
        template, total_rows = _edge_task_template(tasks, edge=edge)
        edge_templates[edge] = template
        edge_total_rows[edge] = int(total_rows)

    canonical_rows: dict[tuple[int, int], int] = {}
    synthesized_canonical_tasks = not bool(request.canonical_tasks)
    if synthesized_canonical_tasks:
        canonical_rows = dict(edge_total_rows)
    else:
        for task in request.canonical_tasks:
            if str(task.phase) != phase_name:
                continue
            edge = (_task_src_rank(task), _task_dst_rank(task))
            canonical_rows[edge] = canonical_rows.get(edge, 0) + int(task.row_count)
    if canonical_rows != edge_total_rows:
        raise ValueError(
            "canonical tasks do not match the physical transfer layout: "
            f"canonical={canonical_rows} physical={edge_total_rows}"
        )

    edge_consumed: dict[tuple[int, int], int] = {edge: 0 for edge in edge_total_rows}
    edge_ordinals: dict[tuple[int, int], int] = {edge: 0 for edge in edge_total_rows}
    compiled_tasks: list[Any] = []
    shadow_waves: list[PlanWave] = []
    logical_to_compiled_wave: list[dict[str, int]] = []
    next_wave_id = 0
    logical_flow_count = 0

    for logical_wave in request.logical_plan.waves:
        wave_tasks: list[Any] = []
        used_sources: set[int] = set()
        used_destinations: set[int] = set()
        for flow in logical_wave.flows:
            if str(flow.phase) != logical_phase_name:
                continue
            rows = int(getattr(flow, "byte_count", 0) or 0)
            if rows <= 0:
                continue
            edge = (int(flow.src_rank), int(flow.dst_rank))
            if edge not in edge_templates:
                raise ValueError(f"logical plan contains an unexpected phase edge {edge}")
            if edge[0] in used_sources or edge[1] in used_destinations:
                raise ValueError(
                    f"logical wave {logical_wave.wave_id} violates endpoint matching for edge {edge}"
                )
            start = int(edge_consumed[edge])
            end = start + rows
            if end > int(edge_total_rows[edge]):
                raise ValueError(
                    f"logical plan over-serves edge {edge}: requested={end} "
                    f"available={edge_total_rows[edge]}"
                )
            ordinal = int(edge_ordinals[edge])
            task = _logical_flow_task(
                template=edge_templates[edge],
                phase_name=phase_name,
                edge=edge,
                edge_offset_rows=start,
                row_count=rows,
                logical_wave_id=int(logical_wave.wave_id),
                edge_ordinal=ordinal,
            )
            wave_tasks.append(task)
            compiled_tasks.append(task)
            edge_consumed[edge] = end
            edge_ordinals[edge] = ordinal + 1
            used_sources.add(edge[0])
            used_destinations.add(edge[1])
            logical_flow_count += 1
        if wave_tasks:
            shadow_waves.append(
                PlanWave(
                    wave_id=int(next_wave_id),
                    phase=phase_name,
                    bucket_tasks=tuple(wave_tasks),
                )
            )
            logical_to_compiled_wave.append(
                {
                    "logical_wave_id": int(logical_wave.wave_id),
                    "compiled_wave_id": int(next_wave_id),
                }
            )
            next_wave_id += 1

    missing_rows = {
        edge: int(total - edge_consumed.get(edge, 0))
        for edge, total in edge_total_rows.items()
        if int(total - edge_consumed.get(edge, 0)) != 0
    }
    if missing_rows:
        raise ValueError(f"logical plan does not fully cover actual phase rows: {missing_rows}")

    diagnostics = {
        "direct_compiler": True,
        "direct_synthesized_canonical_tasks": bool(synthesized_canonical_tasks),
        "direct_missing_task_ids": [],
        "direct_extra_task_ids": [],
        "direct_planned_task_ids": [str(task.task_id) for task in compiled_tasks],
        "direct_logical_flow_count": int(logical_flow_count),
        "direct_flow_chunk_preserved": True,
        "direct_logical_to_compiled_wave": logical_to_compiled_wave,
        "direct_physical_task_count": int(len(physical_tasks)),
    }
    shadow_plan = finalize_execution_plan(
        local_context=request.local_context,
        policy_name=str(request.logical_plan.policy_name),
        policy_version="direct_logical_chunks_v2",
        capabilities=PolicyCapabilities(
            supports_offline=True,
            supports_online_phase_local_execution=True,
            supports_online_multiphase_execution=False,
            uses_current_ready_flows=True,
            uses_blocked_p1_dependency=False,
            uses_p2_forecast=bool(request.phase == "P1"),
            requires_fixed_placement=False,
            evaluation_eligible=True,
        ),
        bucket_rows=int(request.compilation_options.bucket_rows),
        transfer_layouts=transfer_layouts,
        all_tasks=compiled_tasks,
        waves=tuple(shadow_waves),
        diagnostics=diagnostics,
        timing_metrics=build_stats,
    )
    return shadow_plan, {
        "direct_compile_status": "ok",
        "direct_wave_count": int(len(shadow_waves)),
        "direct_task_count": int(len(compiled_tasks)),
        "direct_missing_task_count": 0,
        "direct_extra_task_count": 0,
        "direct_plan_hash": str(shadow_plan.plan_hash),
        "direct_effective_task_count": int(len(compiled_tasks)),
        "direct_logical_flow_count": int(logical_flow_count),
        "direct_flow_chunk_preserved": True,
        "direct_physical_task_count": int(len(physical_tasks)),
    }


class UnifiedScheduleCompiler:
    compiler_id = "unified_schedule_compiler"

    def compile(self, request: PlanCompilationRequest) -> CompilationResult:
        logical_plan_digest = str(
            request.logical_plan.diagnostics.get(
                "logical_plan_digest",
                request.logical_plan.diagnostics.get(
                    "source_logical_plan_hash",
                    stable_hash(request.logical_plan.to_dict()),
                ),
            )
        )
        if request.prepared_plan is not None:
            if not request.canonical_tasks:
                remote_rows = int(
                    sum(
                        int(value)
                        for src, row in enumerate(getattr(request.local_context, "actual_p0_full_row_matrix", ()) or ())
                        for dst, value in enumerate(row)
                        if int(src) != int(dst)
                    )
                )
                require_invariant(
                    not invariant_mode_forbids_diagnostic_bridge(request.compilation_options.invariant_mode)
                    and bool(request.compilation_options.diagnostic_compiler_fallback or invariant_mode_allows_diagnostic_bridge(request.compilation_options.invariant_mode))
                    and remote_rows == 0,
                    context=InvariantContext(
                        stage="compiler",
                        error_code="RS-COMPILER-MISSING-TASKS",
                        rank=int(request.local_context.global_rank),
                        layer_name=str(request.local_context.layer_id),
                        phase=str(request.phase),
                        logical_plan_digest=logical_plan_digest,
                    ),
                    message="canonical tasks are required for prepared-plan runtime compilation",
                    expected="non-empty canonical_tasks or diagnostic legacy bridge with remote_rows=0",
                    actual={"canonical_task_count": 0, "remote_rows": remote_rows, "invariant_mode": request.compilation_options.invariant_mode},
                )
                from rs.scheduling.runtime_bridge.prepared_priority import PreparedPriorityPhasePolicy

                phase_policy = PreparedPriorityPhasePolicy(
                    bucket_rows=int(request.compilation_options.bucket_rows),
                    p0_weight=float(request.compilation_options.p0_weight),
                    p1_reservation_weight=float(request.compilation_options.p1_reservation_weight),
                    p2_hint_weight=float(request.compilation_options.p2_hint_weight),
                )
                plan = phase_policy.build_plan(
                    local_context=request.local_context,
                    global_contexts=request.global_contexts,
                )
                diagnostic_fallback = True
                direct_plan = None
                direct_metrics = {
                    "direct_compile_status": "zero_remote_fallback",
                    "direct_compile_reason": "missing_canonical_tasks",
                    "materializer_secondary_call_count": 1,
                    "direct_compiler_selected_count": 0,
                    "compiler_shadow_compare_count": 0,
                }
            else:
                diagnostic_fallback = True
                try:
                    direct_plan, direct_metrics = _build_direct_phase_plan(request)
                except Exception as exc:  # pragma: no cover - shadow diagnostics must not break runtime
                    direct_plan = None
                    direct_metrics = {
                        "direct_compile_status": "error",
                        "direct_compile_error_type": type(exc).__name__,
                        "direct_compile_error": str(exc),
                    }
                if direct_plan is None:
                    raise ValueError(f"UnifiedScheduleCompiler failed to build a direct phase plan: {direct_metrics}")
                if str(direct_metrics.get("direct_compile_status", "")) != "ok":
                    raise ValueError(f"UnifiedScheduleCompiler direct compile failed: {direct_metrics}")
                if int(direct_metrics.get("direct_missing_task_count", 0) or 0) != 0:
                    raise ValueError(f"UnifiedScheduleCompiler missing logical tasks: {direct_metrics}")
                if int(direct_metrics.get("direct_extra_task_count", 0) or 0) != 0:
                    raise ValueError(f"UnifiedScheduleCompiler emitted unexpected logical tasks: {direct_metrics}")
                plan = direct_plan
                diagnostic_fallback = False
                direct_metrics["direct_cutover_selected"] = True
                direct_metrics["secondary_policy_call_count"] = 0
                direct_metrics["direct_compiler_selected_count"] = 1
                direct_metrics["compiler_shadow_compare_count"] = 0
        else:
            abstract_plan = request.logical_plan.diagnostics.get("abstract_phase_execution_plan")
            if abstract_plan is None:
                raise ValueError("UnifiedScheduleCompiler currently requires prepared_plan or abstract_phase_execution_plan bridge")
            plan = materialize_local_execution_plan(
                local_context=request.local_context,
                abstract_plan=abstract_plan,
            )
            diagnostic_fallback = False
            direct_plan = None
            direct_metrics = {
                "direct_compile_status": "not_applicable",
                "secondary_policy_call_count": 0,
                "direct_compiler_selected_count": 0,
                "compiler_shadow_compare_count": 0,
            }
        effective_tasks = request.canonical_tasks
        if direct_plan is not None:
            effective_tasks = tuple(task for wave in direct_plan.waves for task in wave.bucket_tasks)
        task_digest = _compiled_task_digest(tuple(effective_tasks))
        total_rows = int(sum(task.row_count for task in effective_tasks))
        send_count, recv_count, local_copy_count = _local_counts(plan, global_rank=int(request.local_context.global_rank))
        plan_metrics = dict(plan.metrics or {})
        plan_metrics["compiler_id"] = self.compiler_id
        plan_metrics["secondary_policy_invocation_count"] = int(1 if diagnostic_fallback else 0)
        plan_metrics["secondary_policy_call_count"] = int(direct_metrics.get("secondary_policy_call_count", 1 if diagnostic_fallback else 0) or 0)
        plan_metrics["direct_compiler_selected_count"] = int(direct_metrics.get("direct_compiler_selected_count", 0) or 0)
        plan_metrics["compiler_shadow_compare_count"] = int(direct_metrics.get("compiler_shadow_compare_count", 0) or 0)
        plan_metrics["logical_plan_policy_id"] = str(request.logical_plan.policy_name)
        plan_metrics.update(direct_metrics)
        if direct_plan is not None:
            direct_task_ids = tuple(str(task.task_id) for wave in direct_plan.waves for task in wave.bucket_tasks)
            plan_metrics["shadow_plan_hash"] = str(direct_plan.plan_hash)
            plan_metrics["shadow_plan_hash_matches_legacy"] = False
            plan_metrics["shadow_execution_order_matches_legacy"] = False
            plan_metrics["shadow_legacy_task_count"] = 0
            plan_metrics["shadow_direct_task_count"] = int(len(direct_task_ids))
            plan_metrics["shadow_status"] = "not_compared_direct_cutover"
            plan_metrics["logical_flow_chunk_preserved"] = bool(
                direct_metrics.get("direct_flow_chunk_preserved", False)
            )
            plan_metrics["shadow_missing_task_count"] = int(direct_metrics.get("direct_missing_task_count", 0) or 0)
            plan_metrics["shadow_extra_task_count"] = int(direct_metrics.get("direct_extra_task_count", 0) or 0)
        updated_plan = PhaseExecutionPlan(
            plan_key=dict(plan.plan_key),
            phase=str(plan.phase),
            policy_name=str(plan.policy_name),
            policy_version=str(plan.policy_version),
            control_mode=str(plan.control_mode),
            execution_mode=str(plan.execution_mode),
            transport_mutation=bool(plan.transport_mutation),
            is_shadow_only=bool(plan.is_shadow_only),
            future_hint_mode=str(plan.future_hint_mode),
            root_rank=int(plan.root_rank),
            observation_digest=str(plan.observation_digest),
            plan_hash=str(plan.plan_hash),
            waves=tuple(plan.waves),
            metrics=plan_metrics,
        )
        return CompilationResult(
            execution_plan=updated_plan,
            audit=CompilationAudit(
                compiler_id=self.compiler_id,
                task_digest=task_digest,
                task_count=len(effective_tasks),
                total_rows=total_rows,
                phase=str(request.phase),
                phase_policy_invoked=diagnostic_fallback,
                logical_plan_digest=logical_plan_digest,
                compiled_plan_digest=str(updated_plan.plan_hash),
                local_send_task_count=send_count,
                local_recv_task_count=recv_count,
                local_copy_task_count=local_copy_count,
                metrics={"policy_name": updated_plan.policy_name, **plan_metrics},
            ),
        )


def compile_schedule(request: PlanCompilationRequest) -> CompilationResult:
    return UnifiedScheduleCompiler().compile(request)


__all__ = [
    "CompilationAudit",
    "CompilationOptions",
    "CompilationResult",
    "PlanCompilationRequest",
    "ScheduleCompiler",
    "UnifiedScheduleCompiler",
    "build_phase_canonical_tasks",
    "compile_schedule",
]
