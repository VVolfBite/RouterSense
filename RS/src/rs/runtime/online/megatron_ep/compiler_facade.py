from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import LogicalSchedulePlan
from rs.scheduling.bucketizer import CanonicalBucketTask, CanonicalBucketizer
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_local.common import build_transfer_layouts_and_tasks, finalize_execution_plan
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan
from rs.scheduling.validation import stable_hash
from rs.runtime.online.megatron_ep.pending_window.policy_adapter import (
    build_phase_policy_fast_path,
    compile_prepared_window_phase_plan,
)


@dataclass(frozen=True)
class CompilationOptions:
    bucket_rows: int
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0
    debug_trace: bool = False
    compiler_id: str = "unified_schedule_compiler"


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
    legacy_phase_policy_name: str = ""


@dataclass(frozen=True)
class CompilationAudit:
    compiler_id: str
    task_digest: str
    task_count: int
    total_rows: int
    phase: str
    legacy_phase_policy_invoked: bool
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


def _build_direct_shadow_plan(request: PlanCompilationRequest) -> tuple[PhaseExecutionPlan | None, dict[str, Any]]:
    if not request.canonical_tasks:
        return None, {"shadow_status": "missing_canonical_tasks"}
    transfer_layouts, all_tasks, build_stats = build_transfer_layouts_and_tasks(
        local_context=request.local_context,
        global_contexts=request.global_contexts,
        bucket_rows=int(request.compilation_options.bucket_rows),
        return_stats=True,
    )
    phase_name = str(request.phase)
    logical_phase_name = _phase_flow_name(phase_name)
    tasks_by_id = {str(task.task_id): task for task in all_tasks}
    edge_to_task_ids: dict[tuple[int, int], list[str]] = {}
    for task in sorted(request.canonical_tasks, key=lambda item: (int(item.src_group_rank), int(item.dst_group_rank), int(item.row_offset), str(item.task_id))):
        if str(task.phase) != phase_name:
            continue
        edge_to_task_ids.setdefault((int(task.src_group_rank), int(task.dst_group_rank)), []).append(str(task.task_id))
    shadow_waves: list[PlanWave] = []
    planned_task_ids: list[str] = []
    for wave in request.logical_plan.waves:
        wave_tasks: list[Any] = []
        seen_ids: set[str] = set()
        for flow in wave.flows:
            if str(flow.phase) != logical_phase_name:
                continue
            for task_id in edge_to_task_ids.get((int(flow.src_rank), int(flow.dst_rank)), []):
                task = tasks_by_id.get(task_id)
                if task is None or task_id in seen_ids:
                    continue
                wave_tasks.append(task)
                seen_ids.add(task_id)
                planned_task_ids.append(task_id)
        if wave_tasks:
            shadow_waves.append(PlanWave(wave_id=int(wave.wave_id), phase=phase_name, bucket_tasks=tuple(wave_tasks)))
    shadow_task_ids = tuple(planned_task_ids)
    missing_task_ids = tuple(sorted(set(tasks_by_id) - set(shadow_task_ids)))
    extra_task_ids = tuple(sorted(set(shadow_task_ids) - set(tasks_by_id)))
    diagnostics = {
        "direct_compiler_shadow": True,
        "shadow_missing_task_ids": list(missing_task_ids),
        "shadow_extra_task_ids": list(extra_task_ids),
        "shadow_planned_task_ids": list(shadow_task_ids),
    }
    shadow_plan = finalize_execution_plan(
        local_context=request.local_context,
        policy_name=str(request.logical_plan.policy_name),
        policy_version="direct_shadow",
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
        all_tasks=[tasks_by_id[task_id] for task_id in shadow_task_ids if task_id in tasks_by_id],
        waves=tuple(shadow_waves),
        diagnostics=diagnostics,
        timing_metrics=build_stats,
    )
    return shadow_plan, {
        "shadow_status": "ok",
        "shadow_wave_count": int(len(shadow_waves)),
        "shadow_task_count": int(len(shadow_task_ids)),
        "shadow_missing_task_count": int(len(missing_task_ids)),
        "shadow_extra_task_count": int(len(extra_task_ids)),
        "shadow_plan_hash": str(shadow_plan.plan_hash),
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
            phase_policy = build_phase_policy_fast_path(
                bucket_rows=int(request.compilation_options.bucket_rows),
                p0_weight=float(request.compilation_options.p0_weight),
                p1_reservation_weight=float(request.compilation_options.p1_reservation_weight),
                p2_hint_weight=float(request.compilation_options.p2_hint_weight),
            )
            plan = compile_prepared_window_phase_plan(
                prepared_plan=request.prepared_plan,
                local_context=request.local_context,
                global_contexts=request.global_contexts,
                bucket_rows=int(request.compilation_options.bucket_rows),
                p0_weight=float(request.compilation_options.p0_weight),
                p1_reservation_weight=float(request.compilation_options.p1_reservation_weight),
                p2_hint_weight=float(request.compilation_options.p2_hint_weight),
                policy_name=str(request.legacy_phase_policy_name or "routersense_p0p1p2_hint"),
                prepared_priority_cache=request.prepared_priority_cache,
                phase_policy=phase_policy,
            )
            legacy_bridge = True
            try:
                shadow_plan, shadow_metrics = _build_direct_shadow_plan(request)
            except Exception as exc:  # pragma: no cover - shadow diagnostics must not break runtime
                shadow_plan = None
                shadow_metrics = {
                    "shadow_status": "error",
                    "shadow_error_type": type(exc).__name__,
                    "shadow_error": str(exc),
                }
            if (
                shadow_plan is not None
                and str(shadow_metrics.get("shadow_status", "")) == "ok"
                and int(shadow_metrics.get("shadow_missing_task_count", 0) or 0) == 0
                and int(shadow_metrics.get("shadow_extra_task_count", 0) or 0) == 0
            ):
                legacy_task_ids = tuple(str(task.task_id) for wave in plan.waves for task in wave.bucket_tasks)
                shadow_task_ids = tuple(str(task.task_id) for wave in shadow_plan.waves for task in wave.bucket_tasks)
                if legacy_task_ids == shadow_task_ids:
                    plan = shadow_plan
                    legacy_bridge = False
                    shadow_metrics["shadow_cutover_selected"] = True
                else:
                    shadow_metrics["shadow_cutover_selected"] = False
            else:
                shadow_metrics["shadow_cutover_selected"] = False
        else:
            abstract_plan = request.logical_plan.diagnostics.get("abstract_phase_execution_plan")
            if abstract_plan is None:
                raise ValueError("UnifiedScheduleCompiler currently requires prepared_plan or abstract_phase_execution_plan bridge")
            plan = materialize_local_execution_plan(
                local_context=request.local_context,
                abstract_plan=abstract_plan,
            )
            legacy_bridge = False
            shadow_plan = None
            shadow_metrics = {"shadow_status": "not_applicable"}
        task_digest = CanonicalBucketizer.digest(request.canonical_tasks)
        total_rows = int(sum(task.row_count for task in request.canonical_tasks))
        send_count, recv_count, local_copy_count = _local_counts(plan, global_rank=int(request.local_context.global_rank))
        plan_metrics = dict(plan.metrics or {})
        plan_metrics["compiler_id"] = self.compiler_id
        plan_metrics["legacy_secondary_policy_invocation_count"] = int(1 if legacy_bridge else 0)
        plan_metrics["logical_plan_policy_id"] = str(request.logical_plan.policy_name)
        plan_metrics.update(shadow_metrics)
        if shadow_plan is not None:
            legacy_task_ids = tuple(str(task.task_id) for wave in plan.waves for task in wave.bucket_tasks)
            shadow_task_ids = tuple(str(task.task_id) for wave in shadow_plan.waves for task in wave.bucket_tasks)
            plan_metrics["shadow_plan_hash"] = str(shadow_plan.plan_hash)
            plan_metrics["shadow_plan_hash_matches_legacy"] = bool(str(shadow_plan.plan_hash) == str(plan.plan_hash))
            plan_metrics["shadow_execution_order_matches_legacy"] = bool(legacy_task_ids == shadow_task_ids)
            plan_metrics["shadow_legacy_task_count"] = int(len(legacy_task_ids))
            plan_metrics["shadow_direct_task_count"] = int(len(shadow_task_ids))
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
                task_count=len(request.canonical_tasks),
                total_rows=total_rows,
                phase=str(request.phase),
                legacy_phase_policy_invoked=legacy_bridge,
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
