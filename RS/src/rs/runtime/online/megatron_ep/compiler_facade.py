from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from rs.runtime.offline.replay_unified import CanonicalBucketTask, CanonicalBucketizer
from rs.scheduling.contracts import LogicalSchedulePlan
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.phase_execution_utils import materialize_local_execution_plan
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


class UnifiedScheduleCompiler:
    compiler_id = "unified_schedule_compiler"

    def compile(self, request: PlanCompilationRequest) -> CompilationResult:
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
        else:
            abstract_plan = request.logical_plan.diagnostics.get("abstract_phase_execution_plan")
            if abstract_plan is None:
                raise ValueError("UnifiedScheduleCompiler currently requires prepared_plan or abstract_phase_execution_plan bridge")
            plan = materialize_local_execution_plan(
                local_context=request.local_context,
                abstract_plan=abstract_plan,
            )
            legacy_bridge = False
        task_digest = CanonicalBucketizer.digest(request.canonical_tasks)
        total_rows = int(sum(task.row_count for task in request.canonical_tasks))
        send_count, recv_count, local_copy_count = _local_counts(plan, global_rank=int(request.local_context.global_rank))
        plan_metrics = dict(plan.metrics or {})
        plan_metrics["compiler_id"] = self.compiler_id
        plan_metrics["legacy_secondary_policy_invocation_count"] = int(1 if legacy_bridge else 0)
        plan_metrics["logical_plan_policy_id"] = str(request.logical_plan.policy_name)
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
                logical_plan_digest=str(request.logical_plan.diagnostics.get("logical_plan_digest", request.logical_plan.diagnostics.get("source_logical_plan_hash", ""))),
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
    "compile_schedule",
]
