from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts import EvaluationSpec, EvaluationTaskSet, MaterializedPlan, WindowPlan
from rs.offline.evaluation import evaluate_window_plan_against_task_set


@dataclass(frozen=True)
class SimulationSpec:
    service_model: str
    task_granularity: str
    launch_cost: float
    bandwidth: float
    bytes_per_row: int
    max_inflight: int
    release_model: str
    port_model: str
    time_unit: str


@dataclass(frozen=True)
class SimulationResult:
    success: bool
    realized_makespan: float | None
    task_timeline: tuple[str, ...]
    completed_tasks: tuple[str, ...]
    unresolved_tasks: tuple[str, ...]
    dependency_violations: tuple[str, ...]
    coverage_valid: bool
    port_valid: bool
    model_digest: str


class Simulator(Protocol):
    def simulate(
        self,
        task_set: EvaluationTaskSet,
        plan: WindowPlan | MaterializedPlan,
        spec: SimulationSpec,
    ) -> SimulationResult:
        ...


class CommonTaskSetSimulator:
    def simulate(
        self,
        task_set: EvaluationTaskSet,
        plan: WindowPlan | MaterializedPlan,
        spec: SimulationSpec,
    ) -> SimulationResult:
        model_digest = f"simulation_v1:{spec.service_model}:{spec.port_model}:{spec.release_model}:{spec.task_granularity}"
        if isinstance(plan, MaterializedPlan):
            unresolved = tuple(task.task_id for task in task_set.tasks)
            return SimulationResult(
                success=False,
                realized_makespan=None,
                task_timeline=(),
                completed_tasks=(),
                unresolved_tasks=unresolved,
                dependency_violations=(),
                coverage_valid=False,
                port_valid=False,
                model_digest=model_digest,
            )
        phase_filter = {"p0_dispatch", "p1_return", "p2_next_dispatch"}
        evaluation = evaluate_window_plan_against_task_set(
            plan=plan,
            task_set=task_set,
            phase_filter=phase_filter,
            spec=EvaluationSpec(
                track="execution_window",
                world_size=int(task_set.world_size),
                task_granularity=str(spec.task_granularity),
                matrix_unit="rows",
                time_unit=str(spec.time_unit),
                cost_model_id=str(spec.service_model),
                release_model=str(spec.release_model),
                return_model="transpose_dispatch",
                full_duplex=str(spec.port_model) == "full_duplex",
                launch_cost=float(spec.launch_cost),
                bytes_per_row=int(spec.bytes_per_row),
                bandwidth=float(spec.bandwidth),
                compute_delay=0.0,
                p2_semantics="actual",
                residual_policy="reject",
            ),
        )
        return SimulationResult(
            success=bool(evaluation.valid),
            realized_makespan=evaluation.realized_makespan,
            task_timeline=tuple(evaluation.completed_tasks),
            completed_tasks=tuple(evaluation.completed_tasks),
            unresolved_tasks=tuple(evaluation.unresolved_tasks),
            dependency_violations=tuple(evaluation.dependency_violations),
            coverage_valid=bool(evaluation.coverage_valid),
            port_valid=bool(evaluation.port_valid),
            model_digest=model_digest,
        )


__all__ = ["CommonTaskSetSimulator", "SimulationResult", "SimulationSpec", "Simulator"]
