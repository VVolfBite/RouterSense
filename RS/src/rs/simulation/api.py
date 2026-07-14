from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts import EvaluationTaskSet, MaterializedPlan, WindowPlan


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
        unresolved = tuple(task.task_id for task in task_set.tasks)
        model_digest = f"simulation_phase_a:{spec.service_model}:{spec.port_model}"
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


__all__ = ["CommonTaskSetSimulator", "SimulationResult", "SimulationSpec", "Simulator"]
