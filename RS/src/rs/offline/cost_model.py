from __future__ import annotations

from rs.core.contracts import EvaluationSpec, EvaluationTask


class EvaluationCostModel:
    @staticmethod
    def task_duration(task: EvaluationTask, spec: EvaluationSpec) -> float:
        task.validate(world_size=int(spec.world_size))
        spec.validate()
        return float(task.byte_count) / float(spec.bandwidth)

    @staticmethod
    def wave_duration(tasks: tuple[EvaluationTask, ...], spec: EvaluationSpec) -> float:
        spec.validate()
        if not tasks:
            return 0.0
        return float(spec.launch_cost) + max(EvaluationCostModel.task_duration(task, spec) for task in tasks)
