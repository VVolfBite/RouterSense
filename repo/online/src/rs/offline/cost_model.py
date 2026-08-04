from __future__ import annotations

from rs.core.contracts import EvaluationSpec, EvaluationTask


class EvaluationCostModel:
    @staticmethod
    def task_duration(task: EvaluationTask, spec: EvaluationSpec) -> float:
        task.validate(world_size=int(spec.world_size))
        spec.validate()
        return float(task.byte_count) / float(spec.bandwidth)

    @staticmethod
    def flow_duration(*, row_count: int, bytes_per_row: int, spec: EvaluationSpec) -> float:
        spec.validate()
        if int(row_count) < 0:
            raise ValueError("row_count must be >= 0")
        if int(bytes_per_row) <= 0:
            raise ValueError("bytes_per_row must be > 0")
        return float(int(row_count) * int(bytes_per_row)) / float(spec.bandwidth)

    @staticmethod
    def wave_duration(tasks: tuple[EvaluationTask, ...], spec: EvaluationSpec) -> float:
        spec.validate()
        if not tasks:
            return 0.0
        return float(spec.launch_cost) + max(
            EvaluationCostModel.flow_duration(
                row_count=int(task.row_count),
                bytes_per_row=max(int(task.byte_count) // max(int(task.row_count), 1), int(spec.bytes_per_row)),
                spec=spec,
            )
            for task in tasks
        )
