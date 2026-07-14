from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import EvaluationSpec, ExecutionTruth, PlanEvaluation, WindowPlan


def _normalize_phase(name: str) -> str:
    if str(name) in {"p2_next_dispatch_forecast", "p2_next_dispatch"}:
        return "p2_next_dispatch"
    return str(name)


@dataclass(frozen=True)
class OfflineEvaluator:
    def evaluate(
        self,
        plan: WindowPlan,
        truth: ExecutionTruth,
        spec: EvaluationSpec,
    ) -> PlanEvaluation:
        plan.validate()
        truth.validate()
        spec.validate()
        phase_filter = {"p0_dispatch", "p1_return"} if str(spec.track) == "runtime_lookahead" else {"p0_dispatch", "p1_return", "p2_next_dispatch"}
        expected_by_edge = {}
        for task in truth.task_set.tasks:
            phase = _normalize_phase(task.phase)
            if phase not in phase_filter:
                continue
            expected_by_edge[(phase, int(task.src_rank), int(task.dst_rank))] = int(task.row_count)
        seen_by_edge = {key: 0 for key in expected_by_edge}
        completed_edges: dict[tuple[str, int, int], float] = {}
        completed_ids: list[str] = []
        dependency_violations: list[str] = []
        current_time = 0.0
        for wave in plan.waves:
            used_src: set[int] = set()
            used_dst: set[int] = set()
            wave_tasks = []
            for flow in wave.flows:
                phase = _normalize_phase(flow.phase)
                if phase not in phase_filter:
                    continue
                edge_key = (phase, int(flow.src_rank), int(flow.dst_rank))
                expected_rows = expected_by_edge.get(edge_key)
                if expected_rows is None:
                    return PlanEvaluation(
                        valid=False,
                        reason=f"unexpected_task:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                        realized_makespan=None,
                        completed_tasks=tuple(completed_ids),
                        unresolved_tasks=tuple(sorted(f"{item[0]}:{item[1]}->{item[2]}" for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                        dependency_violations=tuple(dependency_violations),
                        coverage_valid=False,
                        port_valid=False,
                    )
                if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                    return PlanEvaluation(
                        valid=False,
                        reason=f"port_conflict:wave_{wave.wave_id}",
                        realized_makespan=None,
                        completed_tasks=tuple(completed_ids),
                        unresolved_tasks=tuple(sorted(f"{item[0]}:{item[1]}->{item[2]}" for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                        dependency_violations=tuple(dependency_violations),
                        coverage_valid=False,
                        port_valid=False,
                    )
                new_rows = int(seen_by_edge[edge_key]) + int(flow.row_count)
                if int(flow.row_count) < 0 or new_rows > int(expected_rows):
                    return PlanEvaluation(
                        valid=False,
                        reason=f"row_count_mismatch:{phase}:{int(flow.src_rank)}->{int(flow.dst_rank)}",
                        realized_makespan=None,
                        completed_tasks=tuple(completed_ids),
                        unresolved_tasks=tuple(sorted(f"{item[0]}:{item[1]}->{item[2]}" for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                        dependency_violations=tuple(dependency_violations),
                        coverage_valid=False,
                        port_valid=False,
                    )
                used_src.add(int(flow.src_rank))
                used_dst.add(int(flow.dst_rank))
                if phase == "p1_return":
                    pending = [
                        key
                        for key in expected_by_edge
                        if key[0] == "p0_dispatch" and key[2] == int(flow.src_rank) and seen_by_edge[key] < expected_by_edge[key]
                    ]
                    if pending:
                        dependency_violations.append(f"p1_return:{int(flow.src_rank)}")
                elif phase == "p2_next_dispatch":
                    pending = [
                        key
                        for key in expected_by_edge
                        if key[0] == "p1_return" and key[2] == int(flow.src_rank) and seen_by_edge[key] < expected_by_edge[key]
                    ]
                    if pending:
                        dependency_violations.append(f"p2_next_dispatch:{int(flow.src_rank)}")
                seen_by_edge[edge_key] = new_rows
                wave_tasks.append((edge_key, int(flow.row_count), int(flow.row_count)))
            if dependency_violations:
                return PlanEvaluation(
                    valid=False,
                    reason="dependency_violation",
                    realized_makespan=None,
                    completed_tasks=tuple(completed_ids),
                    unresolved_tasks=tuple(sorted(f"{item[0]}:{item[1]}->{item[2]}" for item, value in seen_by_edge.items() if value != expected_by_edge[item])),
                    dependency_violations=tuple(dependency_violations),
                    coverage_valid=False,
                    port_valid=True,
                )
            if not wave_tasks:
                continue
            wave_duration = float(spec.launch_cost) + max(float(row_count * int(spec.bytes_per_row)) / float(spec.bandwidth) for _edge, row_count, _byte_count in wave_tasks)
            end_time = current_time + wave_duration
            for edge_key, _row_count, _byte_count in wave_tasks:
                completed_edges[edge_key] = end_time
                completed_ids.append(f"{edge_key[0]}:{edge_key[1]}->{edge_key[2]}")
            current_time = end_time
        unresolved = tuple(sorted(f"{item[0]}:{item[1]}->{item[2]}" for item, value in seen_by_edge.items() if value != expected_by_edge[item]))
        return PlanEvaluation(
            valid=len(unresolved) == 0 and len(dependency_violations) == 0,
            reason=None if len(unresolved) == 0 and len(dependency_violations) == 0 else "incomplete_coverage",
            realized_makespan=current_time if len(unresolved) == 0 and len(dependency_violations) == 0 else None,
            completed_tasks=tuple(completed_ids),
            unresolved_tasks=unresolved,
            dependency_violations=tuple(dependency_violations),
            coverage_valid=len(unresolved) == 0,
            port_valid=True,
            metrics={
                "completed_task_count": len(completed_ids),
                "expected_task_count": len(expected_by_edge),
            },
        )


__all__ = ["OfflineEvaluator"]
