from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.core.contracts import PlanningRequest, WindowPlan

from .exact_small_instance import exact_result_to_window_plan, solve_problem_exact_with_scope


@dataclass(frozen=True)
class OracleSolveResult:
    oracle_id: str
    status: str
    plan: WindowPlan | None
    runtime_ms: float
    best_bound: float | None
    optimality_gap: float | None
    comparable: bool
    comparable_reason: str
    solver_payload: dict[str, Any]


class OracleRegistry:
    @staticmethod
    def supported() -> tuple[str, ...]:
        return ("O_local", "O_joint")

    @staticmethod
    def solve(oracle_id: str, request: PlanningRequest) -> OracleSolveResult:
        from rs.planning.runtime_adapter import _problem_from_planning_request

        normalized = str(oracle_id)
        if normalized not in {"O_local", "O_joint"}:
            raise ValueError(f"unsupported oracle_id {oracle_id!r}")
        request.validate()
        problem = _problem_from_planning_request(request)
        result = solve_problem_exact_with_scope(
            problem,
            time_limit_ms=5000,
            scope="local" if normalized == "O_local" else "joint",
        )
        status = str(result.get("solver_status", "unknown") or "unknown")
        plan = exact_result_to_window_plan(
            result,
            planner_id=normalized,
            planner_family="exact_local" if normalized == "O_local" else "exact_joint",
            request_digest=request.semantic_digest(),
        )
        return OracleSolveResult(
            oracle_id=normalized,
            status=status,
            plan=plan,
            runtime_ms=float(result.get("solver_runtime_ms_wall", 0.0) or 0.0),
            best_bound=None if result.get("best_bound") is None else float(result.get("best_bound")),
            optimality_gap=None if result.get("optimality_gap") is None else float(result.get("optimality_gap")),
            comparable=bool(result.get("supported", False))
            and status == "optimal"
            and bool(result.get("certified_optimal", False))
            and result.get("objective_logical_makespan") is not None,
            comparable_reason=""
            if (
                bool(result.get("supported", False))
                and status == "optimal"
                and bool(result.get("certified_optimal", False))
                and result.get("objective_logical_makespan") is not None
            )
            else str(status),
            solver_payload=dict(result),
        )


__all__ = ["OracleRegistry", "OracleSolveResult"]
