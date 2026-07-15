from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.core.contracts import PlanningRequest, WindowPlan

from .exact_small_instance import exact_result_to_logical_plan, solve_problem_exact


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
        from rs.planning.runtime_adapter import _problem_from_planning_request, _window_plan_from_logical_plan

        normalized = str(oracle_id)
        if normalized not in {"O_local", "O_joint"}:
            raise ValueError(f"unsupported oracle_id {oracle_id!r}")
        request.validate()
        problem = _problem_from_planning_request(request)
        result = solve_problem_exact(problem, time_limit_ms=5000)
        status = str(result.get("solver_status", "unknown") or "unknown")
        logical_plan = exact_result_to_logical_plan(result, policy_name=normalized)
        plan = _window_plan_from_logical_plan(
            planner_id=normalized,
            planner_family="exact_local" if normalized == "O_local" else "exact_joint",
            request=request,
            logical_plan=logical_plan,
        )
        return OracleSolveResult(
            oracle_id=normalized,
            status=status,
            plan=plan,
            runtime_ms=float(result.get("time_limit_ms", 0) or 0.0),
            best_bound=None if result.get("best_bound") is None else float(result.get("best_bound")),
            optimality_gap=None if result.get("optimality_gap") is None else float(result.get("optimality_gap")),
            comparable=bool(result.get("supported", False)),
            comparable_reason="" if bool(result.get("supported", False)) else str(status),
            solver_payload=dict(result),
        )


__all__ = ["OracleRegistry", "OracleSolveResult"]
