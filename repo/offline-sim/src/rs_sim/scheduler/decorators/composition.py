from __future__ import annotations

"""Orthogonal scheduler composition.

A scheduling algorithm has exactly one registered ordering core.  Scope,
planning cadence, and optional safety selection are outer decorators and never
select an alternate core implementation.
"""

import ast
from dataclasses import dataclass

from rs_sim.scheduler.planning.planner import PlannerScope
from rs_sim.scheduler.decorators.planning_gate import PlanningMode

REGISTERED_CORE_IDS = frozenset({
    "null",
    "fifo",
    "greedy",
    "birkhoff",
    "islip",
    "residual_mwm",
    "fast",
    "aurora",
    "rscf",
    "oracle",
})


@dataclass(frozen=True, slots=True)
class SchedulingAlgorithm:
    core_id: str
    scope: PlannerScope
    planning: PlanningMode
    safe: bool = False

    def __post_init__(self) -> None:
        if self.core_id not in REGISTERED_CORE_IDS:
            raise ValueError(f"unregistered algorithm core {self.core_id!r}")
        if self.safe and self.scope is not PlannerScope.WINDOW_JOINT:
            raise ValueError("safe requires a declared Joint scope")

    @property
    def expression(self) -> str:
        core = f"{self.core_id}()"
        cadence = (
            f"event({core})"
            if self.planning is PlanningMode.EVENT
            else f"global_({core})"
        )
        scoped = (
            f"local({cadence})"
            if self.scope is PlannerScope.PHASE_LOCAL
            else f"joint({cadence})"
        )
        return f"safe({scoped})" if self.safe else scoped


@dataclass(slots=True)
class _Parsed:
    core_id: str | None = None
    scope: PlannerScope | None = None
    planning: PlanningMode | None = None
    safe: bool = False


def _parse_call(node: ast.AST, parsed: _Parsed) -> None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("algorithm expression must contain function-style decorators")
    if node.keywords:
        raise ValueError("algorithm expression does not accept keyword arguments")
    name = node.func.id
    if name in REGISTERED_CORE_IDS:
        if node.args:
            raise ValueError(f"algorithm core {name} takes no arguments")
        if parsed.core_id is not None:
            raise ValueError("algorithm expression contains multiple cores")
        parsed.core_id = name
        return
    if name not in {"local", "joint", "event", "global_", "safe"}:
        raise ValueError(f"unsupported scheduler decorator {name!r}")
    if len(node.args) != 1:
        raise ValueError(f"scheduler decorator {name} requires exactly one argument")
    if name == "local":
        if parsed.scope is not None:
            raise ValueError("algorithm expression contains multiple scope decorators")
        parsed.scope = PlannerScope.PHASE_LOCAL
    elif name == "joint":
        if parsed.scope is not None:
            raise ValueError("algorithm expression contains multiple scope decorators")
        parsed.scope = PlannerScope.WINDOW_JOINT
    elif name == "event":
        if parsed.planning is not None:
            raise ValueError("algorithm expression contains multiple planning decorators")
        parsed.planning = PlanningMode.EVENT
    elif name == "global_":
        if parsed.planning is not None:
            raise ValueError("algorithm expression contains multiple planning decorators")
        parsed.planning = PlanningMode.GLOBAL
    else:
        if parsed.safe:
            raise ValueError("algorithm expression contains multiple safe decorators")
        parsed.safe = True
    _parse_call(node.args[0], parsed)


def parse_algorithm_expression(value: str) -> SchedulingAlgorithm:
    text = str(value).strip()
    if not text:
        raise ValueError("algorithm expression must be non-empty")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid algorithm expression {text!r}") from exc
    parsed = _Parsed()
    _parse_call(tree.body, parsed)
    if parsed.core_id is None:
        raise ValueError("algorithm expression does not contain a registered core")
    if parsed.scope is None:
        raise ValueError("algorithm expression requires local(...) or joint(...)")
    if parsed.planning is None:
        raise ValueError("algorithm expression requires event(...) or global_(...)")
    result = SchedulingAlgorithm(
        core_id=parsed.core_id,
        scope=parsed.scope,
        planning=parsed.planning,
        safe=parsed.safe,
    )
    # Safe is an outer policy over Local/Joint.  Requiring it to be the outer
    # call prevents ambiguous nested semantics while preserving one core.
    if result.safe and not text.startswith("safe("):
        raise ValueError("safe must be the outermost scheduler decorator")
    return result


__all__ = [
    "REGISTERED_CORE_IDS",
    "SchedulingAlgorithm",
    "parse_algorithm_expression",
]
