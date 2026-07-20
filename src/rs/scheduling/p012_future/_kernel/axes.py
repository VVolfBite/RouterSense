from __future__ import annotations

"""Orthogonal planner-axis contract for P012-family scheduling.

The scheduling core is intentionally separated from wrappers that decide when
planning runs, how much information is visible, and how a complete plan is
constructed.  Every formal planner is represented unambiguously by the five-axis canonical form::

    <timing>:<horizon>:<scope>:<engine>:<core>

Examples::

    current:p012:local:event:rscf
    current:p012:joint:global:rscf
    future:p012:joint:global:rscf
"""

from dataclasses import dataclass, replace
from typing import Final

TIMINGS: Final[tuple[str, ...]] = ("current", "future")
HORIZONS: Final[tuple[str, ...]] = ("p012", "p0123")
SCOPES: Final[tuple[str, ...]] = ("local", "joint")
ENGINES: Final[tuple[str, ...]] = ("event", "global")
CORES: Final[tuple[str, ...]] = ("gmwd", "rsbc", "rscf")


@dataclass(frozen=True)
class PlannerAxes:
    timing: str
    horizon: str
    scope: str
    engine: str
    core: str

    def __post_init__(self) -> None:
        values = {
            "timing": str(self.timing).lower(),
            "horizon": str(self.horizon).lower(),
            "scope": str(self.scope).lower(),
            "engine": str(self.engine).lower(),
            "core": str(self.core).lower(),
        }
        for key, value in values.items():
            object.__setattr__(self, key, value)
        if self.timing not in TIMINGS:
            raise ValueError(f"unsupported planning timing {self.timing!r}")
        if self.horizon not in HORIZONS:
            raise ValueError(f"unsupported planning horizon {self.horizon!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"unsupported planning scope {self.scope!r}")
        if self.engine not in ENGINES:
            raise ValueError(f"unsupported planning engine {self.engine!r}")
        if self.core not in CORES:
            raise ValueError(f"unsupported planning core {self.core!r}")
        if self.timing == "future" and self.horizon != "p012":
            raise ValueError("Future preparation currently supports the P012 horizon only")

    @property
    def canonical_id(self) -> str:
        return f"{self.timing}:{self.horizon}:{self.scope}:{self.engine}:{self.core}"

    @property
    def requires_prediction(self) -> bool:
        return bool(self.timing == "future" or self.scope == "joint")

    @property
    def planner_family(self) -> str:
        return self.scope

    def with_timing(self, timing: str) -> "PlannerAxes":
        return replace(self, timing=str(timing))

    def with_scope(self, scope: str) -> "PlannerAxes":
        return replace(self, scope=str(scope))

    def with_engine(self, engine: str) -> "PlannerAxes":
        return replace(self, engine=str(engine))

    def with_horizon(self, horizon: str) -> "PlannerAxes":
        return replace(self, horizon=str(horizon))

    def to_dict(self) -> dict[str, str]:
        return {
            "timing": self.timing,
            "horizon": self.horizon,
            "scope": self.scope,
            "engine": self.engine,
            "core": self.core,
            "canonical_id": self.canonical_id,
        }


def parse_planner_axes(planner_id: str) -> PlannerAxes:
    value = str(planner_id).strip().lower()
    parts = value.split(":")
    if len(parts) == 5 and parts[0] in TIMINGS:
        return PlannerAxes(*parts)
    raise ValueError(f"unknown formal planner ID {planner_id!r}; use <timing>:<horizon>:<scope>:<engine>:<core>")


def is_axes_planner_id(planner_id: str) -> bool:
    try:
        parse_planner_axes(planner_id)
    except ValueError:
        return False
    return True


def planner_axis_matrix(*, include_future: bool = True, include_p0123: bool = True) -> tuple[PlannerAxes, ...]:
    rows: list[PlannerAxes] = []
    for timing in TIMINGS:
        if timing == "future" and not include_future:
            continue
        for horizon in HORIZONS:
            if horizon == "p0123" and not include_p0123:
                continue
            if timing == "future" and horizon != "p012":
                continue
            for scope in SCOPES:
                for engine in ENGINES:
                    for core in CORES:
                        rows.append(PlannerAxes(timing, horizon, scope, engine, core))
    return tuple(rows)


__all__ = [
    "CORES",
    "ENGINES",
    "HORIZONS",
    "PlannerAxes",
    "SCOPES",
    "TIMINGS",
    "is_axes_planner_id",
    "parse_planner_axes",
    "planner_axis_matrix",
]
