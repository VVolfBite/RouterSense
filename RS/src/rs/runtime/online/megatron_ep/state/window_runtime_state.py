"""Typed window-state helpers used by the formal runtime.

These structures used to live under the legacy pending_window namespace.
The formal runtime now depends on this neutral state module instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rs.runtime.online.megatron_ep.contracts import RuntimeObservation
from rs.scheduling.contracts import PreparedWindowPlan

from ..observation.contracts import parse_layer_id


@dataclass(frozen=True)
class WindowReleaseState:
    p0_dispatch_completed_ranks: tuple[int, ...] = ()
    p1_return_materialized_ranks: tuple[int, ...] = ()
    p1_return_completed_ranks: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedPlanBinding:
    source_layer_name: str
    source_layer_id: str
    target_layer_id: str
    window_key: str
    forecast_digest: str
    source_logical_plan_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineWindowState:
    window_key: str
    layer_name: str
    layer_id: str
    ep_group_ranks: tuple[int, ...]
    local_rank: int
    p0_observation: RuntimeObservation | None = None
    p1_observation: RuntimeObservation | None = None
    prepared_plan_binding: PreparedPlanBinding | None = None
    prepared_plan: PreparedWindowPlan | None = None
    release_state: WindowReleaseState = field(default_factory=WindowReleaseState)

    def to_record(self) -> dict[str, Any]:
        payload = {
            "window_key": self.window_key,
            "layer_name": self.layer_name,
            "layer_id": self.layer_id,
            "ep_group_ranks": list(self.ep_group_ranks),
            "local_rank": self.local_rank,
            "has_p0_observation": self.p0_observation is not None,
            "has_p1_observation": self.p1_observation is not None,
            "prepared_plan_binding": None if self.prepared_plan_binding is None else self.prepared_plan_binding.to_dict(),
            "release_state": self.release_state.to_dict(),
        }
        if self.p0_observation is not None:
            payload["p0_per_peer_bytes"] = list(int(v) for v in self.p0_observation.per_peer_bytes)
        if self.p1_observation is not None:
            payload["p1_per_peer_bytes"] = list(int(v) for v in self.p1_observation.per_peer_bytes)
        return payload


def bind_prepared_plan(
    *,
    layer_name: str,
    prepared_plan: PreparedWindowPlan | None,
    source_layer_name: str,
    source_logical_plan_hash: str,
) -> PreparedPlanBinding | None:
    if prepared_plan is None:
        return None
    layer_id = parse_layer_id(layer_name)
    if str(prepared_plan.applies_from_layer_id) != str(layer_id):
        return None
    return PreparedPlanBinding(
        source_layer_name=source_layer_name,
        source_layer_id=str(prepared_plan.created_at_layer_id),
        target_layer_id=str(prepared_plan.applies_from_layer_id),
        window_key=str(prepared_plan.window_key),
        forecast_digest=str(prepared_plan.forecast_digest),
        source_logical_plan_hash=source_logical_plan_hash,
    )


def build_window_state(
    *,
    layer_name: str,
    ep_group_ranks: tuple[int, ...],
    local_rank: int,
    p0_observation: RuntimeObservation | None,
    p1_observation: RuntimeObservation | None,
    prepared_plan: PreparedWindowPlan | None,
    prepared_plan_binding: PreparedPlanBinding | None,
    release_state: WindowReleaseState,
) -> OnlineWindowState:
    layer_id = parse_layer_id(layer_name)
    window_key = (
        str(prepared_plan_binding.window_key)
        if prepared_plan_binding is not None
        else f"window:{layer_id}:{','.join(str(rank) for rank in ep_group_ranks)}"
    )
    return OnlineWindowState(
        window_key=window_key,
        layer_name=layer_name,
        layer_id=layer_id,
        ep_group_ranks=ep_group_ranks,
        local_rank=local_rank,
        p0_observation=p0_observation,
        p1_observation=p1_observation,
        prepared_plan_binding=prepared_plan_binding,
        prepared_plan=prepared_plan,
        release_state=release_state,
    )


__all__ = [
    "OnlineWindowState",
    "PreparedPlanBinding",
    "WindowReleaseState",
    "bind_prepared_plan",
    "build_window_state",
]
