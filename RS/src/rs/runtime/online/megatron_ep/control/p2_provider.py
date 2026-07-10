"""P2 hint provider：为在线 phase policy 提供未来压力提示。

控制面在这里负责：
- 选择 hint provider
- 从 prepared-plan 共享状态抽取 phase-local 优先级
- 产出 FutureDemandHint 给 lifecycle 和 phase policy 消费

它只生成 hint，不直接做调度或执行。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from rs.runtime.online.megatron_ep.phase.contracts import FutureDemandHint
from rs.scheduling.contracts import PreparedWindowPlan

from .p2_contracts import P2HintRequest


class P2HintProvider(Protocol):
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        ...


class NoP2HintProvider:
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        return FutureDemandHint(hint_mode="none", hint_digest="none", hint_source="no_p2_hint_provider")


class DeterministicStubP2HintProvider:
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        payload = {
            "plan_key": request.plan_key,
            "layer_id": request.layer_id,
            "phase": request.phase,
            "global_rank": request.global_rank,
            "local_rank": request.local_rank,
            "ep_group_ranks": list(request.ep_group_ranks),
        }
        digest = _digest(payload)
        return FutureDemandHint(
            hint_mode="deterministic_stub",
            hint_digest=digest,
            hint_source="deterministic_stub_from_current_plan_key",
            metadata={"tag": f"stub:{request.phase}:{request.layer_id}:{request.global_rank}", "digest": digest},
        )


class CalibratedArtifactP2HintProvider:
    def __init__(self, *, shared_state: dict[str, Any]) -> None:
        self._shared_state = shared_state

    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        prepared_plan = self._shared_state.get("prepared_plan")
        if prepared_plan is None:
            return FutureDemandHint(
                hint_mode="none",
                hint_digest="none",
                hint_source="no_prepared_plan_available",
                metadata={"requested_layer_id": str(request.layer_id)},
            )
        plan_source_layer = str(self._shared_state.get("plan_source_layer", ""))
        plan_created_at_us = int(self._shared_state.get("plan_created_at_us", 0) or 0)
        forecast_digest = str(getattr(prepared_plan, "forecast_digest", ""))
        digest = hashlib.sha256(f"{forecast_digest}:{request.layer_id}".encode("utf-8")).hexdigest()[:16]
        plan_priority = extract_prepared_plan_priority(prepared_plan)
        return FutureDemandHint(
            hint_mode="calibrated_artifact",
            hint_digest=digest,
            hint_source=f"calibrated_artifact_from_layer_{plan_source_layer}",
            metadata={
                "source_layer": plan_source_layer,
                "window_key": str(getattr(prepared_plan, "window_key", "")),
                "plan_created_at_us": plan_created_at_us,
                "source_logical_plan_hash": _digest(getattr(prepared_plan, "logical_plan").to_dict()),
                "forecast_digest": forecast_digest,
                "applies_from_layer_id": str(getattr(prepared_plan, "applies_from_layer_id", "")),
                "p2_matrix_source": str(self._shared_state.get("p2_matrix_source", "")),
                "p2_matrix_is_replicated_local_row": bool(self._shared_state.get("p2_matrix_is_replicated_local_row", False)),
                "predictor_name": str(self._shared_state.get("predictor_name", "")),
                "prediction_digest": str(self._shared_state.get("prediction_digest", "")),
                "predicted_row_sums": list(self._shared_state.get("predicted_row_sums", ()) or ()),
                "predicted_col_sums": list(self._shared_state.get("predicted_col_sums", ()) or ()),
                "prepared_priority_mode": str(self._shared_state.get("prepared_priority_mode", "mapped_p2_tiebreak")),
                "has_real_p1_reservation": bool(self._shared_state.get("has_real_p1_reservation", False)),
                "p1_reservation_row_sums": list(self._shared_state.get("p1_reservation_row_sums", ()) or ()),
                "p1_reservation_col_sums": list(self._shared_state.get("p1_reservation_col_sums", ()) or ()),
                **plan_priority,
            },
        )


def build_p2_hint_provider(mode: str, *, shared_state: dict[str, Any] | None = None) -> P2HintProvider:
    if mode == "none":
        return NoP2HintProvider()
    if mode == "deterministic_stub":
        return DeterministicStubP2HintProvider()
    if mode == "calibrated_artifact":
        if shared_state is None:
            raise ValueError("p2_hint_mode='calibrated_artifact' requires shared_state")
        return CalibratedArtifactP2HintProvider(shared_state=shared_state)
    raise ValueError(f"Unsupported p2_hint_mode={mode!r}")


def extract_prepared_plan_priority(prepared_plan: PreparedWindowPlan | Any) -> dict[str, Any]:
    """Extract phase-local edge priority hints from a prepared logical window plan.

    The online executor can only bind tasks after the current phase layout exists.
    This payload therefore carries logical edge/wave preferences, not tensor
    offsets or future P2 executable work.
    """

    preferred_edges: list[dict[str, Any]] = []
    preferred_waves: list[dict[str, Any]] = []
    stale_edges: list[dict[str, Any]] = []
    seen_edges: dict[tuple[str, int, int], int] = {}
    seen_preferred_edges: set[tuple[str, int, int]] = set()
    source_layer_id = str(getattr(prepared_plan, "created_at_layer_id", ""))
    target_layer_id = str(getattr(prepared_plan, "applies_from_layer_id", ""))
    logical_plan = getattr(prepared_plan, "logical_plan")
    for wave in getattr(logical_plan, "waves", ()):
        wave_edges: list[dict[str, Any]] = []
        for flow in getattr(wave, "flows", ()):
            logical_phase = str(getattr(flow, "phase", ""))
            runtime_phase = "P0" if logical_phase in {"p2_next_dispatch", "P2", "p2"} else _runtime_phase_name(logical_phase)
            edge_key = (runtime_phase, int(getattr(flow, "src_rank")), int(getattr(flow, "dst_rank")))
            priority = seen_edges.setdefault(edge_key, len(seen_edges))
            edge = {
                "phase": runtime_phase,
                "src_rank": edge_key[1],
                "dst_rank": edge_key[2],
                "priority": priority,
                "origin_phase": logical_phase,
                "origin_flow_id": str(getattr(flow, "flow_id", "")),
                "byte_count": int(getattr(flow, "byte_count", 0)),
                "wave_id": int(getattr(wave, "wave_id", 0)),
                "source_layer_id": source_layer_id,
                "target_layer_id": target_layer_id,
                "forecast_digest": str(getattr(prepared_plan, "forecast_digest", "")),
            }
            if logical_phase in {"p2_next_dispatch", "P2", "p2"}:
                wave_edges.append(edge)
                if edge_key not in seen_preferred_edges:
                    preferred_edges.append(edge)
                    seen_preferred_edges.add(edge_key)
            else:
                stale_edges.append(edge)
        if wave_edges:
            preferred_waves.append({"wave_id": int(getattr(wave, "wave_id", 0)), "edges": wave_edges})
    return {
        "preferred_edges": preferred_edges,
        "preferred_waves": preferred_waves,
        "preferred_edge_count": len(preferred_edges),
        "preferred_wave_count": len(preferred_waves),
        "mapped_p2_edge_count": len(preferred_edges),
        "stale_p0_p1_edge_count_ignored": len(stale_edges),
        "stale_prepared_edges": stale_edges,
    }


def _runtime_phase_name(phase: str) -> str:
    normalized = phase.lower()
    if normalized in {"p0", "p0_dispatch", "dispatch"}:
        return "P0"
    if normalized in {"p1", "p1_return", "combine", "return"}:
        return "P1"
    return phase


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
