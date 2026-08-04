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
from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_diagonal_report,
    matrix_digest_remote,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
)

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
        active_prediction = self._shared_state.get("active_next_dispatch_prediction")
        if isinstance(active_prediction, dict) and active_prediction:
            target_layer_id = str(active_prediction.get("target_layer_id", ""))
            if target_layer_id == str(request.layer_id):
                metadata = _build_prediction_metadata(
                    prediction=active_prediction,
                    request=request,
                    shared_state=self._shared_state,
                )
                self._shared_state.setdefault("prediction_consumption_records", []).append(
                    {
                        "requested_layer_id": str(request.layer_id),
                        "requested_phase": str(request.phase),
                        "source_layer_id": str(active_prediction.get("source_layer_id", "")),
                        "target_layer_id": target_layer_id,
                        "prediction_created_stage": str(active_prediction.get("created_at_stage", "")),
                        "prediction_first_consumed_stage": f"before_{str(request.phase).lower()}",
                        "consumer_layer": str(request.layer_id),
                        "consumer_phase": str(request.phase),
                        "consumed_before_p1": str(request.phase) == "P1",
                    }
                )
                return FutureDemandHint(
                    hint_mode="active_prediction",
                    hint_digest=str(active_prediction.get("matrix_digest", "none")),
                    hint_source=f"active_prediction_from_layer_{active_prediction.get('source_layer_id', '')}",
                    metadata=metadata,
                )
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
                "prediction_confidence": float(self._shared_state.get("prediction_confidence", 0.0) or 0.0),
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
    forecast_priority = build_next_p0_priority_from_forecast(
        getattr(prepared_plan, "forecast_matrix", ()),
        source_layer_id=source_layer_id,
        target_layer_id=target_layer_id,
        prediction_confidence=1.0,
        prediction_digest=str(getattr(prepared_plan, "forecast_digest", "")),
    )
    for edge in forecast_priority["preferred_edges"]:
        edge_key = (str(edge["phase"]), int(edge["src_rank"]), int(edge["dst_rank"]))
        if edge_key in seen_preferred_edges:
            continue
        preferred_edges.append(edge)
        seen_preferred_edges.add(edge_key)
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
        "priority_origin": "forecast_matrix" if forecast_priority["mapped_p2_edge_count"] > 0 else "logical_plan_only",
    }


def build_next_p0_priority_from_forecast(
    forecast_matrix: Any,
    *,
    source_layer_id: str,
    target_layer_id: str,
    prediction_confidence: float,
    prediction_digest: str,
) -> dict[str, Any]:
    canonical = canonicalize_remote_matrix(forecast_matrix)
    preferred_edges: list[dict[str, Any]] = []
    scored_edges: list[tuple[float, int, int, dict[str, Any]]] = []
    for src_rank, row in enumerate(canonical):
        row_total = max(1, int(sum(row)))
        for dst_rank, predicted_bytes in enumerate(row):
            if src_rank == dst_rank or int(predicted_bytes) <= 0:
                continue
            normalized_pressure = float(predicted_bytes) / float(row_total)
            edge = {
                "phase": "P0",
                "src_rank": int(src_rank),
                "dst_rank": int(dst_rank),
                "priority": 0,
                "origin_phase": "forecast_matrix",
                "origin_flow_id": f"forecast:{source_layer_id}->{target_layer_id}:{src_rank}->{dst_rank}",
                "byte_count": int(predicted_bytes),
                "predicted_bytes": int(predicted_bytes),
                "normalized_pressure": normalized_pressure,
                "priority_score": float(prediction_confidence) * normalized_pressure,
                "source_layer_id": str(source_layer_id),
                "target_layer_id": str(target_layer_id),
                "prediction_confidence": float(prediction_confidence),
                "prediction_digest": str(prediction_digest or matrix_digest_remote(canonical)),
                "forecast_digest": str(prediction_digest or matrix_digest_remote(canonical)),
                "origin": "forecast_matrix",
            }
            scored_edges.append((edge["priority_score"], int(src_rank), int(dst_rank), edge))
    for priority_index, (_score, _src_rank, _dst_rank, edge) in enumerate(
        sorted(scored_edges, key=lambda item: (-item[0], item[1], item[2]))
    ):
        edge["priority"] = int(priority_index)
        preferred_edges.append(edge)
    return {
        "preferred_edges": preferred_edges,
        "preferred_waves": (),
        "preferred_edge_count": len(preferred_edges),
        "preferred_wave_count": 0,
        "mapped_p2_edge_count": len(preferred_edges),
        "predicted_row_sums": list(matrix_row_sums_remote(canonical)),
        "predicted_col_sums": list(matrix_col_sums_remote(canonical)),
        "forecast_remote_bytes": int(matrix_remote_bytes(canonical)),
        "forecast_nonzero_edge_count": int(matrix_nonzero_remote_edge_count(canonical)),
        "forecast_diagonal_report": matrix_diagonal_report(forecast_matrix),
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


def _build_prediction_metadata(
    *,
    prediction: dict[str, Any],
    request: P2HintRequest,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    forecast_matrix = tuple(tuple(int(value) for value in row) for row in prediction.get("forecast_matrix", ()))
    source_layer_id = str(prediction.get("source_layer_id", ""))
    target_layer_id = str(prediction.get("target_layer_id", request.layer_id))
    payload = build_next_p0_priority_from_forecast(
        forecast_matrix,
        source_layer_id=source_layer_id,
        target_layer_id=target_layer_id,
        prediction_confidence=float(prediction.get("confidence", 0.0) or 0.0),
        prediction_digest=str(prediction.get("matrix_digest", "")),
    )
    return {
        "source_layer": source_layer_id,
        "target_layer": target_layer_id,
        "predictor_name": str(prediction.get("predictor_name", "")),
        "predictor_version": str(prediction.get("predictor_version", "")),
        "prediction_digest": str(prediction.get("matrix_digest", "")),
        "prediction_confidence": float(prediction.get("confidence", 0.0) or 0.0),
        "prediction_created_stage": str(prediction.get("created_at_stage", "")),
        "prediction_first_consumed_stage": f"before_{str(request.phase).lower()}",
        "consumer_layer": str(request.layer_id),
        "consumer_phase": str(request.phase),
        "consumed_before_p1": str(request.phase) == "P1",
        "prepared_priority_mode": str(shared_state.get("prepared_priority_mode", "mapped_p2_tiebreak")),
        "has_real_p1_reservation": bool(shared_state.get("has_real_p1_reservation", False)),
        "p1_reservation_row_sums": list(shared_state.get("p1_reservation_row_sums", ()) or ()),
        "p1_reservation_col_sums": list(shared_state.get("p1_reservation_col_sums", ()) or ()),
        **payload,
    }
