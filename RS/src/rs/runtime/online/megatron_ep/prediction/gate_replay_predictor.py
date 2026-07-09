"""CPU/offline contract for a faithful FATE-style gate-replay predictor.

This module intentionally provides only an interface and a mock implementation.
It does not claim to be a real next-layer router execution path yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from rs.scheduling.traffic_matrix import matrix_digest_remote, matrix_nonzero_remote_edge_count, matrix_remote_bytes

from .expert_evaluation import ExpertPredictionMetrics, evaluate_expert_prediction
from .expert_to_traffic import source_expert_counts_to_traffic_matrix
from .expert_trace import SourceExpertCountMatrix


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GateReplayPredictionResult:
    predictor_name: str
    predictor_family: str
    predictor_version: str
    predicted_source_expert_counts: SourceExpertCountMatrix
    predicted_traffic_matrix: Matrix
    predicted_traffic_digest: str
    predicted_remote_bytes: int
    predicted_nonzero_edge_count: int
    requires_router_input: bool
    gpu_collection_required: bool
    expert_metrics: ExpertPredictionMetrics | None = None
    traffic_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predicted_source_expert_counts"] = self.predicted_source_expert_counts.to_dict()
        payload["predicted_traffic_matrix"] = [list(row) for row in self.predicted_traffic_matrix]
        if self.expert_metrics is not None:
            payload["expert_metrics"] = self.expert_metrics.to_dict()
        return payload


class GateReplayPredictor(Protocol):
    predictor_name: str
    predictor_version: str

    def predict_next_layer_expert_counts(
        self,
        current_layer_id: int,
        current_router_input: Any,
        next_layer_router: Any,
        expert_to_rank: Mapping[int, int],
        *,
        top_k: int,
        bytes_per_token: int,
        actual_next_expert_counts: SourceExpertCountMatrix | None = None,
    ) -> GateReplayPredictionResult:
        ...


class MockGateReplayPredictor:
    predictor_name = "gate_replay"
    predictor_family = "fate_style_gate_replay"
    predictor_version = "mock_v1"

    def predict_next_layer_expert_counts(
        self,
        current_layer_id: int,
        current_router_input: Any,
        next_layer_router: Any,
        expert_to_rank: Mapping[int, int],
        *,
        top_k: int,
        bytes_per_token: int,
        actual_next_expert_counts: SourceExpertCountMatrix | None = None,
    ) -> GateReplayPredictionResult:
        world_size = int(current_router_input.get("world_size", 1))
        num_experts = int(current_router_input.get("num_experts", len(expert_to_rank)))
        counts = tuple(
            tuple(int(value) for value in row)
            for row in current_router_input.get(
                "source_expert_counts",
                tuple(tuple(0 for _ in range(num_experts)) for _ in range(world_size)),
            )
        )
        predicted_counts = SourceExpertCountMatrix(
            layer_id=int(current_layer_id + 1),
            world_size=world_size,
            num_experts=num_experts,
            counts=counts,
        )
        predicted_matrix = source_expert_counts_to_traffic_matrix(
            predicted_counts,
            expert_to_rank,
            bytes_per_token=bytes_per_token,
            top_k=top_k,
        )
        expert_metrics = None
        if actual_next_expert_counts is not None:
            expert_metrics = evaluate_expert_prediction(
                predicted_counts,
                actual_next_expert_counts,
                expert_to_rank=expert_to_rank,
                bytes_per_token=bytes_per_token,
                topk=max(16, top_k),
            )
        traffic_metrics = None if expert_metrics is None else {
            "traffic_relative_l1_error": expert_metrics.traffic_relative_l1_error,
            "traffic_cosine_similarity": expert_metrics.traffic_cosine_similarity,
            "traffic_topk_edge_overlap": expert_metrics.traffic_topk_edge_overlap,
        }
        return GateReplayPredictionResult(
            predictor_name=self.predictor_name,
            predictor_family=self.predictor_family,
            predictor_version=self.predictor_version,
            predicted_source_expert_counts=predicted_counts,
            predicted_traffic_matrix=predicted_matrix,
            predicted_traffic_digest=_traffic_digest(predicted_matrix),
            predicted_remote_bytes=int(matrix_remote_bytes(predicted_matrix)),
            predicted_nonzero_edge_count=int(matrix_nonzero_remote_edge_count(predicted_matrix)),
            requires_router_input=True,
            gpu_collection_required=True,
            expert_metrics=expert_metrics,
            traffic_metrics=traffic_metrics,
        )


def _traffic_digest(matrix: Matrix) -> str:
    payload = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = ["GateReplayPredictionResult", "GateReplayPredictor", "MockGateReplayPredictor"]
