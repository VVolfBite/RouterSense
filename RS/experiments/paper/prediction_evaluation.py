from __future__ import annotations

from pathlib import Path
from typing import Any

from rs.scheduling.traffic_matrix import matrix_digest_remote

from .adapters.prediction_adapter import evaluate_traffic_prediction
from .adapters.scheduling_adapter import execute_policy, replay_window_from_matrices
from .contracts import PredictionEvaluationRecord, RecordMetadata
from .trace_dataset import discover_replay_fixtures, load_replay_fixture, replay_fixture_to_trace_sample
from .traffic_builder import build_traffic_instance


def _zero_matrix_like(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in row) for row in matrix)


def _shuffle_rows(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if len(matrix) <= 1:
        return matrix
    return tuple(matrix[(index + 1) % len(matrix)] for index in range(len(matrix)))


def evaluate_prediction(
    *,
    fixture_dir: Path,
    metadata: RecordMetadata,
    model_id: str,
    model_revision: str,
    joint_policy_id: str,
) -> dict[str, Any]:
    records: list[PredictionEvaluationRecord] = []
    for path in discover_replay_fixtures(fixture_dir):
        fixture = load_replay_fixture(path)
        trace_sample = replay_fixture_to_trace_sample(
            fixture,
            model_id=model_id,
            model_revision=model_revision,
            metadata=metadata,
        )
        traffic = build_traffic_instance(
            trace_sample=trace_sample,
            p0_matrix=fixture["p0_dispatch_matrix"],
            p1_matrix=fixture["p1_return_matrix"],
            p2_matrix=fixture["p2_next_dispatch_matrix"],
            virtual_ep_size=len(fixture["p0_dispatch_matrix"]),
            metadata=metadata,
        )
        replay_window = replay_window_from_matrices(
            fixture_id=trace_sample.trace_sample_id,
            layer_id=int(fixture["metadata"].get("layer_id", 0) or 0),
            p0_matrix=traffic.P0_matrix,
            p1_matrix=traffic.P1_matrix,
            p2_matrix=traffic.P2_truth_matrix,
        )
        zero_rows = _zero_matrix_like(traffic.P2_truth_matrix)
        shuffled_rows = _shuffle_rows(traffic.P2_truth_matrix)
        perfect = execute_policy(
            replay_window=replay_window,
            policy_name=joint_policy_id,
            hint_type="perfect_trace_hint",
            p2_hint_rows=traffic.P2_truth_matrix,
        )
        zero = execute_policy(
            replay_window=replay_window,
            policy_name=joint_policy_id,
            hint_type="zero_hint",
            p2_hint_rows=zero_rows,
            confidence=0.0,
        )
        shuffled = execute_policy(
            replay_window=replay_window,
            policy_name=joint_policy_id,
            hint_type="shuffled_control",
            p2_hint_rows=shuffled_rows,
        )
        zero_metrics = evaluate_traffic_prediction(
            predictor_id="zero",
            current_dispatch_rows=traffic.P0_matrix,
            current_return_rows=traffic.P1_matrix,
            target_next_dispatch_rows=traffic.P2_truth_matrix,
            source_layer_id=str(fixture["metadata"].get("layer_id", "0")),
            target_layer_id=str(int(fixture["metadata"].get("layer_id", 0) or 0) + 1),
        )
        records.append(
            PredictionEvaluationRecord(
                instance_id=traffic.instance_id,
                predictor_id=None,
                input_digest=matrix_digest_remote(traffic.P0_matrix),
                prediction_digest=None,
                truth_digest=matrix_digest_remote(traffic.P2_truth_matrix),
                no_future_leakage=None,
                allowed_input_fields=("current_dispatch_rows", "current_return_rows", "history_dispatch_rows", "source_layer_id", "target_layer_id"),
                prediction_ready_at=None,
                truth_available_at=None,
                raw_prediction_metrics={"status": "MISSING_CAPABILITY"},
                perfect_plan_metrics={"status": "BASELINE", "objective": perfect.get("objective"), "hint_type": "perfect_trace_hint"},
                predicted_plan_metrics={
                    "status": "MISSING_CAPABILITY",
                    "reason": "formal predicted paper path not executed",
                    "allowed_input_fields": ["current_dispatch_rows", "current_return_rows", "history_dispatch_rows", "source_layer_id", "target_layer_id"],
                    "truth_digest_present_in_input": False,
                },
                zero_plan_metrics={"status": "BASELINE", "objective": zero.get("objective"), **zero_metrics},
                shuffled_plan_metrics={"status": "BASELINE", "objective": shuffled.get("objective"), "hint_type": "shuffled_control"},
                prediction_regret=None,
                gain_over_zero=None,
                metadata=metadata,
            )
        )
    return {"records": [record.to_dict() for record in records], "status": "PARTIAL_MISSING_PREDICTED"}
