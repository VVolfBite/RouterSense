"""Prediction contracts and lightweight online heuristics.

Current traffic-matrix predictors (`zero_hint`, `copy_current_dispatch`) are
debug/replay baselines, not faithful FATE-style expert predictors.
"""

from .audit import compare_predicted_to_actual
from .contracts import PredictionAuditRecord, PredictionInput, PredictedTrafficMatrix
from .expert_evaluation import ExpertPredictionMetrics, evaluate_expert_prediction
from .expert_to_traffic import ExpertToTrafficAudit, compare_reconstructed_traffic, source_expert_counts_to_traffic_matrix
from .expert_trace import ExpertRouteRecord, SourceExpertCountMatrix, aggregate_route_records
from .gate_replay_predictor import GateReplayPredictionResult, MockGateReplayPredictor
from .simple_predictors import CopyCurrentDispatchPredictor, ZeroHintPredictor
from .traffic_calibration import TrafficCalibrationAudit, calibrate_traffic_matrix

__all__ = [
    "PredictionAuditRecord",
    "PredictionInput",
    "PredictedTrafficMatrix",
    "ExpertPredictionMetrics",
    "ExpertRouteRecord",
    "ExpertToTrafficAudit",
    "GateReplayPredictionResult",
    "MockGateReplayPredictor",
    "SourceExpertCountMatrix",
    "TrafficCalibrationAudit",
    "ZeroHintPredictor",
    "CopyCurrentDispatchPredictor",
    "aggregate_route_records",
    "calibrate_traffic_matrix",
    "compare_predicted_to_actual",
    "compare_reconstructed_traffic",
    "evaluate_expert_prediction",
    "source_expert_counts_to_traffic_matrix",
]
