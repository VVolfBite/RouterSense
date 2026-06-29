from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .schemas import AblationRecord


def _feature_matrix(records: list[AblationRecord]) -> list[list[float]]:
    return [
        [
            record.effective_gate_weight,
            record.router_probability,
            float(record.topk_rank),
            record.top1_top2_gap,
            record.routing_entropy,
            float(record.layer_id),
            float(record.expert_id),
        ]
        for record in records
    ]


def train_calibrator(records: list[AblationRecord], config, output_dir: Path) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=config.seed)
    features = _feature_matrix(records)
    targets = [record.delta_nll for record in records]
    model.fit(features, targets)
    joblib.dump(model, output_dir / "calibrator.joblib")
    return model


def evaluate_calibrator(calibrator, test_records: list[AblationRecord]) -> dict:
    features = _feature_matrix(test_records)
    targets = [record.delta_nll for record in test_records]
    predictions = calibrator.predict(features)
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
        "num_records": len(test_records),
    }

