from __future__ import annotations

import json
from pathlib import Path

from rs.runtime.offline.prediction import (
    FATEStyleHistoryPredictor,
    FATEStyleLinearTrafficPredictor,
    load_predictor_artifact,
    rolling_predictor_records,
    save_predictor_artifact,
    summarize_prediction_records,
)
from rs.runtime.offline.prediction.feature_builder import load_fixture_samples


def _write_fixture_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    fixtures = [
        {
            "num_gpus": 2,
            "p0_dispatch_matrix": [[0, 8], [4, 0]],
            "p1_return_matrix": [[0, 4], [8, 0]],
            "p2_next_dispatch_matrix": [[0, 10], [5, 0]],
            "metadata": {"layer_id": "0", "next_layer_id": "1"},
        },
        {
            "num_gpus": 2,
            "p0_dispatch_matrix": [[0, 10], [5, 0]],
            "p1_return_matrix": [[0, 5], [10, 0]],
            "p2_next_dispatch_matrix": [[0, 12], [6, 0]],
            "metadata": {"layer_id": "1", "next_layer_id": "2"},
        },
        {
            "num_gpus": 2,
            "p0_dispatch_matrix": [[0, 12], [6, 0]],
            "p1_return_matrix": [[0, 6], [12, 0]],
            "p2_next_dispatch_matrix": [[0, 14], [7, 0]],
            "metadata": {"layer_id": "2", "next_layer_id": "3"},
        },
    ]
    for idx, fixture in enumerate(fixtures):
        (path / f"replay_layer_{idx}.json").write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_fate_style_linear_predictor_trains_without_target_leak(tmp_path: Path) -> None:
    fixture_dir = _write_fixture_dir(tmp_path / "fixtures")
    samples = load_fixture_samples(fixture_dir)
    predictor = FATEStyleLinearTrafficPredictor().fit(samples[:2])
    predicted = predictor.predict_matrix(samples[2])
    assert predicted[0][1] >= 0
    assert predicted[1][0] >= 0
    assert predicted != samples[2].target_next_dispatch_matrix


def test_fate_style_history_predictor_exports_and_loads(tmp_path: Path) -> None:
    fixture_dir = _write_fixture_dir(tmp_path / "fixtures")
    samples = load_fixture_samples(fixture_dir)
    predictor = FATEStyleHistoryPredictor().fit(samples[:2])
    artifact_path = tmp_path / "history.json"
    save_predictor_artifact(artifact_path, predictor.to_artifact())
    restored = FATEStyleHistoryPredictor.from_artifact(load_predictor_artifact(artifact_path))
    assert restored.predict_matrix(samples[2]) == predictor.predict_matrix(samples[2])


def test_prediction_records_and_summary_are_available(tmp_path: Path) -> None:
    fixture_dir = _write_fixture_dir(tmp_path / "fixtures")
    records = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")
    assert records
    summary = summarize_prediction_records(records)
    assert summary["record_count"] == len(records)
    assert "mean_relative_l1_error" in summary
    assert "per_layer" in summary
