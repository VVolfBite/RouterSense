from __future__ import annotations

from pathlib import Path

from routesense_poc1.calibration import evaluate_calibrator, train_calibrator
from routesense_poc1.schemas import AblationRecord, RunConfig


def _record(document_id: int, window_id: int, expert_id: int, delta_nll: float) -> AblationRecord:
    topk_rank = expert_id % 2
    return AblationRecord(
        run_id="test",
        document_id=document_id,
        window_id=window_id,
        layer_id=document_id % 3,
        layer_path=f"layer.{document_id % 3}",
        token_pos=7,
        expert_id=expert_id,
        topk_rank=topk_rank,
        router_logit=float(expert_id) * 0.25,
        router_probability=0.1 + 0.05 * expert_id,
        effective_gate_weight=0.2 + 0.03 * expert_id,
        top1_top2_gap=0.05 * (document_id + 1),
        routing_entropy=0.3 + 0.02 * window_id,
        baseline_nll=1.0,
        ablated_nll=1.0 + delta_nll,
        delta_nll=delta_nll,
    )


def test_calibration_train_and_evaluate(tmp_path: Path) -> None:
    records = []
    for document_id in range(6):
        for window_id in range(2):
            for expert_id in range(2):
                delta_nll = 0.01 * document_id + 0.02 * expert_id + 0.005 * window_id
                records.append(_record(document_id, window_id, expert_id, delta_nll))
    bundle = train_calibrator(records, RunConfig(seed=7), tmp_path)
    metrics = evaluate_calibrator(bundle, records, output_dir=tmp_path)
    assert bundle.feature_names
    assert bundle.feature_family in {"routing_only", "state_like", "combined"}
    assert "candidate_feature_sets" in metrics
    assert "family_best_feature_sets" in metrics
    assert metrics["train_metrics"]["num_records"] > 0
    assert metrics["validation_metrics"]["num_records"] > 0
    assert metrics["test_metrics"]["num_records"] > 0
