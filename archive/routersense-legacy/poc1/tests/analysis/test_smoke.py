from __future__ import annotations

from pathlib import Path

from routesense_poc1.analysis import analyze_records, write_report
from routesense_poc1.core.schemas import AblationRecord


def _fake_record(expert_id: int, delta_nll: float) -> AblationRecord:
    return AblationRecord(
        run_id="smoke",
        document_id=0,
        window_id=0,
        layer_id=0,
        layer_path="model.layers.0.mlp",
        token_pos=3,
        expert_id=expert_id,
        topk_rank=expert_id,
        router_logit=0.1 * expert_id,
        router_probability=0.5,
        effective_gate_weight=0.2 + expert_id,
        top1_top2_gap=0.1,
        routing_entropy=0.2,
        baseline_nll=1.0,
        ablated_nll=1.0 + delta_nll,
        delta_nll=delta_nll,
    )


def test_summary_has_all_metrics(tmp_path: Path) -> None:
    summary = analyze_records([_fake_record(0, 0.1), _fake_record(1, 0.05)])
    assert "pairwise_ranking_accuracy" in summary
    assert "strategies" in summary
    report_path = tmp_path / "report.md"
    write_report(summary, report_path)
    assert report_path.exists()
