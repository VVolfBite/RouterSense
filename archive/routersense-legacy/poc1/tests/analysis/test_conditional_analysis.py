from __future__ import annotations

from routesense_poc1.analysis.conditional_analysis import (
    analyze_expert_specialization,
    split_by_context_entropy,
    split_by_delta_magnitude,
    split_by_top1_dominance,
)
from routesense_poc1.core.schemas import AblationRecord


def _record(
    window_id: int,
    layer_id: int,
    expert_id: int,
    topk_rank: int,
    delta_nll: float,
    routing_entropy: float,
    top1_top2_gap: float,
) -> AblationRecord:
    return AblationRecord(
        run_id="test",
        document_id=window_id,
        window_id=window_id,
        layer_id=layer_id,
        layer_path=f"layer.{layer_id}",
        token_pos=3,
        expert_id=expert_id,
        topk_rank=topk_rank,
        router_logit=float(expert_id),
        router_probability=0.1 * (topk_rank + 1),
        effective_gate_weight=0.1 * (topk_rank + 1),
        top1_top2_gap=top1_top2_gap,
        routing_entropy=routing_entropy,
        baseline_nll=1.0,
        ablated_nll=1.0 + delta_nll,
        delta_nll=delta_nll,
    )


def test_conditional_analysis_outputs() -> None:
    records = [
        _record(0, 0, 0, 0, 0.1, 0.1, 0.5),
        _record(0, 0, 1, 1, -0.1, 0.1, 0.5),
        _record(1, 0, 0, 0, 0.2, 0.8, 0.05),
        _record(1, 0, 1, 1, -0.2, 0.8, 0.05),
    ]
    entropy = split_by_context_entropy(records)
    gap = split_by_top1_dominance(records)
    magnitude = split_by_delta_magnitude(records, threshold=0.15)
    specialization = analyze_expert_specialization(records)
    assert "low_entropy" in entropy and "high_entropy" in entropy
    assert "small_gap" in gap and "large_gap" in gap
    assert "large_magnitude_groups" in magnitude
    assert 0 in specialization and 1 in specialization
