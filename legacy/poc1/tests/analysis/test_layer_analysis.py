from __future__ import annotations

from routesense_poc1.analysis.layer_analysis import (
    analyze_by_expert,
    analyze_by_layer,
    analyze_oracle_subset,
    compute_rank_layer_heatmap,
)
from routesense_poc1.core.schemas import AblationRecord


def _record(window_id: int, layer_id: int, expert_id: int, topk_rank: int, delta_nll: float) -> AblationRecord:
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
        top1_top2_gap=0.1,
        routing_entropy=0.2,
        baseline_nll=1.0,
        ablated_nll=1.0 + delta_nll,
        delta_nll=delta_nll,
    )


def test_layer_analysis_outputs() -> None:
    records = [
        _record(0, 0, 0, 0, 0.1),
        _record(0, 0, 1, 1, -0.1),
        _record(1, 1, 0, 0, 0.2),
        _record(1, 1, 1, 1, -0.2),
    ]
    by_layer = analyze_by_layer(records)
    by_expert = analyze_by_expert(records)
    oracle_subset = analyze_oracle_subset(records, top_k=1)
    heatmap = compute_rank_layer_heatmap(records)
    assert 0 in by_layer and 1 in by_layer
    assert 0 in by_expert and 1 in by_expert
    assert oracle_subset["selected_group_count"] == 1
    assert len(heatmap) == 2
