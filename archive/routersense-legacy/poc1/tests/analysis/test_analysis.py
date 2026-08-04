from __future__ import annotations

from routesense_poc1.analysis import analyze_records
from routesense_poc1.core.schemas import AblationRecord


def _record(
    *,
    window_id: int,
    layer_id: int,
    expert_id: int,
    topk_rank: int,
    delta_nll: float,
    effective_gate_weight: float,
    router_probability: float,
    router_logit: float,
    top1_top2_gap: float,
    routing_entropy: float,
) -> AblationRecord:
    return AblationRecord(
        run_id="test",
        document_id=0,
        window_id=window_id,
        layer_id=layer_id,
        layer_path=f"layer.{layer_id}",
        token_pos=7,
        expert_id=expert_id,
        topk_rank=topk_rank,
        router_logit=router_logit,
        router_probability=router_probability,
        effective_gate_weight=effective_gate_weight,
        top1_top2_gap=top1_top2_gap,
        routing_entropy=routing_entropy,
        baseline_nll=1.0,
        ablated_nll=1.0 + delta_nll,
        delta_nll=delta_nll,
    )


def test_analyze_records_emits_factor_diagnostics() -> None:
    records = [
        _record(
            window_id=0,
            layer_id=0,
            expert_id=0,
            topk_rank=0,
            delta_nll=0.1,
            effective_gate_weight=0.9,
            router_probability=0.8,
            router_logit=3.0,
            top1_top2_gap=0.4,
            routing_entropy=0.2,
        ),
        _record(
            window_id=0,
            layer_id=0,
            expert_id=1,
            topk_rank=1,
            delta_nll=-0.1,
            effective_gate_weight=0.1,
            router_probability=0.2,
            router_logit=0.5,
            top1_top2_gap=0.4,
            routing_entropy=0.2,
        ),
        _record(
            window_id=1,
            layer_id=0,
            expert_id=0,
            topk_rank=0,
            delta_nll=0.2,
            effective_gate_weight=0.8,
            router_probability=0.7,
            router_logit=2.0,
            top1_top2_gap=0.3,
            routing_entropy=0.3,
        ),
        _record(
            window_id=1,
            layer_id=0,
            expert_id=1,
            topk_rank=1,
            delta_nll=-0.2,
            effective_gate_weight=0.2,
            router_probability=0.3,
            router_logit=0.8,
            top1_top2_gap=0.3,
            routing_entropy=0.3,
        ),
    ]
    summary = analyze_records(records)
    assert "routing_factor_diagnostics" in summary
    assert "effective_gate_weight" in summary["routing_factor_diagnostics"]
    assert "router_probability_min" in summary["routing_factor_strategy_subset"]
    assert summary["pairwise_total"] > 0


def test_raw_routing_and_router_probability_min_match() -> None:
    records = [
        _record(
            window_id=0,
            layer_id=0,
            expert_id=0,
            topk_rank=0,
            delta_nll=0.1,
            effective_gate_weight=0.9,
            router_probability=0.9,
            router_logit=3.0,
            top1_top2_gap=0.2,
            routing_entropy=0.4,
        ),
        _record(
            window_id=0,
            layer_id=0,
            expert_id=1,
            topk_rank=1,
            delta_nll=-0.05,
            effective_gate_weight=0.1,
            router_probability=0.1,
            router_logit=0.5,
            top1_top2_gap=0.2,
            routing_entropy=0.4,
        ),
    ]
    summary = analyze_records(records)
    assert (
        summary["strategies"]["raw_routing"]["mean_delta_nll"]
        == summary["strategies"]["router_probability_min"]["mean_delta_nll"]
    )
