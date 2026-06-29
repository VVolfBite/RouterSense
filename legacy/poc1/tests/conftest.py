from __future__ import annotations

from routesense_poc1.core.schemas import AblationRecord


def make_ablation_record(**overrides) -> AblationRecord:
    payload = {
        "run_id": "test",
        "document_id": 0,
        "window_id": 0,
        "layer_id": 0,
        "layer_path": "layer.0",
        "token_pos": 7,
        "expert_id": 0,
        "topk_rank": 0,
        "router_logit": 0.0,
        "router_probability": 0.5,
        "effective_gate_weight": 0.5,
        "top1_top2_gap": 0.1,
        "routing_entropy": 0.2,
        "baseline_nll": 1.0,
        "ablated_nll": 1.1,
        "delta_nll": 0.1,
    }
    payload.update(overrides)
    return AblationRecord(**payload)
