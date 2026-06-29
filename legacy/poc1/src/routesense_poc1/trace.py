from __future__ import annotations

from typing import Any

from .metrics import next_token_nll
from .schemas import MoELayerSpec, RoutingContext, RunConfig, Window


def collect_routing_context(
    model: Any,
    tokenizer: Any,
    window: Window,
    moe_layer: MoELayerSpec,
    run_id: str,
) -> list[RoutingContext]:
    import torch  # type: ignore

    device = next(model.parameters()).device
    input_ids = torch.tensor([window.input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_router_logits=True, return_dict=True, use_cache=False)
    baseline_nll = next_token_nll(outputs.logits, input_ids)
    router_logits = outputs.router_logits[moe_layer.layer_index][window.target_pos].detach().float()
    probs = torch.softmax(router_logits, dim=-1)
    top_weights, top_indices = torch.topk(probs, k=moe_layer.top_k, dim=-1)
    if moe_layer.norm_topk_prob:
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    gap = float(top_weights[0].item() - top_weights[1].item()) if moe_layer.top_k >= 2 else 0.0
    contexts: list[RoutingContext] = []
    for topk_rank, expert_id in enumerate(top_indices.tolist()):
        contexts.append(
            RoutingContext(
                run_id=run_id,
                document_id=window.document_id,
                window_id=window.window_id,
                layer_id=moe_layer.layer_index,
                layer_path=moe_layer.module_path,
                token_pos=window.target_pos,
                expert_id=int(expert_id),
                topk_rank=topk_rank,
                router_logit=float(router_logits[expert_id].item()),
                router_probability=float(probs[expert_id].item()),
                effective_gate_weight=float(top_weights[topk_rank].item()),
                top1_top2_gap=gap,
                routing_entropy=entropy,
                baseline_nll=baseline_nll,
            )
        )
    return contexts
