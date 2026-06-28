from __future__ import annotations

# Router trace helpers for mock and real OLMoE-style routing inspection.

import re
from typing import Any

from .schemas import BatchRoutingSummary, BucketSummary, RouterTrace


class RouterTraceIntegrationError(RuntimeError):
    pass


class MockRouterTraceProvider:
    """A demo-only provider for tests and local scaffolding.

    This does not attempt to run the real OLMoE model. It returns a structured trace
    containing the fields required by the PoC1 interface.
    """

    def __call__(self, text: str, layer_path: str) -> RouterTrace:
        return RouterTrace(
            layer_path=layer_path,
            layer_id=0,
            token_pos=0,
            target_token_id=0,
            target_token_text="<mock>",
            next_token_id=0,
            next_token_text="<mock>",
            router_logits=[1.0, 0.5, -0.2],
            router_probabilities=[0.6, 0.3, 0.1],
            selected_expert_ids=[3, 7, 11],
            effective_gate_weights=[0.6, 0.3, 0.1],
            topk_ranks=[0, 1, 2],
            top1_top2_gap=0.3,
            routing_entropy=0.95,
            mode="mock",
        )


def build_batch_routing_summary(trace: RouterTrace | None = None) -> BatchRoutingSummary:
    if trace is None:
        return BatchRoutingSummary()
    token_routes = trace.all_token_selected_expert_ids or [trace.selected_expert_ids]
    bucket_positions: dict[int, list[int]] = {}
    for token_pos, expert_ids in enumerate(token_routes):
        for expert_id in expert_ids:
            bucket_positions.setdefault(int(expert_id), []).append(token_pos)
    active_destinations = sorted(bucket_positions)
    buckets = [
        BucketSummary(
            destination_id=int(expert_id),
            expert_ids_in_bucket=[int(expert_id)],
            token_count_in_bucket=len(token_positions),
            token_positions=token_positions,
            selected_route_count=len(token_positions),
            bucket_rank_by_size=1,
            fanin_proxy=1,
            fanout_proxy=len(token_routes[0]) if token_routes else 0,
            is_hot_bucket=len(token_positions) > 1,
        )
        for expert_id, token_positions in sorted(bucket_positions.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    for rank, bucket in enumerate(buckets, start=1):
        bucket.bucket_rank_by_size = rank
    return BatchRoutingSummary(
        sample_id=trace.sample_id,
        microbatch_id=trace.microbatch_id,
        layer_id=trace.layer_id,
        layer_path=trace.layer_path,
        sequence_length=trace.sequence_length,
        active_destinations=active_destinations,
        destination_count=len(active_destinations),
        total_selected_routes=sum(len(expert_ids) for expert_ids in token_routes),
        source_count=len(token_routes),
        notes=f"summary generated from {trace.mode} router trace",
        buckets=buckets,
        source_destination_matrix=[[int(expert_id) for expert_id in expert_ids] for expert_ids in token_routes],
        destination_participation_list=active_destinations,
        co_active_destinations=[active_destinations],
        barrier_relevant_destinations=active_destinations,
    )


def collect_router_trace(model: Any, tokenizer: Any, text: str, layer_path: str, use_mock: bool = False) -> RouterTrace:
    """Collect a router trace for the selected layer.

    Real mode uses the native Hugging Face OLMoE `output_router_logits=True`
    forward output. For OLMoE, each router-logit tensor is shaped
    `(batch_size * sequence_length, num_experts)`.
    """

    if use_mock:
        return MockRouterTraceProvider()(text, layer_path)
    if model is None or tokenizer is None:
        raise RouterTraceIntegrationError("real router trace requires a loaded model and tokenizer")

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import availability
        raise RouterTraceIntegrationError("PyTorch is required for real router trace") from exc

    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    sequence_length = int(input_ids.shape[1])
    if sequence_length < 1:
        raise RouterTraceIntegrationError("tokenizer returned an empty sequence")

    input_device = _model_input_device(model)
    if input_device is not None:
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True, return_dict=True, use_cache=False)

    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise RouterTraceIntegrationError(
            "model forward did not return router_logits; confirm this OLMoE build supports output_router_logits=True"
        )

    layer_id = _resolve_layer_id(layer_path, len(router_logits_by_layer))
    layer_router_logits = router_logits_by_layer[layer_id].detach().float().cpu()
    if layer_router_logits.ndim != 2:
        raise RouterTraceIntegrationError(f"unexpected router logits shape: {tuple(layer_router_logits.shape)}")
    if layer_router_logits.shape[0] < sequence_length:
        raise RouterTraceIntegrationError(
            f"router logits have too few rows for sequence length: {tuple(layer_router_logits.shape)} vs {sequence_length}"
        )

    token_pos = sequence_length - 2 if sequence_length > 1 else 0
    token_logits = layer_router_logits[token_pos]
    probabilities = torch.softmax(token_logits, dim=-1)
    top_k = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or min(2, token_logits.numel()))
    top_k = max(1, min(top_k, int(token_logits.numel())))
    top_weights, selected_experts = torch.topk(probabilities, top_k, dim=-1)
    if bool(getattr(getattr(model, "config", None), "norm_topk_prob", False)):
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

    all_token_selected_expert_ids = [
        [int(expert_id) for expert_id in torch.topk(torch.softmax(layer_router_logits[pos], dim=-1), top_k).indices.tolist()]
        for pos in range(sequence_length)
    ]
    selected_expert_ids = [int(expert_id) for expert_id in selected_experts.tolist()]
    router_probabilities = [float(value) for value in probabilities.tolist()]
    top_weights_list = [float(value) for value in top_weights.tolist()]
    top1_top2_gap = None
    if len(top_weights_list) >= 2:
        top1_top2_gap = float(top_weights_list[0] - top_weights_list[1])
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())

    target_token_id = int(input_ids[0, token_pos].detach().cpu().item())
    next_token_id = None
    next_token_text = None
    if token_pos + 1 < sequence_length:
        next_token_id = int(input_ids[0, token_pos + 1].detach().cpu().item())
        next_token_text = tokenizer.decode([next_token_id])

    return RouterTrace(
        layer_path=_resolve_layer_path(model, layer_path, layer_id),
        layer_id=layer_id,
        token_pos=token_pos,
        target_token_id=target_token_id,
        target_token_text=tokenizer.decode([target_token_id]),
        next_token_id=next_token_id,
        next_token_text=next_token_text,
        router_logits=[float(value) for value in token_logits.tolist()],
        router_probabilities=router_probabilities,
        selected_expert_ids=selected_expert_ids,
        effective_gate_weights=top_weights_list,
        topk_ranks=list(range(len(selected_expert_ids))),
        top1_top2_gap=top1_top2_gap,
        routing_entropy=entropy,
        sequence_length=sequence_length,
        all_expert_count=int(token_logits.numel()),
        selected_destination_ids=selected_expert_ids,
        layer_name=_resolve_layer_path(model, layer_path, layer_id),
        mode="real",
        all_token_selected_expert_ids=all_token_selected_expert_ids,
    )


def _model_input_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _resolve_layer_id(layer_path: str, layer_count: int) -> int:
    if layer_count <= 0:
        raise RouterTraceIntegrationError("model returned no router-logit layers")
    if layer_path in ("", "auto", None):
        return layer_count // 2
    if isinstance(layer_path, str) and layer_path.isdigit():
        layer_id = int(layer_path)
    else:
        matches = [int(match) for match in re.findall(r"\d+", str(layer_path))]
        if not matches:
            raise RouterTraceIntegrationError(f"cannot infer layer id from layer path: {layer_path}")
        layer_id = matches[-1]
    if layer_id < 0 or layer_id >= layer_count:
        raise RouterTraceIntegrationError(f"layer id {layer_id} is outside router-logit range 0..{layer_count - 1}")
    return layer_id


def _resolve_layer_path(model: Any, requested_layer_path: str, layer_id: int) -> str:
    if requested_layer_path not in ("", "auto", None) and not str(requested_layer_path).isdigit():
        return str(requested_layer_path)
    for candidate in (
        f"model.layers.{layer_id}.mlp",
        f"model.layers.{layer_id}.block_sparse_moe",
        f"model.layers.{layer_id}",
    ):
        try:
            model.get_submodule(candidate)
            return candidate
        except AttributeError:
            break
        except Exception:
            continue
    return f"router_logits[{layer_id}]"
