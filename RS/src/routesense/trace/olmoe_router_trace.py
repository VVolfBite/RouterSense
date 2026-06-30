from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RouterTraceRecord:
    request_id: str
    sample_id: str
    token_position: int
    layer_id: int
    expert_id: int
    expert_rank: int
    routing_weight: float
    expert_rank_within_topk: int
    topk: int
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FullSequenceTraceRecord:
    request_id: str
    sample_id: str
    token_position: int
    layer_id: int
    expert_id: int
    topk_rank: int
    routing_weight: float
    topk: int
    router_capture_timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_layer_id(layer_path: str, layer_count: int) -> int:
    if layer_count <= 0:
        raise RuntimeError("model returned no router layers")
    if layer_path in ("", "auto", None):
        return layer_count // 2
    if isinstance(layer_path, str) and layer_path.isdigit():
        value = int(layer_path)
    else:
        digits = [int(token) for token in "".join(ch if ch.isdigit() else " " for ch in str(layer_path)).split()]
        if not digits:
            raise RuntimeError(f"cannot infer layer id from layer_path={layer_path!r}")
        value = digits[-1]
    if value < 0 or value >= layer_count:
        raise RuntimeError(f"layer id {value} outside router layer range")
    return value


def collect_olmoe_router_trace(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    layer_path: str = "auto",
    request_id: str = "request-0",
    sample_id: str = "sample-0",
) -> dict[str, Any]:
    import torch  # type: ignore

    if model is None or tokenizer is None:
        raise RuntimeError("real router trace requires a model and tokenizer")

    model.eval()
    encoded = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_router_logits=True, return_dict=True, use_cache=False)
    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise RuntimeError("model forward did not return router_logits")
    selected_layer_id = _resolve_layer_id(layer_path, len(router_logits_by_layer))
    selected_logits = router_logits_by_layer[selected_layer_id].detach().float().cpu()
    if selected_logits.ndim != 2:
        raise RuntimeError(f"unexpected router logits shape: {tuple(selected_logits.shape)}")
    token_count = int(selected_logits.shape[0])
    topk = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or min(2, selected_logits.shape[-1]))
    topk = max(1, min(topk, int(selected_logits.shape[-1])))
    records: list[RouterTraceRecord] = []
    per_expert_token_count: dict[str, int] = {}
    weights_seen: list[float] = []
    for token_position in range(token_count):
        probs = torch.softmax(selected_logits[token_position], dim=-1)
        weights, experts = torch.topk(probs, k=topk, dim=-1)
        for expert_rank_within_topk, (weight, expert_id) in enumerate(zip(weights.tolist(), experts.tolist(), strict=False)):
            record = RouterTraceRecord(
                request_id=request_id,
                sample_id=sample_id,
                token_position=token_position,
                layer_id=selected_layer_id,
                expert_id=int(expert_id),
                expert_rank=int(expert_id),
                routing_weight=float(weight),
                expert_rank_within_topk=expert_rank_within_topk,
                topk=topk,
                timestamp_ns=time.time_ns(),
            )
            records.append(record)
            per_expert_token_count[str(int(expert_id))] = per_expert_token_count.get(str(int(expert_id)), 0) + 1
            weights_seen.append(float(weight))
    summary = {
        "trace_schema_version": 1,
        "request_id": request_id,
        "sample_id": sample_id,
        "moe_layer_count": len(router_logits_by_layer),
        "topk": topk,
        "token_count": token_count,
        "route_record_count": len(records),
        "per_expert_token_count": per_expert_token_count,
        "routing_weight_summary": _summarize_weights(weights_seen),
        "layer_id": selected_layer_id,
        "layer_path": layer_path,
    }
    return {"summary": summary, "records": [record.to_dict() for record in records]}


def discover_moe_layer_ids(model: Any) -> list[int]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []
    moe_layer_ids: list[int] = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            moe_layer_ids.append(layer_index)
    return moe_layer_ids


def collect_full_sequence_trace(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    request_id: str = "req-0",
    sample_id: str = "sample-0",
) -> dict[str, Any]:
    import torch  # type: ignore

    if model is None or tokenizer is None:
        raise RuntimeError("full sequence trace requires a model and tokenizer")

    model.eval()
    encoded = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_router_logits=True, return_dict=True, use_cache=False)

    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise RuntimeError("model forward did not return router_logits")
    moe_layer_ids = discover_moe_layer_ids(model)
    if moe_layer_ids and len(moe_layer_ids) != len(router_logits_by_layer):
        raise RuntimeError(
            f"router_logits count {len(router_logits_by_layer)} does not match discovered MoE layers {len(moe_layer_ids)}"
        )
    if not moe_layer_ids:
        moe_layer_ids = list(range(len(router_logits_by_layer)))

    topk = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or 1)
    records: list[FullSequenceTraceRecord] = []
    per_expert_token_count: dict[str, int] = {}
    routing_weights: list[float] = []
    for logical_layer_index, logits in enumerate(router_logits_by_layer):
        layer_id = int(moe_layer_ids[logical_layer_index])
        layer_logits = logits.detach().float().cpu()
        if layer_logits.ndim != 2:
            raise RuntimeError(f"unexpected router logits shape for layer {layer_id}: {tuple(layer_logits.shape)}")
        probs = torch.softmax(layer_logits, dim=-1)
        weights, experts = torch.topk(probs, k=min(topk, probs.shape[-1]), dim=-1)
        for token_position in range(layer_logits.shape[0]):
            for topk_rank, (weight, expert_id) in enumerate(
                zip(weights[token_position].tolist(), experts[token_position].tolist(), strict=False)
            ):
                record = FullSequenceTraceRecord(
                    request_id=request_id,
                    sample_id=sample_id,
                    token_position=int(token_position),
                    layer_id=layer_id,
                    expert_id=int(expert_id),
                    topk_rank=int(topk_rank),
                    routing_weight=float(weight),
                    topk=int(topk),
                    router_capture_timestamp_ns=time.time_ns(),
                )
                records.append(record)
                per_expert_token_count[str(int(expert_id))] = per_expert_token_count.get(str(int(expert_id)), 0) + 1
                routing_weights.append(float(weight))
    token_count = int(router_logits_by_layer[0].shape[0])
    summary = {
        "trace_schema_version": 2,
        "trace_kind": "full_sequence_router_logit_observational",
        "request_id": request_id,
        "sample_id": sample_id,
        "moe_layer_count": len(router_logits_by_layer),
        "moe_layer_ids": moe_layer_ids,
        "topk": topk,
        "token_count": token_count,
        "record_count": len(records),
        "per_expert_token_count": per_expert_token_count,
        "routing_weight_summary": _summarize_weights(routing_weights),
    }
    return {"summary": summary, "records": [record.to_dict() for record in records]}


def decode_metadata_tensor_rows(metadata_tensor: Any) -> list[dict[str, int]]:
    import torch  # type: ignore

    if metadata_tensor is None:
        return []
    if not isinstance(metadata_tensor, torch.Tensor):
        raise TypeError("metadata_tensor must be a torch.Tensor")
    if metadata_tensor.ndim != 2 or metadata_tensor.shape[-1] != 6:
        raise ValueError(f"metadata tensor must be [N, 6], got {tuple(metadata_tensor.shape)}")
    rows = []
    for row in metadata_tensor.detach().cpu().tolist():
        rows.append(
            {
                "token_id": int(row[0]),
                "global_route_item_index": int(row[1]),
                "origin_rank": int(row[2]),
                "destination_rank": int(row[3]),
                "expert_id": int(row[4]),
                "route_rank": int(row[5]),
            }
        )
    return rows


def summarize_router_trace(trace: dict[str, Any]) -> dict[str, Any]:
    records = trace.get("records", [])
    per_expert_token_count: dict[str, int] = {}
    weights: list[float] = []
    for record in records:
        expert_id = str(record["expert_id"])
        per_expert_token_count[expert_id] = per_expert_token_count.get(expert_id, 0) + 1
        weights.append(float(record["routing_weight"]))
    return {
        **trace["summary"],
        "per_expert_token_count": per_expert_token_count,
        "routing_weight_summary": _summarize_weights(weights),
    }


def _summarize_weights(weights: list[float]) -> dict[str, float]:
    if not weights:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = sum(weights) / len(weights)
    variance = sum((value - mean) ** 2 for value in weights) / len(weights)
    return {"min": min(weights), "max": max(weights), "mean": mean, "std": math.sqrt(variance)}
