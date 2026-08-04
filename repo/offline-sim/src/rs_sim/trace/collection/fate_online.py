from __future__ import annotations

"""Bounded sampled FATE adapter for Megatron capture.

The implementation follows the formal online FATE path: deterministically sample
the current layer's router input, evaluate it with the next layer's frozen gate
projection, take hard top-k expert assignments, aggregate them to destination
ranks, and scale the sampled counts to the original token count.  Only rank
traffic is persisted; hidden tensors and gate weights are not written.
"""

from dataclasses import dataclass
from typing import Any


class OnlineFateError(RuntimeError):
    pass


def _nested_attr(value: object, path: str) -> object | None:
    current = value
    for name in path.split("."):
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _router_weight(router: object) -> Any | None:
    for path in ("weight", "gate.weight", "linear.weight", "router.weight"):
        value = _nested_attr(router, path)
        if value is not None and getattr(value, "ndim", None) == 2:
            return value
    return None


def _router_bias(router: object) -> Any | None:
    for path in ("bias", "gate.bias", "linear.bias", "router.bias"):
        value = _nested_attr(router, path)
        if value is not None and getattr(value, "ndim", None) == 1:
            return value
    return None


def _read_positive_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        if result > 0:
            return result
    return None


def deterministic_even_indices(total_tokens: int, max_sample_tokens: int) -> tuple[int, ...]:
    total = int(total_tokens)
    count = min(total, int(max_sample_tokens))
    if total <= 0 or count <= 0:
        raise OnlineFateError("FATE requires positive token and sample counts")
    if count == total:
        return tuple(range(total))
    return tuple(((2 * index + 1) * total) // (2 * count) for index in range(count))


def _balanced_scale(counts: list[int], *, original_tokens: int, sampled_tokens: int, top_k: int) -> tuple[int, ...]:
    target = int(original_tokens) * int(top_k)
    observed = int(sampled_tokens) * int(top_k)
    if observed <= 0 or sum(counts) != observed:
        raise OnlineFateError("sampled FATE assignment mass is inconsistent")
    scaled = [float(value) * float(target) / float(observed) for value in counts]
    base = [int(value) for value in scaled]
    remainder = target - sum(base)
    order = sorted(range(len(base)), key=lambda idx: (-(scaled[idx] - base[idx]), idx))
    for index in order[:remainder]:
        base[index] += 1
    if sum(base) != target:
        raise OnlineFateError("unable to preserve FATE top-k assignment mass")
    return tuple(base)


@dataclass(slots=True)
class SampledGateInput:
    layer_id: int
    decode_step: int
    original_token_count: int
    sampled_hidden_cpu: Any
    sample_indices: tuple[int, ...]


def capture_sampled_gate_input(hidden_states: Any, *, layer_id: int, decode_step: int, max_sample_tokens: int) -> SampledGateInput:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - available in model capture only
        raise OnlineFateError("torch is required for online Megatron FATE") from exc
    value = hidden_states[0] if isinstance(hidden_states, tuple) and hidden_states else hidden_states
    if not isinstance(value, torch.Tensor):
        raise OnlineFateError("Megatron FATE gate input must be a torch.Tensor")
    tensor = value.detach()
    if tensor.ndim == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim != 2:
        raise OnlineFateError(f"FATE gate input must be 2-D/3-D, got {tuple(tensor.shape)}")
    original = int(tensor.shape[0])
    indices = deterministic_even_indices(original, int(max_sample_tokens))
    if len(indices) != original:
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
        tensor = tensor.index_select(0, index_tensor)
    snapshot = tensor.to(device="cpu", dtype=torch.float32).contiguous().clone()
    return SampledGateInput(
        layer_id=int(layer_id), decode_step=int(decode_step), original_token_count=original,
        sampled_hidden_cpu=snapshot, sample_indices=indices,
    )


def predict_rank_row(
    sample: SampledGateInput,
    *,
    target_module: Any,
    expert_to_rank: tuple[int, ...],
    world_size: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover
        raise OnlineFateError("torch is required for online Megatron FATE") from exc
    router = getattr(target_module, "router", None)
    if router is None:
        raise OnlineFateError("target MoE layer has no router")
    weight = _router_weight(router)
    if weight is None:
        raise OnlineFateError("target router has no snapshotable 2-D weight")
    bias = _router_bias(router)
    num_experts = int(weight.shape[0])
    if len(expert_to_rank) != num_experts:
        raise OnlineFateError("expert_to_rank size differs from target router expert count")
    top_k = _read_positive_int(
        _nested_attr(router, "config.moe_router_topk"),
        _nested_attr(target_module, "config.moe_router_topk"),
        getattr(router, "top_k", None), getattr(router, "topk", None),
        getattr(target_module, "top_k", None),
    )
    if top_k is None or top_k > num_experts:
        raise OnlineFateError("unable to determine valid target router top-k")
    frozen_weight = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    frozen_bias = None if bias is None else bias.detach().to(device="cpu", dtype=torch.float32).contiguous()
    with torch.inference_mode():
        logits = functional.linear(sample.sampled_hidden_cpu, frozen_weight, frozen_bias)
        selected = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :top_k]
    counts = [0 for _ in range(int(world_size))]
    for expert in selected.reshape(-1).tolist():
        destination = int(expert_to_rank[int(expert)])
        if not 0 <= destination < int(world_size):
            raise OnlineFateError("expert owner outside world")
        counts[destination] += 1
    scaled = _balanced_scale(
        counts,
        original_tokens=sample.original_token_count,
        sampled_tokens=int(sample.sampled_hidden_cpu.shape[0]),
        top_k=int(top_k),
    )
    evidence = {
        "predictor_id": "fate_cross_layer_gate_sampled_v1",
        "estimator_kind": "SAMPLED_HARD_TOPK_RANK_COUNTS",
        "top_k": int(top_k),
        "num_experts": int(num_experts),
        "original_token_count": int(sample.original_token_count),
        "sample_token_count": int(sample.sampled_hidden_cpu.shape[0]),
        "sampling_method": "DETERMINISTIC_EVEN_MIDPOINT",
    }
    return scaled, evidence


__all__ = [
    "OnlineFateError", "SampledGateInput", "capture_sampled_gate_input",
    "deterministic_even_indices", "predict_rank_row",
]
