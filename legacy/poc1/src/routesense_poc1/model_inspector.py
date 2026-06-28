from __future__ import annotations

import inspect
from typing import Any

from .schemas import MoELayerInfo


def inspect_model(model: Any) -> list[MoELayerInfo]:
    """Inspect a model and collect candidate MoE layers.

    The implementation is generic: it walks named modules, evaluates several signals,
    and records basic metadata for later selection.
    """

    layers: list[MoELayerInfo] = []
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        module_name_lower = module_name.lower()
        class_name_lower = module.__class__.__name__.lower()
        has_gate = hasattr(module, "gate")
        has_experts = hasattr(module, "experts")
        has_router = any(token in module_name_lower for token in ("router", "gate", "moe", "expert", "sparse"))
        has_class_hint = any(token in class_name_lower for token in ("moe", "sparse", "expert", "router", "gate"))
        child_names = [name for name, _ in getattr(module, "named_children", lambda: [])()]
        score = 0.0
        reason_parts: list[str] = []
        if has_gate:
            score += 1.0
            reason_parts.append("gate")
        if has_experts:
            score += 2.0
            reason_parts.append("experts")
        if has_router or has_class_hint:
            score += 1.5
            reason_parts.append("router_hint")
        if any(token in module_name_lower for token in ("moe", "expert", "sparse", "router")):
            score += 1.0
            reason_parts.append("path_hint")
        if not any(token in module_name_lower for token in ("linear", "embedding", "norm")) and not class_name_lower.endswith("linear"):
            score += 0.5
            reason_parts.append("non_linear")
        is_candidate = (score >= 3.0 and has_gate and has_experts) or (score >= 4.0 and (has_gate or has_experts))
        if not is_candidate:
            continue
        forward = getattr(module, "forward", None)
        forward_signature = "(args, kwargs)"
        if forward is not None:
            try:
                forward_signature = inspect.signature(forward).__str__()
            except (TypeError, ValueError):
                forward_signature = "(args, kwargs)"
        layers.append(
            MoELayerInfo(
                layer_path=module_name,
                module_class=module.__class__.__name__,
                forward_signature=forward_signature,
                has_gate=has_gate,
                has_experts=has_experts,
                module_name=module_name,
                candidate_reason=", ".join(reason_parts),
                candidate_score=score,
                child_module_names=child_names,
            )
        )
    return layers


def select_middle_moe_layer(layers: list[MoELayerInfo]) -> MoELayerInfo | None:
    if not layers:
        return None
    if len(layers) == 1:
        return layers[0]
    middle_index = len(layers) // 2
    return layers[middle_index]
