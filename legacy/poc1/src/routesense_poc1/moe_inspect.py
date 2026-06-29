from __future__ import annotations

import inspect
from typing import Any

from .schemas import MoELayerSpec


def discover_moe_layers(model: Any, num_moe_layers: int) -> list[MoELayerSpec]:
    discovered: list[MoELayerSpec] = []
    top_k = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or 2)
    norm_topk_prob = bool(getattr(getattr(model, "config", None), "norm_topk_prob", False))
    for name, module in model.named_modules():
        if not (hasattr(module, "gate") and hasattr(module, "experts")):
            continue
        gate = getattr(module, "gate")
        experts = getattr(module, "experts")
        discovered.append(
            MoELayerSpec(
                layer_index=len(discovered),
                module_path=name,
                gate_module_path=f"{name}.gate",
                experts_module_path=f"{name}.experts",
                num_experts=_infer_num_experts(experts, model),
                top_k=top_k,
                norm_topk_prob=norm_topk_prob,
                module_class=module.__class__.__name__,
                gate_class=gate.__class__.__name__,
                experts_class=experts.__class__.__name__,
            )
        )
    if num_moe_layers >= len(discovered):
        return discovered
    if num_moe_layers <= 1:
        return [discovered[len(discovered) // 2]]
    indices = sorted({round(i * (len(discovered) - 1) / (num_moe_layers - 1)) for i in range(num_moe_layers)})
    return [discovered[index] for index in indices]


def _infer_num_experts(experts: Any, model: Any) -> int:
    if hasattr(experts, "num_experts"):
        return int(getattr(experts, "num_experts"))
    if hasattr(getattr(model, "config", None), "num_experts"):
        return int(getattr(model.config, "num_experts"))
    if hasattr(experts, "__len__"):
        return int(len(experts))
    raise RuntimeError("unable to infer num_experts")
