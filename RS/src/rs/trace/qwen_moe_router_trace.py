from __future__ import annotations

from pathlib import Path
from typing import Any

from .olmoe_router_trace import (
    collect_full_sequence_trace,
    collect_moe_architecture_probe,
    collect_olmoe_router_trace,
)


def _validate_qwen_moe_model(model: Any) -> None:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = [str(item).lower() for item in getattr(config, "architectures", []) or []]
    class_name = type(model).__name__.lower()
    if model_type == "qwen2_moe":
        return
    if any("qwen" in item and "moe" in item for item in architectures):
        return
    if "qwen" in class_name and "moe" in class_name:
        return
    raise RuntimeError(
        "expected a Qwen MoE model (model_type=qwen2_moe or Qwen-MoE architecture), "
        f"got model_type={model_type!r}, class={type(model).__name__!r}"
    )


def collect_qwen_moe_architecture_probe(model: Any) -> dict[str, Any]:
    _validate_qwen_moe_model(model)
    probe = collect_moe_architecture_probe(model)
    probe["trace_backend"] = "qwen_moe"
    return probe


def collect_qwen_moe_router_trace(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    layer_path: str = "auto",
    request_id: str = "request-0",
    sample_id: str = "sample-0",
) -> dict[str, Any]:
    _validate_qwen_moe_model(model)
    payload = collect_olmoe_router_trace(
        model,
        tokenizer,
        text,
        layer_path=layer_path,
        request_id=request_id,
        sample_id=sample_id,
    )
    payload["summary"]["trace_backend"] = "qwen_moe"
    return payload


def collect_qwen_moe_full_sequence_trace(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    request_id: str = "req-0",
    sample_id: str = "sample-0",
    save_auxiliary_dir: str | Path | None = None,
) -> dict[str, Any]:
    _validate_qwen_moe_model(model)
    payload = collect_full_sequence_trace(
        model,
        tokenizer,
        text,
        request_id=request_id,
        sample_id=sample_id,
        save_auxiliary_dir=save_auxiliary_dir,
    )
    payload["summary"]["trace_backend"] = "qwen_moe"
    return payload
