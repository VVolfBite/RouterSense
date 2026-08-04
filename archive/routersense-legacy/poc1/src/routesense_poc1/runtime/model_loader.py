from __future__ import annotations

# Model loading entry points for GPU deployment and inspection flow.

from typing import Any

from ..core.schemas import RunConfig


class ModelLoadError(RuntimeError):
    pass


def load_olmoe_model(config: RunConfig):
    """Load the OLMoE model when running on a GPU-enabled host."""

    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import availability
        raise ModelLoadError("PyTorch is required to load OLMoE models") from exc

    if not torch.cuda.is_available():
        raise ModelLoadError(
            "当前机器仅完成代码开发，真实 OLMoE 推理需在 GPU 服务器执行。"
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - import availability
        raise ModelLoadError("transformers is required to load OLMoE models") from exc

    dtype_by_precision: dict[str, Any] = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype_by_precision.get(config.precision, torch.float16)}
    if config.device_map:
        model_kwargs["device_map"] = config.device_map
    if config.max_memory:
        model_kwargs["max_memory"] = _filter_available_cuda_max_memory(config.max_memory, torch.cuda.device_count())

    source = config.model_path or config.model_id
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(source)
    except Exception as exc:  # pragma: no cover - depends on remote/local model availability
        raise ModelLoadError(f"failed to load OLMoE model '{source}': {exc}") from exc
    if hasattr(model, "config"):
        model.config.output_router_logits = True
    return model, tokenizer


def _filter_available_cuda_max_memory(max_memory: dict[str, str], cuda_device_count: int) -> dict[Any, str]:
    filtered: dict[Any, str] = {}
    for device, memory in max_memory.items():
        if isinstance(device, int):
            if device < cuda_device_count:
                filtered[device] = memory
            continue
        if isinstance(device, str) and device.isdigit():
            device_id = int(device)
            if device_id < cuda_device_count:
                filtered[device_id] = memory
            continue
        filtered[device] = memory
    return filtered
