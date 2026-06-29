from __future__ import annotations

import json
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SingleGPUInferenceResult:
    prompt: str
    generated_text: str
    input_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    model_revision: str
    device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gpu_environment_snapshot() -> dict[str, Any]:
    import torch  # type: ignore

    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        snapshot["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
        snapshot["gpu_total_memory_mb"] = [
            round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 2) for i in range(count)
        ]
    return snapshot


def load_model_and_tokenizer(
    model_id: str,
    model_path: str | None = None,
    precision: str = "bf16",
    device_index: int = 0,
):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this host")
    dtype = _precision_to_dtype(precision, torch)
    source = model_path or model_id
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.output_router_logits = True
    return model, tokenizer, device, dtype, source


def run_single_gpu_text_inference(
    *,
    model_id: str,
    model_path: str | None = None,
    prompt: str,
    max_new_tokens: int = 32,
    precision: str = "bf16",
    device_index: int = 0,
    revision: str | None = None,
    output_dir: str | Path | None = None,
) -> SingleGPUInferenceResult:
    import torch  # type: ignore

    model, tokenizer, device, dtype, source = load_model_and_tokenizer(
        model_id=model_id,
        model_path=model_path,
        precision=precision,
        device_index=device_index,
    )
    revision_value = revision or getattr(model.config, "_commit_hash", None) or getattr(model.config, "_name_or_path", None) or source
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    with torch.inference_mode():
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
        prefill_output = model(**encoded, use_cache=True, return_dict=True)
        prefill_end.record()
        torch.cuda.synchronize(device)
        prefill_ms = float(prefill_start.elapsed_time(prefill_end))

        past_key_values = getattr(prefill_output, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("model did not return past_key_values for decode path")
        next_token = torch.argmax(prefill_output.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = encoded["input_ids"][0].tolist()

        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
        for _ in range(max_new_tokens):
            step_output = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True, return_dict=True)
            past_key_values = getattr(step_output, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model decode step did not return past_key_values")
            next_token = torch.argmax(step_output.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))
        decode_end.record()
        torch.cuda.synchronize(device)
        decode_ms = float(decode_start.elapsed_time(decode_end))

    total_ms = (time.perf_counter() - total_start) * 1000.0
    result = SingleGPUInferenceResult(
        prompt=prompt,
        generated_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        input_tokens=int(encoded["input_ids"].shape[1]),
        generated_tokens=int(max_new_tokens),
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        peak_allocated_mb=round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        peak_reserved_mb=round(torch.cuda.max_memory_reserved(device) / 1024**2, 2),
        model_revision=str(revision_value),
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        (out / "goal.md").write_text(
            "# Single-GPU OLMoE Smoke\n\nThis run loads a real OLMoE checkpoint on one GPU, performs one text generation, and records memory/timing.\n",
            encoding="utf-8",
        )
    return result


def _precision_to_dtype(precision: str, torch_module: Any) -> Any:
    mapping = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    return mapping.get(precision, torch_module.bfloat16)

