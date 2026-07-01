from __future__ import annotations

import json
import platform
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
    model_revision: str | None
    device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dtype(precision: str):
    import torch  # type: ignore

    normalized = precision.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision: {precision}")


def gpu_environment_snapshot() -> dict[str, Any]:
    import torch  # type: ignore

    snapshot: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            }
        )
    snapshot["devices"] = devices
    return snapshot


def load_model_and_tokenizer(
    *,
    model_id: str,
    model_path: str | None = None,
    precision: str = "bf16",
    device_index: int = 0,
    revision: str | None = None,
):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_ref = model_path or model_id
    dtype = _resolve_dtype(precision)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real single-GPU OLMoE loading")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    resolved_revision = revision or getattr(getattr(model, "config", None), "_commit_hash", None)
    return model, tokenizer, resolved_revision, str(device), str(dtype).replace("torch.", "")


def run_single_gpu_text_inference(
    *,
    model_id: str,
    model_path: str | None,
    prompt: str,
    max_new_tokens: int,
    precision: str = "bf16",
    device_index: int = 0,
    revision: str | None = None,
    output_dir: str | Path | None = None,
) -> SingleGPUInferenceResult:
    import torch  # type: ignore

    model, tokenizer, resolved_revision, device, dtype = load_model_and_tokenizer(
        model_id=model_id,
        model_path=model_path,
        precision=precision,
        device_index=device_index,
        revision=revision,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.cuda.reset_peak_memory_stats(device_index)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    total_ms = (time.perf_counter() - start) * 1000.0
    input_tokens = int(encoded["input_ids"].shape[-1])
    output_tokens = int(generated.shape[-1])
    generated_tokens = max(0, output_tokens - input_tokens)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    result = SingleGPUInferenceResult(
        prompt=prompt,
        generated_text=text,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        prefill_ms=total_ms,
        decode_ms=0.0,
        total_ms=total_ms,
        peak_allocated_mb=round(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024), 2),
        peak_reserved_mb=round(torch.cuda.max_memory_reserved(device_index) / (1024 * 1024), 2),
        model_revision=resolved_revision,
        device=device,
        dtype=dtype,
    )
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "single_gpu_text_infer.json").write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
    return result
