from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .schemas import EnvironmentInfo, RunConfig


def run_doctor(config: RunConfig) -> dict[str, Any]:
    import torch  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    if config.precision.lower() in {"bf16", "bfloat16"}:
        config.precision = "fp16"
    info = EnvironmentInfo(
        python_version=platform.python_version(),
        torch_version=getattr(torch, "__version__", "unknown"),
        transformers_version=version("transformers"),
        cuda_available=cuda_available,
        cuda_version=getattr(torch.version, "cuda", None),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
        gpu_memory_mb=[
            round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 2) for i in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    )
    return info.to_dict()
