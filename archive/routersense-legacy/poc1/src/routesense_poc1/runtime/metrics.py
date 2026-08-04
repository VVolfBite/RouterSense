from __future__ import annotations

from typing import Any

import numpy as np


def next_token_nll(logits: Any, input_ids: Any) -> float:
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        torch = None

    if torch is not None and isinstance(logits, torch.Tensor) and isinstance(input_ids, torch.Tensor):
        if logits.ndim != 3:
            raise ValueError("logits must be rank-3")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2")
        if logits.shape[0] != input_ids.shape[0]:
            raise ValueError("batch dimensions do not match")
        if logits.shape[1] < 2:
            raise ValueError("logits sequence length must be at least 2")
        if input_ids.shape[1] != logits.shape[1]:
            raise ValueError("input_ids sequence length must match logits sequence length")
        loss = torch.nn.functional.cross_entropy(logits[:, -2, :], input_ids[:, -1])
        return float(loss.detach().cpu().item())

    logits_array = np.asarray(logits)
    input_ids_array = np.asarray(input_ids)
    if logits_array.ndim != 3:
        raise ValueError("logits must be rank-3")
    if input_ids_array.ndim != 2:
        raise ValueError("input_ids must be rank-2")
    if logits_array.shape[0] != input_ids_array.shape[0]:
        raise ValueError("batch dimensions do not match")
    if logits_array.shape[1] < 2:
        raise ValueError("logits sequence length must be at least 2")
    if input_ids_array.shape[1] != logits_array.shape[1]:
        raise ValueError("input_ids sequence length must match logits sequence length")

    target_logits = logits_array[:, -2, :]
    targets = input_ids_array[:, -1]
    log_probs = target_logits - np.max(target_logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    losses = -log_probs[np.arange(targets.shape[0]), targets]
    return float(np.mean(losses))
