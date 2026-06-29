from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorrectnessResult:
    generated_token_match: bool
    max_abs_logit_error: float | None = None
    mean_abs_logit_error: float | None = None
