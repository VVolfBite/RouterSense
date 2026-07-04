from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    correctness_status: str
    numerical_correctness_pass: bool | None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    relative_error: float | None = None
    cosine_similarity: float | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
