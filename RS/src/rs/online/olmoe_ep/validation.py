from __future__ import annotations

from ...contracts import ValidationResult


def validation_not_checked() -> dict[str, object]:
    return ValidationResult(
        correctness_status="not_checked",
        numerical_correctness_pass=None,
    ).to_dict()
