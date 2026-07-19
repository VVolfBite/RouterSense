from __future__ import annotations

from typing import Mapping

from rs.core.contracts.checks import CheckResult, CheckSeverity, CheckStage
from rs.runtime.checks.codes import INVALID_STATUS, MISSING_FIELD


def require_fields(*, stage: CheckStage, payload: Mapping[str, object], fields: tuple[str, ...]) -> tuple[CheckResult, ...]:
    results: list[CheckResult] = []
    for field in fields:
        present = field in payload and payload[field] not in (None, "")
        results.append(
            CheckResult(
                code=MISSING_FIELD,
                stage=stage,
                severity=CheckSeverity.ERROR if not present else CheckSeverity.INFO,
                message=f"{field} {'present' if present else 'missing'}",
                passed=bool(present),
                details={"field": str(field)},
            )
        )
    return tuple(results)


def require_status(*, stage: CheckStage, status: str, allowed: tuple[str, ...]) -> CheckResult:
    passed = str(status) in {str(item) for item in allowed}
    return CheckResult(
        code=INVALID_STATUS,
        stage=stage,
        severity=CheckSeverity.ERROR if not passed else CheckSeverity.INFO,
        message=f"status={status}",
        passed=passed,
        details={"allowed": [str(item) for item in allowed]},
    )
