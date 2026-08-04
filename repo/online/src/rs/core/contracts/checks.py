from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping


class CheckStage(str, Enum):
    CONTRACT = "contract"
    PREDICTION = "prediction"
    PLANNING = "planning"
    PUBLICATION = "publication"
    MATERIALIZATION = "materialization"
    VALIDATION = "validation"
    EXECUTION = "execution"
    EVIDENCE = "evidence"


class CheckSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True)
class CheckCode:
    value: str

    def __post_init__(self) -> None:
        if not str(self.value).strip():
            raise ValueError("check code must be non-empty")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class CheckResult:
    code: CheckCode
    stage: CheckStage
    severity: CheckSeverity
    message: str
    passed: bool
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["code"] = str(self.code)
        payload["stage"] = str(self.stage.value)
        payload["severity"] = str(self.severity.value)
        payload["details"] = dict(self.details)
        return payload
