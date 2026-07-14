from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping


class RunKind(str, Enum):
    OFFLINE_EVALUATION = "OFFLINE_EVALUATION"
    GLOO_FUNCTIONAL = "GLOO_FUNCTIONAL"
    GPU_CORRECTNESS = "GPU_CORRECTNESS"
    GPU_PERFORMANCE = "GPU_PERFORMANCE"
    MULTINODE_CORRECTNESS = "MULTINODE_CORRECTNESS"
    MULTINODE_PERFORMANCE = "MULTINODE_PERFORMANCE"
    TRACE_COLLECTION = "TRACE_COLLECTION"
    DIAGNOSTIC = "DIAGNOSTIC"


@dataclass(frozen=True)
class PlanningCase:
    case_id: str
    run_kind: RunKind
    planner_id: str
    planner_family: str
    selector_mode: str
    predictor_id: str
    prediction_mode: str
    execution_backend: str
    instrumentation_mode: str
    fallback_policy: str

    def validate(self) -> None:
        required = (
            self.case_id,
            self.planner_id,
            self.planner_family,
            self.selector_mode,
            self.predictor_id,
            self.prediction_mode,
            self.execution_backend,
            self.instrumentation_mode,
            self.fallback_policy,
        )
        if any(not str(item).strip() for item in required):
            raise ValueError("planning case fields must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["run_kind"] = self.run_kind.value
        return payload


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    markers: tuple[str, ...]
    case_ids: tuple[str, ...]
    run_kinds: tuple[RunKind, ...]
    description: str = ""

    def validate(self) -> None:
        if not str(self.suite_id).strip():
            raise ValueError("suite_id must be non-empty")
        if not self.markers:
            raise ValueError("suite markers must be non-empty")
        if not self.case_ids:
            raise ValueError("suite case_ids must be non-empty")
        if not self.run_kinds:
            raise ValueError("suite run_kinds must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "suite_id": str(self.suite_id),
            "markers": list(self.markers),
            "case_ids": list(self.case_ids),
            "run_kinds": [item.value for item in self.run_kinds],
            "description": str(self.description),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    experiment_id: str
    suites: tuple[SuiteSpec, ...]
    planning_cases: tuple[PlanningCase, ...]
    defaults: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.schema_version) != 2:
            raise ValueError("experiment spec schema_version must be 2")
        if not str(self.experiment_id).strip():
            raise ValueError("experiment_id must be non-empty")
        if not self.suites:
            raise ValueError("at least one suite is required")
        if not self.planning_cases:
            raise ValueError("at least one planning case is required")
        case_ids = {case.case_id for case in self.planning_cases}
        if len(case_ids) != len(self.planning_cases):
            raise ValueError("planning case ids must be unique")
        for case in self.planning_cases:
            case.validate()
        for suite in self.suites:
            suite.validate()
            missing = set(suite.case_ids) - case_ids
            if missing:
                raise ValueError(f"suite {suite.suite_id} references unknown cases: {sorted(missing)}")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": int(self.schema_version),
            "experiment_id": str(self.experiment_id),
            "suites": [suite.to_dict() for suite in self.suites],
            "planning_cases": [case.to_dict() for case in self.planning_cases],
            "defaults": dict(self.defaults),
        }

    def config_digest(self) -> str:
        payload = self.to_dict()
        encoded = str(payload).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunPlan:
    experiment_id: str
    suite_id: str
    case_id: str
    run_kind: RunKind
    config_digest: str
    planning_case: PlanningCase
    commit_sha: str = ""
    defaults: Mapping[str, Any] = field(default_factory=dict)
    config_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": str(self.experiment_id),
            "suite_id": str(self.suite_id),
            "case_id": str(self.case_id),
            "run_kind": self.run_kind.value,
            "config_digest": str(self.config_digest),
            "commit_sha": str(self.commit_sha),
            "defaults": dict(self.defaults),
            "config_path": str(self.config_path),
            "planning_case": self.planning_case.to_dict(),
        }
