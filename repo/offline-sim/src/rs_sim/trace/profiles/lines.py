"""Trace-owned cost-profile references for the three formal service lines.

These are immutable experiment inputs only.  They do not reserve simulated
service, enqueue jobs, or create scheduler plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..schema.canonical import read_json, stable_digest, write_json
from ..schema.constants import (
    THREE_LINE_PROFILE_SCHEMA_VERSION,
    VALID_LINE_PROFILE_SOURCE_KINDS,
    VALID_SERVICE_LINES,
)
from ..schema.model import TraceValidationError


@dataclass(frozen=True)
class ServiceLineCostProfile:
    profile_id: str
    line_name: str
    source_kind: str
    source_digest: str
    environment_digest: str
    service_model: str
    fixed_service_ns: int
    per_input_item_ns: int
    minimum_service_ns: int = 0
    sample_count: int = 0
    hardware_profile_calibrated: bool = False
    performance_eligible: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.line_name not in VALID_SERVICE_LINES:
            raise TraceValidationError(f"unsupported service line={self.line_name!r}")
        if self.source_kind not in VALID_LINE_PROFILE_SOURCE_KINDS:
            raise TraceValidationError(f"unsupported line profile source_kind={self.source_kind!r}")
        for name in ("profile_id", "source_digest", "environment_digest", "service_model"):
            if not str(getattr(self, name)).strip():
                raise TraceValidationError(f"{name} must be non-empty")
        for name in ("fixed_service_ns", "per_input_item_ns", "minimum_service_ns", "sample_count"):
            if int(getattr(self, name)) < 0:
                raise TraceValidationError(f"{name} must be nonnegative")
        if self.source_kind == "synthetic" and (
            self.hardware_profile_calibrated or self.performance_eligible
        ):
            raise TraceValidationError("synthetic line profile cannot be calibrated or performance eligible")
        if self.source_kind == "measured" and self.hardware_profile_calibrated:
            raise TraceValidationError("measured and calibrated line profile provenance must remain distinct")
        if self.source_kind == "calibrated" and not self.hardware_profile_calibrated:
            raise TraceValidationError("calibrated line profile must set hardware_profile_calibrated=true")
        if self.performance_eligible and self.source_kind != "calibrated":
            raise TraceValidationError("performance eligibility requires calibrated line profile provenance")
        if self.source_kind in {"measured", "calibrated"} and self.sample_count <= 0:
            raise TraceValidationError("measured/calibrated line profiles require positive sample_count")

    def digest(self) -> str:
        return stable_digest(asdict(self), prefix="service-line-profile")


@dataclass(frozen=True)
class ThreeLineCostProfileDataset:
    dataset_id: str
    profiles: tuple[ServiceLineCostProfile, ...]
    measured_input_status: str
    schema_version: str = THREE_LINE_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise TraceValidationError("three-line dataset_id is required")
        if {profile.line_name for profile in self.profiles} != set(VALID_SERVICE_LINES):
            raise TraceValidationError("three-line dataset must contain exactly one profile per formal line")
        if len(self.profiles) != len(VALID_SERVICE_LINES):
            raise TraceValidationError("three-line dataset contains duplicate service-line profiles")
        if self.measured_input_status not in {"MISSING_MEASURED_INPUT", "MEASURED_AVAILABLE", "CALIBRATED"}:
            raise TraceValidationError("unsupported measured_input_status")
        source_kinds = {profile.source_kind for profile in self.profiles}
        if source_kinds == {"synthetic"} and self.measured_input_status != "MISSING_MEASURED_INPUT":
            raise TraceValidationError("synthetic line profiles must expose the measured-input gap")
        if source_kinds == {"measured"} and self.measured_input_status != "MEASURED_AVAILABLE":
            raise TraceValidationError("measured line profiles require MEASURED_AVAILABLE status")
        if source_kinds == {"calibrated"} and self.measured_input_status != "CALIBRATED":
            raise TraceValidationError("calibrated line profiles require CALIBRATED status")
        if len(source_kinds) != 1:
            raise TraceValidationError("one three-line dataset may not mix provenance classes")

    @property
    def hardware_profile_calibrated(self) -> bool:
        return all(profile.hardware_profile_calibrated for profile in self.profiles)

    @property
    def performance_eligible(self) -> bool:
        return all(profile.performance_eligible for profile in self.profiles)

    def digest(self) -> str:
        return stable_digest(asdict(self), prefix="three-line-profile-dataset")

    def references(self) -> dict[str, dict[str, Any]]:
        return {
            profile.line_name: {
                **asdict(profile),
                "profile_digest": profile.digest(),
            }
            for profile in sorted(self.profiles, key=lambda value: value.line_name)
        }


def write_three_line_profile_dataset(path: Path, dataset: ThreeLineCostProfileDataset) -> None:
    payload = asdict(dataset)
    payload["dataset_digest"] = dataset.digest()
    write_json(Path(path), payload)


def load_three_line_profile_dataset(path: Path) -> ThreeLineCostProfileDataset:
    payload = read_json(Path(path))
    declared = payload.pop("dataset_digest", None)
    profiles = tuple(ServiceLineCostProfile(**row) for row in payload["profiles"])
    dataset = ThreeLineCostProfileDataset(
        dataset_id=str(payload["dataset_id"]),
        profiles=profiles,
        measured_input_status=str(payload["measured_input_status"]),
        schema_version=str(payload.get("schema_version", THREE_LINE_PROFILE_SCHEMA_VERSION)),
    )
    if dataset.schema_version != THREE_LINE_PROFILE_SCHEMA_VERSION:
        raise TraceValidationError("unsupported three-line cost profile schema")
    if declared is None or str(declared) != dataset.digest():
        raise TraceValidationError(
            f"three-line profile digest mismatch: declared={declared}, actual={dataset.digest()}"
        )
    return dataset


def three_line_profile_summary(dataset: ThreeLineCostProfileDataset) -> dict[str, Any]:
    return {
        "status": "PASS",
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest(),
        "line_names": sorted(profile.line_name for profile in dataset.profiles),
        "source_kind": dataset.profiles[0].source_kind,
        "measured_input_status": dataset.measured_input_status,
        "hardware_profile_calibrated": dataset.hardware_profile_calibrated,
        "performance_eligible": dataset.performance_eligible,
    }


__all__ = [
    "ServiceLineCostProfile",
    "ThreeLineCostProfileDataset",
    "load_three_line_profile_dataset",
    "three_line_profile_summary",
    "write_three_line_profile_dataset",
]
