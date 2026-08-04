"""Receiver, ControlPlane, and DataPlane profile data formats.

The checked-in profile datasets are format examples only. Synthetic datasets
are fail-closed as non-calibrated and non-performance-eligible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..schema.canonical import read_json, stable_digest, write_json
from ..schema.constants import PROFILE_COMPONENTS_BY_PLANE, VALID_PROFILE_PLANES, VALID_PROFILE_SOURCE_KINDS
from ..schema.model import TraceValidationError


@dataclass(frozen=True)
class RuntimeProfileProvenance:
    dataset_id: str
    plane: str
    source_kind: str
    source_digest: str
    environment_digest: str
    capture_id: str
    collector_version: str
    measurement_method: str
    hardware_profile_calibrated: bool = False
    performance_eligible: bool = False
    queue_wait_included: bool = False
    scheduler_wait_included: bool = False
    network_wait_included: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.plane not in VALID_PROFILE_PLANES:
            raise TraceValidationError(f"unsupported profile plane={self.plane!r}")
        if self.source_kind not in VALID_PROFILE_SOURCE_KINDS:
            raise TraceValidationError(f"unsupported profile source_kind={self.source_kind!r}")
        for name in (
            "dataset_id",
            "source_digest",
            "environment_digest",
            "capture_id",
            "collector_version",
            "measurement_method",
        ):
            if not str(getattr(self, name)).strip():
                raise TraceValidationError(f"{name} must be non-empty")
        if self.source_kind == "synthetic_format_example" and (
            self.hardware_profile_calibrated or self.performance_eligible
        ):
            raise TraceValidationError("synthetic profile cannot be calibrated or performance eligible")
        if self.performance_eligible and not self.hardware_profile_calibrated:
            raise TraceValidationError("performance eligibility requires calibrated hardware profile")
        if self.queue_wait_included or self.scheduler_wait_included:
            raise TraceValidationError("profile service samples must exclude queue and scheduler wait")
        if self.plane != "DATA_PLANE" and self.network_wait_included:
            raise TraceValidationError("non-DataPlane profiles must exclude network wait")

    def digest(self) -> str:
        return stable_digest(asdict(self), prefix="profile-provenance")


@dataclass(frozen=True)
class RuntimeProfileSample:
    sample_id: str
    component: str
    payload_bytes: int
    observed_service_ns: int
    repetition_index: int
    src_rank: int | None = None
    dst_rank: int | None = None
    node_relation: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.component:
            raise TraceValidationError("sample_id and component are required")
        if int(self.payload_bytes) < 0 or int(self.observed_service_ns) < 0 or int(self.repetition_index) < 0:
            raise TraceValidationError("profile payload/service/repetition values must be nonnegative")
        for name in ("src_rank", "dst_rank"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise TraceValidationError(f"{name} must be nonnegative when present")
        if not self.node_relation:
            raise TraceValidationError("node_relation must be non-empty")


@dataclass(frozen=True)
class RuntimeProfileDataset:
    provenance: RuntimeProfileProvenance
    samples: tuple[RuntimeProfileSample, ...]
    schema_version: str = "RS_SIM_RUNTIME_PROFILE"

    def __post_init__(self) -> None:
        if not self.samples:
            raise TraceValidationError("runtime profile dataset must contain samples")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise TraceValidationError("runtime profile sample IDs must be unique")
        allowed = PROFILE_COMPONENTS_BY_PLANE[self.provenance.plane]
        found = {sample.component for sample in self.samples}
        invalid = sorted(found - allowed)
        missing = sorted(allowed - found)
        if invalid or missing:
            raise TraceValidationError(
                f"profile component mismatch plane={self.provenance.plane}: invalid={invalid}, missing={missing}"
            )

    def digest(self) -> str:
        return stable_digest(asdict(self), prefix="runtime-profile")


def write_runtime_profile_dataset(path: Path, dataset: RuntimeProfileDataset) -> None:
    payload = asdict(dataset)
    payload["dataset_digest"] = dataset.digest()
    write_json(path, payload)


def load_runtime_profile_dataset(path: Path) -> RuntimeProfileDataset:
    data = read_json(path)
    declared = data.pop("dataset_digest", None)
    provenance = RuntimeProfileProvenance(**data["provenance"])
    samples = tuple(
        RuntimeProfileSample(
            sample_id=str(row["sample_id"]),
            component=str(row["component"]),
            payload_bytes=int(row["payload_bytes"]),
            observed_service_ns=int(row["observed_service_ns"]),
            repetition_index=int(row["repetition_index"]),
            src_rank=None if row.get("src_rank") is None else int(row["src_rank"]),
            dst_rank=None if row.get("dst_rank") is None else int(row["dst_rank"]),
            node_relation=str(row.get("node_relation", "unspecified")),
            metadata=dict(row.get("metadata", {})),
        )
        for row in data["samples"]
    )
    dataset = RuntimeProfileDataset(
        provenance=provenance,
        samples=samples,
        schema_version=str(data.get("schema_version", "RS_SIM_RUNTIME_PROFILE")),
    )
    if provenance.source_kind == "measured" and declared is None:
        raise TraceValidationError(
            "measured runtime profiles require an externally supplied dataset_digest"
        )
    if declared is not None and str(declared) != dataset.digest():
        raise TraceValidationError(
            f"runtime profile digest mismatch: declared={declared}, actual={dataset.digest()}"
        )
    return dataset


def runtime_profile_summary(dataset: RuntimeProfileDataset) -> dict[str, Any]:
    return {
        "status": "PASS",
        "plane": dataset.provenance.plane,
        "dataset_digest": dataset.digest(),
        "provenance_digest": dataset.provenance.digest(),
        "sample_count": len(dataset.samples),
        "components": sorted({sample.component for sample in dataset.samples}),
        "hardware_profile_calibrated": dataset.provenance.hardware_profile_calibrated,
        "performance_eligible": dataset.provenance.performance_eligible,
        "model_fit_performed": False,
    }


__all__ = [
    "RuntimeProfileDataset",
    "RuntimeProfileProvenance",
    "RuntimeProfileSample",
    "load_runtime_profile_dataset",
    "runtime_profile_summary",
    "write_runtime_profile_dataset",
]
