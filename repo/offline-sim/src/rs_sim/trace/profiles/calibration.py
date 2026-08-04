"""Receiver posting/drain calibration sample provenance.

Trace owns the samples and provenance only.  It does not fit or freeze the
ReceiverModel cost function; that remains an integration/oracle-contract task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..schema.canonical import read_json, stable_digest, write_json
from ..schema.model import TraceValidationError

VALID_COMPONENTS = frozenset({"receiver_posting_service", "receiver_drain"})
VALID_NODE_RELATIONS = frozenset({"same_rank", "same_node", "cross_node", "not_applicable"})
CALIBRATION_SCHEMA_VERSION = "RS_SIM_RECEIVER_CALIBRATION"


@dataclass(frozen=True)
class ReceiverCalibrationProvenance:
    dataset_id: str
    source_digest: str
    environment_digest: str
    capture_id: str
    collector_version: str
    measurement_method: str
    units: str = "ns"
    queue_wait_included: bool = False
    network_transfer_included: bool = False
    scheduler_wait_included: bool = False
    performance_eligible: bool = False
    hardware_profile_calibrated: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
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
        if self.units != "ns":
            raise TraceValidationError("receiver calibration units must be ns")
        if self.queue_wait_included or self.network_transfer_included or self.scheduler_wait_included:
            raise TraceValidationError(
                "receiver calibration service samples must exclude queue, network transfer, and scheduler wait"
            )

    def digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class ReceiverCalibrationSample:
    sample_id: str
    component: str
    payload_bytes: int
    observed_service_ns: int
    repetition_index: int
    src_rank: int | None
    dst_rank: int
    node_relation: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise TraceValidationError("calibration sample_id is required")
        if self.component not in VALID_COMPONENTS:
            raise TraceValidationError(f"unsupported calibration component={self.component!r}")
        if int(self.payload_bytes) < 0 or int(self.observed_service_ns) < 0:
            raise TraceValidationError("payload_bytes and observed_service_ns must be nonnegative")
        if int(self.repetition_index) < 0 or int(self.dst_rank) < 0:
            raise TraceValidationError("repetition_index and dst_rank must be nonnegative")
        if self.src_rank is not None and int(self.src_rank) < 0:
            raise TraceValidationError("src_rank must be nonnegative when present")
        if self.node_relation not in VALID_NODE_RELATIONS:
            raise TraceValidationError(f"unsupported node_relation={self.node_relation!r}")


@dataclass(frozen=True)
class ReceiverCalibrationDataset:
    provenance: ReceiverCalibrationProvenance
    samples: tuple[ReceiverCalibrationSample, ...]
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.samples:
            raise TraceValidationError("receiver calibration dataset must contain samples")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise TraceValidationError("receiver calibration sample IDs must be unique")
        components = {sample.component for sample in self.samples}
        missing = sorted(VALID_COMPONENTS - components)
        if missing:
            raise TraceValidationError(f"receiver calibration dataset missing components: {missing}")

    def digest(self) -> str:
        return stable_digest(asdict(self))


def write_receiver_calibration_dataset(path: Path, dataset: ReceiverCalibrationDataset) -> None:
    payload = asdict(dataset)
    payload["dataset_digest"] = dataset.digest()
    write_json(path, payload)


def load_receiver_calibration_dataset(path: Path) -> ReceiverCalibrationDataset:
    data = read_json(path)
    declared = data.pop("dataset_digest", None)
    provenance = ReceiverCalibrationProvenance(**data["provenance"])
    samples = tuple(
        ReceiverCalibrationSample(
            sample_id=str(row["sample_id"]),
            component=str(row["component"]),
            payload_bytes=int(row["payload_bytes"]),
            observed_service_ns=int(row["observed_service_ns"]),
            repetition_index=int(row["repetition_index"]),
            src_rank=None if row.get("src_rank") is None else int(row["src_rank"]),
            dst_rank=int(row["dst_rank"]),
            node_relation=str(row["node_relation"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in data["samples"]
    )
    dataset = ReceiverCalibrationDataset(
        provenance=provenance,
        samples=samples,
        schema_version=str(data.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
    )
    if declared is not None and str(declared) != dataset.digest():
        raise TraceValidationError(
            f"receiver calibration digest mismatch: declared={declared}, actual={dataset.digest()}"
        )
    return dataset


def calibration_summary(dataset: ReceiverCalibrationDataset) -> dict[str, Any]:
    by_component: dict[str, list[ReceiverCalibrationSample]] = {component: [] for component in VALID_COMPONENTS}
    for sample in dataset.samples:
        by_component[sample.component].append(sample)
    return {
        "status": "PASS",
        "schema_version": dataset.schema_version,
        "dataset_digest": dataset.digest(),
        "provenance_digest": dataset.provenance.digest(),
        "sample_count": len(dataset.samples),
        "components": {
            component: {
                "sample_count": len(samples),
                "payload_bytes_min": min(sample.payload_bytes for sample in samples),
                "payload_bytes_max": max(sample.payload_bytes for sample in samples),
                "service_ns_min": min(sample.observed_service_ns for sample in samples),
                "service_ns_max": max(sample.observed_service_ns for sample in samples),
            }
            for component, samples in sorted(by_component.items())
        },
        "model_fit_performed": False,
        "performance_eligible": dataset.provenance.performance_eligible,
        "hardware_profile_calibrated": dataset.provenance.hardware_profile_calibrated,
    }
