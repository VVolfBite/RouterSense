from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class PublishedPlan:
    planner_id: str
    logical_plan_digest: str
    published_plan_digest: str
    publication_slot_digest: str
    root_rank: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class PayloadSpec:
    payload_role: str
    row_count: int
    element_count: int
    byte_count: int
    bytes_per_row: int
    dtype: str
    shape_suffix: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActualPhaseContext:
    layer_id: str
    phase: str
    world_size: int
    rank_space: str
    layout_digest: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TransferSlice:
    task_id: str
    payload_role: str
    src_rank: int
    dst_rank: int
    row_count: int
    send_offset_rows: int
    recv_offset_rows: int
    dependency_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionBatch:
    batch_id: str
    wave_id: int
    slices: tuple[TransferSlice, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": str(self.batch_id),
            "wave_id": int(self.wave_id),
            "slices": [item.to_dict() for item in self.slices],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MaterializedPlan:
    published_plan_digest: str
    materialized_plan_digest: str
    layout_digest: str
    payload_specs: tuple[PayloadSpec, ...]
    batches: tuple[ExecutionBatch, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "published_plan_digest": str(self.published_plan_digest),
            "materialized_plan_digest": str(self.materialized_plan_digest),
            "layout_digest": str(self.layout_digest),
            "payload_specs": [item.to_dict() for item in self.payload_specs],
            "batches": [item.to_dict() for item in self.batches],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    stage: str
    reason: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    forward_generation: int
    layer_id: str
    phase: str
    rank_space: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ExecutionOutcome:
    success: bool
    executed_batch_count: int
    all_work_completed: bool
    execution_digest: str
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


class PlanPublisher(Protocol):
    def publish(self, plan: PublishedPlan) -> PublishedPlan:
        ...


class PlanMaterializer(Protocol):
    def materialize(self, plan: PublishedPlan, context: ActualPhaseContext) -> MaterializedPlan:
        ...


class PlanValidator(Protocol):
    def validate(self, plan: MaterializedPlan, context: ActualPhaseContext) -> ValidationResult:
        ...


class ExecutionGuard(Protocol):
    def validate(self, plan: MaterializedPlan, context: ExecutionContext) -> ValidationResult:
        ...


class Executor(Protocol):
    def execute(self, plan: MaterializedPlan, payload: PayloadSpec, context: ExecutionContext) -> ExecutionOutcome:
        ...
