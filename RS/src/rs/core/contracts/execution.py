from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Protocol


@dataclass(frozen=True)
class PublishedPlan:
    planner_id: str
    logical_plan_digest: str
    published_plan_digest: str
    publication_slot_digest: str
    root_rank: int
    root_group_rank: int = 0
    version: int = 1
    logical_plan: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.planner_id):
            raise ValueError("planner_id must be non-empty")
        if not str(self.logical_plan_digest):
            raise ValueError("logical_plan_digest must be non-empty")
        if not str(self.published_plan_digest):
            raise ValueError("published_plan_digest must be non-empty")
        if not str(self.publication_slot_digest):
            raise ValueError("publication_slot_digest must be non-empty")
        if int(self.root_rank) < 0 or int(self.root_group_rank) < 0:
            raise ValueError("root ranks must be >= 0")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["logical_plan"] = dict(self.logical_plan)
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

    def validate(self) -> None:
        if not str(self.payload_role):
            raise ValueError("payload_role must be non-empty")
        if int(self.row_count) < 0 or int(self.element_count) < 0 or int(self.byte_count) < 0 or int(self.bytes_per_row) < 0:
            raise ValueError("payload counts must be >= 0")
        if not str(self.dtype):
            raise ValueError("dtype must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ActualPhaseContext:
    layer_id: str
    phase: str
    world_size: int
    rank_space: str
    layout_digest: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.layer_id):
            raise ValueError("layer_id must be non-empty")
        if not str(self.phase):
            raise ValueError("phase must be non-empty")
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if not str(self.rank_space):
            raise ValueError("rank_space must be non-empty")
        if not str(self.layout_digest):
            raise ValueError("layout_digest must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TransferSlice:
    task_id: str
    flow_id: str
    payload_role: str
    src_rank: int
    dst_rank: int
    row_count: int
    send_offset_rows: int
    recv_offset_rows: int
    dependency_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if not str(self.task_id):
            raise ValueError("task_id must be non-empty")
        if not str(self.flow_id):
            raise ValueError("flow_id must be non-empty")
        if not str(self.payload_role):
            raise ValueError("payload_role must be non-empty")
        if min(int(self.src_rank), int(self.dst_rank), int(self.row_count), int(self.send_offset_rows), int(self.recv_offset_rows)) < 0:
            raise ValueError("transfer slice numeric fields must be >= 0")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ExecutionBatch:
    batch_id: str
    wave_id: int
    slices: tuple[TransferSlice, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.batch_id):
            raise ValueError("batch_id must be non-empty")
        if int(self.wave_id) < 0:
            raise ValueError("wave_id must be >= 0")
        for item in self.slices:
            item.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
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
    local_rank: int = 0
    layer_id: str = ""
    phase: str = ""
    logical_plan_digest: str = ""
    expected_payload_roles: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.published_plan_digest):
            raise ValueError("published_plan_digest must be non-empty")
        if not str(self.materialized_plan_digest):
            raise ValueError("materialized_plan_digest must be non-empty")
        if not str(self.layout_digest):
            raise ValueError("layout_digest must be non-empty")
        if int(self.local_rank) < 0:
            raise ValueError("local_rank must be >= 0")
        for item in self.payload_specs:
            item.validate()
        for batch in self.batches:
            batch.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "published_plan_digest": str(self.published_plan_digest),
            "materialized_plan_digest": str(self.materialized_plan_digest),
            "layout_digest": str(self.layout_digest),
            "payload_specs": [item.to_dict() for item in self.payload_specs],
            "batches": [item.to_dict() for item in self.batches],
            "local_rank": int(self.local_rank),
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "logical_plan_digest": str(self.logical_plan_digest),
            "expected_payload_roles": [str(item) for item in self.expected_payload_roles],
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

    def validate(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must be non-empty")
        if int(self.forward_generation) < 0:
            raise ValueError("forward_generation must be >= 0")
        if not str(self.layer_id) or not str(self.phase) or not str(self.rank_space):
            raise ValueError("execution context identity must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ExecutionOutcome:
    success: bool
    executed_batch_count: int
    all_work_completed: bool
    execution_digest: str
    completed_task_ids: tuple[str, ...] = ()
    failed_task_ids: tuple[str, ...] = ()
    unresolved_task_ids: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        if int(self.executed_batch_count) < 0:
            raise ValueError("executed_batch_count must be >= 0")
        if self.success and not str(self.execution_digest):
            raise ValueError("successful outcomes require execution_digest")
        if not math.isfinite(float(len(self.completed_task_ids))):
            raise ValueError("invalid completed_task_ids")
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
