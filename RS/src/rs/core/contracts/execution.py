from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol

from rs.core.contracts.planning import WindowPlan
from rs.scheduling.validation import stable_hash


def _validate_string(name: str, value: str) -> None:
    if not str(value):
        raise ValueError(f"{name} must be non-empty")


def _validate_non_negative(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class PublishedPlan:
    publication_slot: Mapping[str, object]
    window_plan: WindowPlan
    logical_plan_digest: str
    published_plan_digest: str
    root_global_rank: int
    root_group_rank: int
    version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.publication_slot, Mapping):
            raise ValueError("publication_slot must be a mapping payload")
        slot = dict(self.publication_slot)
        for key in ("run_id", "forward_generation", "microbatch_id", "source_layer_id", "target_layer_id", "planning_slot"):
            if key not in slot:
                raise ValueError(f"publication_slot missing {key!r}")
        self.window_plan.validate()
        recomputed_logical = str(self.window_plan.semantic_digest())
        if recomputed_logical != str(self.logical_plan_digest):
            raise ValueError("logical_plan_digest must match window_plan.semantic_digest()")
        _validate_non_negative("root_global_rank", self.root_global_rank)
        _validate_non_negative("root_group_rank", self.root_group_rank)
        _validate_non_negative("version", self.version)
        if not str(self.published_plan_digest):
            raise ValueError("published_plan_digest must be non-empty")
        if str(self.window_plan.request_digest) != str(slot["planning_slot"]) and not str(slot.get("planning_slot", "")):
            raise ValueError("publication_slot planning_slot must be non-empty")
        if str(self.window_plan.metadata.get("source_layer_id", slot["source_layer_id"])) != str(slot["source_layer_id"]):
            raise ValueError("window_plan source_layer_id does not match publication_slot")
        if str(self.window_plan.metadata.get("target_layer_id", slot["target_layer_id"])) != str(slot["target_layer_id"]):
            raise ValueError("window_plan target_layer_id does not match publication_slot")

    def semantic_payload(self) -> dict[str, object]:
        self.window_plan.validate()
        return {
            "semantic_version": "published_plan_v2",
            "publication_slot": dict(self.publication_slot),
            "window_plan": self.window_plan.to_dict(),
            "logical_plan_digest": str(self.logical_plan_digest),
            "root_global_rank": int(self.root_global_rank),
            "root_group_rank": int(self.root_group_rank),
            "version": int(self.version),
        }

    def recompute_published_plan_digest(self) -> str:
        return str(stable_hash(self.semantic_payload()))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "publication_slot": dict(self.publication_slot),
            "window_plan": self.window_plan.to_dict(),
            "logical_plan_digest": str(self.logical_plan_digest),
            "published_plan_digest": str(self.published_plan_digest),
            "root_global_rank": int(self.root_global_rank),
            "root_group_rank": int(self.root_group_rank),
            "version": int(self.version),
            "metadata": dict(self.metadata),
        }


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
        _validate_string("payload_role", self.payload_role)
        _validate_non_negative("row_count", self.row_count)
        _validate_non_negative("element_count", self.element_count)
        _validate_non_negative("byte_count", self.byte_count)
        _validate_non_negative("bytes_per_row", self.bytes_per_row)
        _validate_string("dtype", self.dtype)
        for dim in self.shape_suffix:
            _validate_non_negative("shape_suffix dim", int(dim))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "payload_role": str(self.payload_role),
            "row_count": int(self.row_count),
            "element_count": int(self.element_count),
            "byte_count": int(self.byte_count),
            "bytes_per_row": int(self.bytes_per_row),
            "dtype": str(self.dtype),
            "shape_suffix": [int(dim) for dim in self.shape_suffix],
        }


@dataclass(frozen=True)
class ActualPhaseContext:
    layer_id: str
    phase: str
    world_size: int
    rank_space: str
    layout_digest: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_string("layer_id", self.layer_id)
        _validate_string("phase", self.phase)
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        _validate_string("rank_space", self.rank_space)
        _validate_string("layout_digest", self.layout_digest)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "world_size": int(self.world_size),
            "rank_space": str(self.rank_space),
            "layout_digest": str(self.layout_digest),
            "metadata": dict(self.metadata),
        }


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
        _validate_string("task_id", self.task_id)
        _validate_string("flow_id", self.flow_id)
        _validate_string("payload_role", self.payload_role)
        _validate_non_negative("src_rank", self.src_rank)
        _validate_non_negative("dst_rank", self.dst_rank)
        if int(self.row_count) <= 0:
            raise ValueError("row_count must be > 0")
        _validate_non_negative("send_offset_rows", self.send_offset_rows)
        _validate_non_negative("recv_offset_rows", self.recv_offset_rows)
        for dep in self.dependency_ids:
            _validate_string("dependency_id", str(dep))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": str(self.task_id),
            "flow_id": str(self.flow_id),
            "payload_role": str(self.payload_role),
            "src_rank": int(self.src_rank),
            "dst_rank": int(self.dst_rank),
            "row_count": int(self.row_count),
            "send_offset_rows": int(self.send_offset_rows),
            "recv_offset_rows": int(self.recv_offset_rows),
            "dependency_ids": [str(dep) for dep in self.dependency_ids],
        }


@dataclass(frozen=True)
class ExecutionBatch:
    batch_id: str
    wave_id: int
    slices: tuple[TransferSlice, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_string("batch_id", self.batch_id)
        _validate_non_negative("wave_id", self.wave_id)
        seen_task_ids: set[str] = set()
        for item in self.slices:
            item.validate()
            if item.task_id in seen_task_ids:
                raise ValueError(f"duplicate task_id within batch: {item.task_id!r}")
            seen_task_ids.add(str(item.task_id))

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
    publication_slot: Mapping[str, object]
    local_global_rank: int
    local_group_rank: int
    phase: str
    payload_specs: tuple[PayloadSpec, ...]
    batches: tuple[ExecutionBatch, ...]
    expected_outgoing_rows: Mapping[str, tuple[int, ...]]
    expected_incoming_rows: Mapping[str, tuple[int, ...]]
    logical_plan_digest: str
    published_plan_digest: str
    layout_digest: str
    materialized_plan_digest: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.publication_slot, Mapping):
            raise ValueError("publication_slot must be a mapping payload")
        for key in ("run_id", "forward_generation", "microbatch_id", "source_layer_id", "target_layer_id", "planning_slot"):
            if key not in self.publication_slot:
                raise ValueError(f"publication_slot missing {key!r}")
        _validate_non_negative("local_global_rank", self.local_global_rank)
        _validate_non_negative("local_group_rank", self.local_group_rank)
        _validate_string("phase", self.phase)
        _validate_string("logical_plan_digest", self.logical_plan_digest)
        _validate_string("published_plan_digest", self.published_plan_digest)
        _validate_string("layout_digest", self.layout_digest)
        _validate_string("materialized_plan_digest", self.materialized_plan_digest)
        payload_roles = {str(item.payload_role) for item in self.payload_specs}
        for item in self.payload_specs:
            item.validate()
        seen_batch_ids: set[str] = set()
        for batch in self.batches:
            batch.validate()
            if batch.batch_id in seen_batch_ids:
                raise ValueError(f"duplicate batch_id {batch.batch_id!r}")
            seen_batch_ids.add(str(batch.batch_id))
            for item in batch.slices:
                if str(item.payload_role) not in payload_roles:
                    raise ValueError(f"slice payload_role {item.payload_role!r} missing from payload_specs")
        for mapping_name, mapping in {
            "expected_outgoing_rows": self.expected_outgoing_rows,
            "expected_incoming_rows": self.expected_incoming_rows,
        }.items():
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{mapping_name} must be a mapping")
            for role, rows in mapping.items():
                _validate_string(mapping_name, str(role))
                for row_count in rows:
                    _validate_non_negative(f"{mapping_name} row_count", int(row_count))

    def semantic_payload(self) -> dict[str, object]:
        return {
            "semantic_version": "materialized_plan_v2",
            "publication_slot": dict(self.publication_slot),
            "local_global_rank": int(self.local_global_rank),
            "local_group_rank": int(self.local_group_rank),
            "phase": str(self.phase),
            "payload_specs": [item.to_dict() for item in self.payload_specs],
            "batches": [item.to_dict() for item in self.batches],
            "expected_outgoing_rows": {
                str(role): [int(value) for value in rows]
                for role, rows in self.expected_outgoing_rows.items()
            },
            "expected_incoming_rows": {
                str(role): [int(value) for value in rows]
                for role, rows in self.expected_incoming_rows.items()
            },
            "logical_plan_digest": str(self.logical_plan_digest),
            "published_plan_digest": str(self.published_plan_digest),
            "layout_digest": str(self.layout_digest),
        }

    def recompute_materialized_plan_digest(self) -> str:
        return str(stable_hash(self.semantic_payload()))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **self.semantic_payload(),
            "materialized_plan_digest": str(self.materialized_plan_digest),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    stage: str
    reason: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": bool(self.valid),
            "stage": str(self.stage),
            "reason": str(self.reason),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    forward_generation: int
    layer_id: str
    phase: str
    rank_space: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_string("run_id", self.run_id)
        _validate_non_negative("forward_generation", self.forward_generation)
        _validate_string("layer_id", self.layer_id)
        _validate_string("phase", self.phase)
        _validate_string("rank_space", self.rank_space)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "run_id": str(self.run_id),
            "forward_generation": int(self.forward_generation),
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "rank_space": str(self.rank_space),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionOutcome:
    success: bool
    output_payload: object | None
    submitted_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    unresolved_task_ids: tuple[str, ...]
    executed_batch_count: int
    all_work_completed: bool
    failure_code: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_non_negative("executed_batch_count", self.executed_batch_count)
        for collection_name, values in {
            "submitted_task_ids": self.submitted_task_ids,
            "completed_task_ids": self.completed_task_ids,
            "failed_task_ids": self.failed_task_ids,
            "unresolved_task_ids": self.unresolved_task_ids,
        }.items():
            for value in values:
                _validate_string(collection_name, str(value))
        if self.success and (self.failed_task_ids or self.unresolved_task_ids or not self.all_work_completed):
            raise ValueError("successful outcome cannot contain failed/unresolved work")
        if not self.success and not self.failure_code:
            raise ValueError("failed outcome requires failure_code")
        if not math.isfinite(float(len(self.submitted_task_ids))):
            raise ValueError("invalid submitted_task_ids")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "success": bool(self.success),
            "output_payload": self.output_payload,
            "submitted_task_ids": [str(value) for value in self.submitted_task_ids],
            "completed_task_ids": [str(value) for value in self.completed_task_ids],
            "failed_task_ids": [str(value) for value in self.failed_task_ids],
            "unresolved_task_ids": [str(value) for value in self.unresolved_task_ids],
            "executed_batch_count": int(self.executed_batch_count),
            "all_work_completed": bool(self.all_work_completed),
            "failure_code": self.failure_code,
            "details": dict(self.details),
        }


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
    def validate(self, *, plan: MaterializedPlan, invocation: Any, context: ExecutionContext) -> ValidationResult:
        ...


class Executor(Protocol):
    def execute(self, *, plan: MaterializedPlan, invocation: Any, context: ExecutionContext) -> ExecutionOutcome:
        ...
