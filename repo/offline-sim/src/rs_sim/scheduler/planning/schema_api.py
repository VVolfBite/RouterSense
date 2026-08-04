from __future__ import annotations

"""Adapter boundary to shared-schema immutable shared schema.

No shared dataclass is defined here.  The test suite supplies fixture-only shared schema
objects; production integration injects the actual shared-schema constructors.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from rs_sim.scheduler.errors import SharedSchemaError
from rs_sim.scheduler.stable import canonical_data


@dataclass(frozen=True)
class ExpectationView:
    edge_key: Any
    phase_key: Any
    src_rank: int
    dst_rank: int
    total_expected_payload_bytes: int
    expectation_digest: str
    origin: str
    created_at_ns: int
    zero_edge: bool


@dataclass(frozen=True)
class CanonicalTaskView:
    task_id: str
    edge_key: Any
    phase_key: Any
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    expectation_digest: str
    taskization_digest: str
    registered_at_ns: int

    def semantic_payload(self, adapter: "SharedSchemaAdapter") -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "edge_key": adapter.edge_payload(self.edge_key),
            "phase_key": adapter.phase_payload(self.phase_key),
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "chunk_index": self.chunk_index,
            "byte_offset": self.byte_offset,
            "payload_bytes": self.payload_bytes,
            "expectation_digest": self.expectation_digest,
            "taskization_digest": self.taskization_digest,
        }


@dataclass(frozen=True)
class PhaseRecordView:
    phase_key: Any
    canonical_task_ids: tuple[str, ...]
    task_catalogue_digest: str
    active_plan_id: str | None
    phase_plan_epoch: int
    committed_task_ids: tuple[str, ...]
    running_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    registered_window_keys: tuple[Any, ...]


@dataclass(frozen=True)
class PlanVersionView:
    plan_id: str
    window_key: Any
    version: int
    status: str
    supersedes_plan_ids: tuple[str, ...]
    commit_index: int
    committed_task_ids: tuple[str, ...]
    remaining_task_ids: tuple[str, ...]
    created_at_ns: int
    activated_at_ns: int | None
    completed_at_ns: int | None
    plan_digest: str


@dataclass(frozen=True)
class SharedSchemaConstructors:
    canonical_task: Callable[..., Any]
    phase_execution_record: Callable[..., Any]
    plan_version: Callable[..., Any]
    plan_status: Callable[[str], Any] = lambda value: value


@runtime_checkable
class SharedSchemaAdapter(Protocol):
    def phase_payload(self, phase_key: Any) -> Any: ...

    def edge_payload(self, edge_key: Any) -> Any: ...

    def window_payload(self, window_key: Any) -> Any: ...

    def expectation_view(self, expectation: Any) -> ExpectationView: ...

    def task_view(self, task: Any) -> CanonicalTaskView: ...

    def make_task(self, **fields: Any) -> Any: ...

    def phase_record_view(self, record: Any) -> PhaseRecordView: ...

    def make_phase_record(self, **fields: Any) -> Any: ...

    def replace_phase_record(self, record: Any, **changes: Any) -> Any: ...

    def plan_view(self, plan: Any) -> PlanVersionView: ...

    def make_plan(self, **fields: Any) -> Any: ...

    def replace_plan(self, plan: Any, **changes: Any) -> Any: ...


class DataclassSchemaAdapter:
    """Strict adapter for the expected shared-schema dataclass field names.

    It is deliberately constructor-injected so scheduler never imports or freezes an
    alternative shared schema.
    """

    def __init__(self, constructors: SharedSchemaConstructors) -> None:
        self._constructors = constructors
        # Shared keys are frozen for the lifetime of one simulation.  Their
        # canonical payloads are requested in the scheduler hot path, so cache
        # by object identity.  Identity caching is deliberately local to one
        # runtime and never participates in serialized semantics.
        self._payload_cache: dict[int, tuple[Any, Any]] = {}

    def _payload(self, value: Any) -> Any:
        cache_key = id(value)
        cached = self._payload_cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        payload = canonical_data(value)
        self._payload_cache[cache_key] = (value, payload)
        return payload

    def phase_payload(self, phase_key: Any) -> Any:
        return self._payload(phase_key)

    def edge_payload(self, edge_key: Any) -> Any:
        return self._payload(edge_key)

    def window_payload(self, window_key: Any) -> Any:
        return self._payload(window_key)

    @staticmethod
    def _require(obj: Any, fields: tuple[str, ...]) -> None:
        missing = [field for field in fields if not hasattr(obj, field)]
        if missing:
            raise SharedSchemaError(
                f"{type(obj).__qualname__} missing frozen shared-schema fields: {', '.join(missing)}"
            )

    def expectation_view(self, expectation: Any) -> ExpectationView:
        fields = (
            "edge_key",
            "phase_key",
            "src_rank",
            "dst_rank",
            "total_expected_payload_bytes",
            "expectation_digest",
            "origin",
            "created_at_ns",
            "zero_edge",
        )
        self._require(expectation, fields)
        payload = {field: getattr(expectation, field) for field in fields}
        edge_key = payload["edge_key"]
        if hasattr(edge_key, "phase_key") and canonical_data(getattr(edge_key, "phase_key")) != canonical_data(payload["phase_key"]):
            raise SharedSchemaError("ReceiveExpectation edge_key.phase_key mismatch")
        if hasattr(edge_key, "src_rank") and int(getattr(edge_key, "src_rank")) != int(payload["src_rank"]):
            raise SharedSchemaError("ReceiveExpectation edge_key.src_rank mismatch")
        if hasattr(edge_key, "dst_rank") and int(getattr(edge_key, "dst_rank")) != int(payload["dst_rank"]):
            raise SharedSchemaError("ReceiveExpectation edge_key.dst_rank mismatch")
        return ExpectationView(**payload)

    def task_view(self, task: Any) -> CanonicalTaskView:
        fields = (
            "task_id",
            "edge_key",
            "phase_key",
            "src_rank",
            "dst_rank",
            "chunk_index",
            "byte_offset",
            "payload_bytes",
            "expectation_digest",
            "taskization_digest",
            "registered_at_ns",
        )
        self._require(task, fields)
        payload = {field: getattr(task, field) for field in fields}
        edge_key = payload["edge_key"]
        if hasattr(edge_key, "phase_key") and canonical_data(getattr(edge_key, "phase_key")) != canonical_data(payload["phase_key"]):
            raise SharedSchemaError("CanonicalTransferTask edge_key.phase_key mismatch")
        if hasattr(edge_key, "src_rank") and int(getattr(edge_key, "src_rank")) != int(payload["src_rank"]):
            raise SharedSchemaError("CanonicalTransferTask edge_key.src_rank mismatch")
        if hasattr(edge_key, "dst_rank") and int(getattr(edge_key, "dst_rank")) != int(payload["dst_rank"]):
            raise SharedSchemaError("CanonicalTransferTask edge_key.dst_rank mismatch")
        return CanonicalTaskView(**payload)

    def make_task(self, **fields: Any) -> Any:
        return self._constructors.canonical_task(**fields)

    def phase_record_view(self, record: Any) -> PhaseRecordView:
        fields = (
            "phase_key",
            "canonical_task_ids",
            "task_catalogue_digest",
            "active_plan_id",
            "phase_plan_epoch",
            "committed_task_ids",
            "running_task_ids",
            "completed_task_ids",
            "registered_window_keys",
        )
        self._require(record, fields)
        payload = {field: getattr(record, field) for field in fields}
        for field in (
            "canonical_task_ids",
            "committed_task_ids",
            "running_task_ids",
            "completed_task_ids",
            "registered_window_keys",
        ):
            payload[field] = tuple(payload[field])
        return PhaseRecordView(**payload)

    def make_phase_record(self, **fields: Any) -> Any:
        return self._constructors.phase_execution_record(**fields)

    def replace_phase_record(self, record: Any, **changes: Any) -> Any:
        if not dataclasses.is_dataclass(record):
            raise SharedSchemaError("shared-schema PhaseExecutionRecord must be a dataclass for immutable replace")
        return dataclasses.replace(record, **changes)

    @staticmethod
    def _status_name(value: Any) -> str:
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
        text = str(value)
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text

    def plan_view(self, plan: Any) -> PlanVersionView:
        fields = (
            "plan_id",
            "window_key",
            "version",
            "status",
            "supersedes_plan_ids",
            "commit_index",
            "committed_task_ids",
            "remaining_task_ids",
            "created_at_ns",
            "activated_at_ns",
            "completed_at_ns",
            "plan_digest",
        )
        self._require(plan, fields)
        payload = {field: getattr(plan, field) for field in fields}
        payload["status"] = self._status_name(payload["status"])
        for field in ("supersedes_plan_ids", "committed_task_ids", "remaining_task_ids"):
            payload[field] = tuple(payload[field])
        return PlanVersionView(**payload)

    def make_plan(self, **fields: Any) -> Any:
        fields = dict(fields)
        fields["status"] = self._constructors.plan_status(str(fields["status"]))
        return self._constructors.plan_version(**fields)

    def replace_plan(self, plan: Any, **changes: Any) -> Any:
        if not dataclasses.is_dataclass(plan):
            raise SharedSchemaError("shared-schema PlanVersion must be a dataclass for immutable replace")
        if "status" in changes:
            changes["status"] = self._constructors.plan_status(str(changes["status"]))
        return dataclasses.replace(plan, **changes)
