from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowKey:
    run_id: str
    window_index: int


@dataclass(frozen=True)
class PhaseKey:
    run_id: str
    layer_id: int
    phase_kind: str


@dataclass(frozen=True)
class EdgeKey:
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int


@dataclass(frozen=True)
class ReceiveExpectation:
    edge_key: EdgeKey
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int
    total_expected_payload_bytes: int
    expectation_digest: str
    origin: str
    created_at_ns: int
    zero_edge: bool


@dataclass(frozen=True)
class CanonicalTransferTask:
    task_id: str
    edge_key: EdgeKey
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    expectation_digest: str
    taskization_digest: str
    registered_at_ns: int


@dataclass(frozen=True)
class PhaseExecutionRecord:
    phase_key: PhaseKey
    canonical_task_ids: tuple[str, ...]
    task_catalogue_digest: str
    active_plan_id: str | None
    phase_plan_epoch: int
    committed_task_ids: tuple[str, ...]
    running_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    registered_window_keys: tuple[WindowKey, ...]


class PlanStatus(enum.Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class PlanVersion:
    plan_id: str
    window_key: WindowKey
    version: int
    status: PlanStatus
    supersedes_plan_ids: tuple[str, ...]
    commit_index: int
    committed_task_ids: tuple[str, ...]
    remaining_task_ids: tuple[str, ...]
    created_at_ns: int
    activated_at_ns: int | None
    completed_at_ns: int | None
    plan_digest: str


class SubmitResult(enum.Enum):
    ACCEPTED = "ACCEPTED"
    RETRYABLE_RESOURCE_BUSY = "RETRYABLE_RESOURCE_BUSY"
    RETRYABLE_STALE_AUTHORITY = "RETRYABLE_STALE_AUTHORITY"
    FATAL_CONTRACT_ERROR = "FATAL_CONTRACT_ERROR"
