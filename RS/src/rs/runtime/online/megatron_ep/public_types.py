from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Protocol


class UnsupportedSchedulerMode(ValueError):
    pass


class SelectedLayerStop(RuntimeError):
    pass


class AggregateRuntimeCloseError(RuntimeError):
    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        super().__init__("runtime close encountered callback failures")


class RuntimeAlreadyAttachedError(RuntimeError):
    pass


class LegacyObserverConflictError(RuntimeError):
    pass


@dataclass
class ControlGroupHandle:
    process_group: Any | None
    group_ranks: tuple[int, ...]
    root_global_rank: int
    root_group_rank: int
    owned: bool = False
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_registry_key"):
            return
        if not self.owned or self.process_group is None:
            return
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group(self.process_group)
        except Exception:
            pass


@dataclass
class RuntimeHandle:
    runtime: Any
    _restore_callbacks: list[Callable[[], None]] = field(default_factory=list)
    _close_callbacks: list[Callable[[], None]] = field(default_factory=list)
    _closed: bool = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def add_restore_callback(self, callback: Callable[[], None]) -> None:
        self._restore_callbacks.append(callback)

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_callbacks.append(callback)

    def detach(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        while self._restore_callbacks:
            callback = self._restore_callbacks.pop()
            try:
                callback()
            except BaseException as exc:  # pragma: no cover - exercised via close tests
                errors.append(exc)
        if errors:
            raise AggregateRuntimeCloseError(errors)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            while self._close_callbacks:
                callback = self._close_callbacks.pop()
                try:
                    callback()
                except BaseException as exc:  # pragma: no cover - exercised via close tests
                    errors.append(exc)
            while self._restore_callbacks:
                callback = self._restore_callbacks.pop()
                try:
                    callback()
                except BaseException as exc:  # pragma: no cover - exercised via close tests
                    errors.append(exc)
        finally:
            self._closed = True
        if errors:
            raise AggregateRuntimeCloseError(errors)


@dataclass(frozen=True)
class RuntimeDecision:
    action: str = "continue"
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeIdentity:
    run_id: str
    forward_generation: int
    microbatch_id: str
    rank: int
    world_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": str(self.run_id),
            "forward_generation": int(self.forward_generation),
            "microbatch_id": str(self.microbatch_id),
            "rank": int(self.rank),
            "world_size": int(self.world_size),
        }


@dataclass(frozen=True)
class PublicationSlot:
    run_id: str
    forward_generation: int
    microbatch_id: str
    source_layer_id: str
    target_layer_id: str
    planning_slot: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "run_id": str(self.run_id),
            "forward_generation": int(self.forward_generation),
            "microbatch_id": str(self.microbatch_id),
            "source_layer_id": str(self.source_layer_id),
            "target_layer_id": str(self.target_layer_id),
            "planning_slot": str(self.planning_slot),
        }

    def semantic_digest(self) -> str:
        import json
        import hashlib

        encoded = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LocalPreparationToken:
    service_session_id: int
    forward_generation: int
    target_layer_id: str
    task_version: int
    publication_slot_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "service_session_id": int(self.service_session_id),
            "forward_generation": int(self.forward_generation),
            "target_layer_id": str(self.target_layer_id),
            "task_version": int(self.task_version),
            "publication_slot_digest": str(self.publication_slot_digest),
        }


@dataclass(frozen=True)
class LocalPublicationCandidate:
    slot: PublicationSlot
    planner_id: str
    logical_plan_digest: str
    token: LocalPreparationToken
    status: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot.semantic_payload(),
            "planner_id": str(self.planner_id),
            "logical_plan_digest": str(self.logical_plan_digest),
            "token": self.token.to_dict(),
            "status": str(self.status),
            "metadata": dict(self.metadata),
        }


class PublicationPollStatus(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SLOT_MISMATCH = "slot_mismatch"


@dataclass(frozen=True)
class PublicationPollResult:
    slot: PublicationSlot
    status: PublicationPollStatus
    root_rank: int | None = None
    published_plan_digest: str | None = None
    canonical_payload: Mapping[str, object] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot.semantic_payload(),
            "status": str(self.status.value),
            "root_rank": None if self.root_rank is None else int(self.root_rank),
            "published_plan_digest": self.published_plan_digest,
            "canonical_payload": dict(self.canonical_payload),
            "details": dict(self.details),
        }


class ControlCommunicationLane(Protocol):
    def poll(self, slot: PublicationSlot, local_candidate: LocalPublicationCandidate | None) -> PublicationPollResult:
        ...

    def cancel_before_generation(
        self,
        *,
        run_id: str,
        microbatch_id: str,
        current_generation: int,
    ) -> None:
        ...


class TargetPlanState(str, Enum):
    EMPTY = "EMPTY"
    LOGICAL_READY = "LOGICAL_READY"
    CLAIMED = "CLAIMED"
    BOUND = "BOUND"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ForwardBeginEvent:
    forward_epoch: int | None = None


@dataclass(frozen=True)
class DispatchReadyEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any
    packed_probs: Any
    layer_role: Literal["prediction_source", "selected", "none"]


@dataclass(frozen=True)
class DispatchCompleteEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any
    result: Any
    layer_role: Literal["prediction_source", "selected", "none"]


@dataclass(frozen=True)
class DispatchFailedEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any
    error: BaseException
    layer_role: Literal["prediction_source", "selected", "none"]


@dataclass(frozen=True)
class CombineReadyEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any


@dataclass(frozen=True)
class CombineCompleteEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any
    result: Any


@dataclass(frozen=True)
class CombineFailedEvent:
    layer_name: str
    dispatcher: Any
    packed_hidden_states: Any
    error: BaseException


@dataclass(frozen=True)
class ForwardEndEvent:
    pass


@dataclass(frozen=True)
class ForwardFailedEvent:
    error: BaseException


RuntimeEvent = (
    ForwardBeginEvent
    | DispatchReadyEvent
    | DispatchCompleteEvent
    | DispatchFailedEvent
    | CombineReadyEvent
    | CombineCompleteEvent
    | CombineFailedEvent
    | ForwardEndEvent
    | ForwardFailedEvent
)
