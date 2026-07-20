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


class FormalRuntimeAttachPreflightError(RuntimeError):
    pass


class DispatcherSynchronizationError(RuntimeError):
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
    _cleanup_state: str = "cleanup_pending"
    _last_close_errors: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    @property
    def cleanup_state(self) -> str:
        return str(self._cleanup_state)

    @property
    def last_close_errors(self) -> tuple[str, ...]:
        return tuple(self._last_close_errors)

    def add_restore_callback(self, callback: Callable[[], None]) -> None:
        self._restore_callbacks.append(callback)

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_callbacks.append(callback)

    @staticmethod
    def _run_callbacks(callbacks: list[Callable[[], None]]) -> tuple[list[Callable[[], None]], list[BaseException], bool]:
        remaining: list[Callable[[], None]] = []
        errors: list[BaseException] = []
        executed_any = False
        for callback in reversed(callbacks):
            executed_any = True
            try:
                callback()
            except BaseException as exc:  # pragma: no cover - exercised via close tests
                errors.append(exc)
                remaining.append(callback)
        remaining.reverse()
        return remaining, errors, executed_any

    def detach(self) -> None:
        if self._closed:
            return
        remaining, errors, _executed_any = self._run_callbacks(self._restore_callbacks)
        self._restore_callbacks = remaining
        self._last_close_errors = tuple(f"{type(exc).__name__}: {exc}" for exc in errors)
        self._cleanup_state = "partially_failed" if errors else "closed"
        if errors:
            raise AggregateRuntimeCloseError(errors)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        remaining_close, close_errors, close_executed = self._run_callbacks(self._close_callbacks)
        self._close_callbacks = remaining_close
        remaining_restore, restore_errors, restore_executed = self._run_callbacks(self._restore_callbacks)
        self._restore_callbacks = remaining_restore
        errors.extend(close_errors)
        errors.extend(restore_errors)
        self._last_close_errors = tuple(f"{type(exc).__name__}: {exc}" for exc in errors)
        if errors:
            self._cleanup_state = "partially_failed" if (close_executed or restore_executed) else "cleanup_pending"
            raise AggregateRuntimeCloseError(errors)
        self._closed = True
        self._cleanup_state = "closed"


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
        planning_request_digest = str(dict(self.metadata).get("planning_request_digest", ""))
        h1_prediction_digest = str(dict(self.metadata).get("h1_prediction_digest", ""))
        h2_prediction_digest = str(dict(self.metadata).get("h2_prediction_digest", ""))
        target_problem_digest = str(dict(self.metadata).get("target_problem_digest", ""))
        return {
            "slot": self.slot.semantic_payload(),
            "planner_id": str(self.planner_id),
            "logical_plan_digest": str(self.logical_plan_digest),
            "token": self.token.to_dict(),
            "status": str(self.status),
            "metadata": {
                "planning_request_digest": planning_request_digest,
                "h1_prediction_digest": h1_prediction_digest,
                "h2_prediction_digest": h2_prediction_digest,
                "target_problem_digest": target_problem_digest,
            },
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
