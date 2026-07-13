from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


class UnsupportedSchedulerMode(ValueError):
    pass


class SelectedLayerStop(RuntimeError):
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
        while self._restore_callbacks:
            callback = self._restore_callbacks.pop()
            callback()

    def close(self) -> None:
        if self._closed:
            return
        try:
            while self._close_callbacks:
                callback = self._close_callbacks.pop()
                callback()
            self.detach()
        finally:
            self._closed = True


@dataclass(frozen=True)
class RuntimeDecision:
    action: str = "continue"
    details: dict[str, Any] | None = None


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
class ForwardEndEvent:
    pass


RuntimeEvent = (
    ForwardBeginEvent
    | DispatchReadyEvent
    | DispatchCompleteEvent
    | CombineReadyEvent
    | CombineCompleteEvent
    | ForwardEndEvent
)
