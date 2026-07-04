from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RouterSenseDispatcherFacade:
    """No-op native dispatcher passthrough."""

    native_dispatcher: Callable[..., Any]
    facade_mode: str = "no_op_native_passthrough"
    scheduler_mode: str = "disabled"
    future_hint_mode: str = "none"

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self.native_dispatcher(*args, **kwargs)
