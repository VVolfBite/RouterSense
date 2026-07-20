from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class P2PSessionState:
    session_key: tuple[str, ...]
    submitted_handle_count: int
    waited_handle_count: int
    drained_handle_count: int
    poisoned: bool
    failure_code: str = ""
    failure_message: str = ""

    def to_details(self) -> dict[str, object]:
        return {
            "session_key": list(self.session_key),
            "submitted_handle_count": int(self.submitted_handle_count),
            "waited_handle_count": int(self.waited_handle_count),
            "drained_handle_count": int(self.drained_handle_count),
            "session_poisoned": bool(self.poisoned),
            "failure_code": str(self.failure_code),
            "failure_message": str(self.failure_message),
        }


def wait_handles_with_drain(*, handles: list[Any], session_key: tuple[str, ...]) -> tuple[str | None, P2PSessionState]:
    waited = 0
    drained = 0
    failure_type = ""
    failure_message = ""
    for index, handle in enumerate(handles):
        try:
            handle.wait()
            waited += 1
        except Exception as exc:
            waited += 1
            failure_type = type(exc).__name__
            failure_message = str(exc)
            for tail in handles[index + 1 :]:
                try:
                    tail.wait()
                except Exception:
                    pass
                finally:
                    drained += 1
            failure_code = f"work_wait_failed:{failure_type}"
            return failure_code, P2PSessionState(
                session_key=session_key,
                submitted_handle_count=len(handles),
                waited_handle_count=waited,
                drained_handle_count=drained,
                poisoned=True,
                failure_code=failure_code,
                failure_message=failure_message,
            )
    return None, P2PSessionState(
        session_key=session_key,
        submitted_handle_count=len(handles),
        waited_handle_count=waited,
        drained_handle_count=drained,
        poisoned=False,
    )


__all__ = ["P2PSessionState", "wait_handles_with_drain"]
