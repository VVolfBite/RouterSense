from __future__ import annotations

from rs.core.contracts.debug import DebugEvent


class NullDebugProbe:
    def record(self, event: DebugEvent) -> None:
        event  # no-op

    def flush(self) -> tuple[DebugEvent, ...]:
        return ()
