from __future__ import annotations

"""Adapter that routes backend payloads through the single simulation clock."""

import hashlib
from collections.abc import Mapping
from typing import Any

from rs_sim import KernelPhase, ProgressSignal, SimulationEvent, SimulationKernel


def _ordinal(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


class BackendKernelBridge:
    EVENT_TYPE = "BACKEND_EVENT"

    def __init__(self, kernel: SimulationKernel) -> None:
        self.kernel = kernel
        self.backend: Any | None = None
        self._payload_by_event_id: dict[str, tuple[str, dict[str, Any]]] = {}
        kernel.register_event_handler(self.EVENT_TYPE, self._handle)

    def attach_backend(self, backend: Any) -> None:
        if self.backend is not None and self.backend is not backend:
            raise RuntimeError("backend backend is already attached")
        self.backend = backend

    def schedule_backend_event(
        self,
        *,
        time_ns: int,
        phase_priority: int,
        stable_event_id: str,
        event_kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        try:
            phase = KernelPhase(int(phase_priority))
        except ValueError as exc:
            raise ValueError(f"invalid backend kernel phase {phase_priority}") from exc
        event = self.kernel.schedule(
            time_ns=int(time_ns),
            phase_priority=phase,
            producer="BACKEND",
            event_type=self.EVENT_TYPE,
            ordinal=_ordinal(stable_event_id),
            subject_id=str(stable_event_id),
            attributes=(("backend_event_kind", str(event_kind)),),
        )
        self._payload_by_event_id[event.stable_event_id] = (
            str(event_kind),
            dict(payload),
        )

    def _handle(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        del kernel
        if self.backend is None:
            raise RuntimeError("backend backend handler fired before backend attachment")
        try:
            event_kind, payload = self._payload_by_event_id.pop(event.stable_event_id)
        except KeyError as exc:
            raise RuntimeError(f"missing owner payload for {event.stable_event_id}") from exc
        self.backend.handle_event(
            event_kind=event_kind,
            payload=payload,
            at_ns=event.time_ns,
        )
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=(f"backend:{event_kind}",),
        )


__all__ = ["BackendKernelBridge"]
