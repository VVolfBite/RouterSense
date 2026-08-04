from __future__ import annotations

"""Plan/phase/window completion audit records owned by scheduler."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rs_sim.scheduler.stable import canonical_data, stable_digest


@dataclass(frozen=True, slots=True)
class PhaseCompletionRecord:
    phase_key: Any
    completed_task_ids: tuple[str, ...]
    completed_at_ns: int
    record_digest: str


@dataclass(frozen=True, slots=True)
class WindowCompletionRecord:
    window_key: Any
    phase_record_digests: tuple[str, ...]
    completed_at_ns: int
    record_digest: str


class WindowCompletionTracker:
    """Emit each phase and window completion exactly once."""

    def __init__(
        self,
        *,
        authority: Any,
        expected_phases_by_window: dict[Any, Iterable[Any]],
        sink: Callable[[WindowCompletionRecord], None] | None = None,
    ) -> None:
        self.authority = authority
        self._expected = {
            window: tuple(phases) for window, phases in expected_phases_by_window.items()
        }
        self._sink = sink
        self._phase_records: dict[Any, PhaseCompletionRecord] = {}
        self._window_records: dict[Any, WindowCompletionRecord] = {}

    def phase_record(self, phase_key: Any) -> PhaseCompletionRecord | None:
        return self._phase_records.get(phase_key)

    def window_record(self, window_key: Any) -> WindowCompletionRecord | None:
        return self._window_records.get(window_key)

    def on_task_completed(self, phase_key: Any, *, at_ns: int) -> WindowCompletionRecord | None:
        view = self.authority.record_view(phase_key)
        if (
            view.active_plan_id is None
            and not view.committed_task_ids
            and not view.running_task_ids
            and set(view.completed_task_ids) == set(view.canonical_task_ids)
            and phase_key not in self._phase_records
        ):
            payload = {
                "phase_key": canonical_data(phase_key),
                "completed_task_ids": tuple(view.completed_task_ids),
                "completed_at_ns": int(at_ns),
            }
            self._phase_records[phase_key] = PhaseCompletionRecord(
                phase_key=phase_key,
                completed_task_ids=tuple(view.completed_task_ids),
                completed_at_ns=int(at_ns),
                record_digest=stable_digest({"domain": "SCHEDULER_PHASE_COMPLETION", **payload}),
            )
        emitted: WindowCompletionRecord | None = None
        for window_key, phases in self._expected.items():
            if window_key in self._window_records or not phases:
                continue
            if all(phase in self._phase_records for phase in phases):
                phase_digests = tuple(self._phase_records[phase].record_digest for phase in phases)
                payload = {
                    "window_key": canonical_data(window_key),
                    "phase_record_digests": phase_digests,
                    "completed_at_ns": int(at_ns),
                }
                emitted = WindowCompletionRecord(
                    window_key=window_key,
                    phase_record_digests=phase_digests,
                    completed_at_ns=int(at_ns),
                    record_digest=stable_digest({"domain": "SCHEDULER_WINDOW_COMPLETION", **payload}),
                )
                self._window_records[window_key] = emitted
                if self._sink is not None:
                    self._sink(emitted)
        return emitted


__all__ = ["PhaseCompletionRecord", "WindowCompletionRecord", "WindowCompletionTracker"]
