from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from rs_sim.scheduler.planning.catalogue import TaskCatalogue
from rs_sim.scheduler.errors import AuthorityError
from rs_sim.scheduler.stable import stable_digest

PENDING_DEPENDENCY = "PENDING_DEPENDENCY"
READY_UNCOMMITTED = "READY_UNCOMMITTED"
COMMITTED = "COMMITTED"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"


def _time_ns(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("at_ns must be a non-negative int")
    return value


@dataclass(frozen=True)
class TaskRuntimeFacts:
    task_id: str
    state: str = PENDING_DEPENDENCY
    permit_granted_at_ns: int | None = None
    source_payload_ready_at_ns: int | None = None
    ready_at_ns: int | None = None
    committed_at_ns: int | None = None
    running_at_ns: int | None = None
    completed_at_ns: int | None = None


class TaskRuntimeIndex:
    """Scheduler-owned canonical logical TaskState.

    transport owns physical reservations and records, while backend owns receiver/assembly
    state.  This index is therefore the only mutable owner of the logical
    PENDING_DEPENDENCY -> READY_UNCOMMITTED -> COMMITTED -> RUNNING -> COMPLETED
    lifecycle.
    """

    def __init__(self, *, catalogue: TaskCatalogue) -> None:
        self.catalogue = catalogue
        self._facts: dict[str, TaskRuntimeFacts] = {}

    def register_catalogue(self, tasks: Iterable[Any] | None = None) -> None:
        """Register newly materialized tasks without rescanning the catalogue.

        ``tasks=None`` remains available for explicit full synchronization, but
        the normal expectation path passes only the tasks created by that edge.
        """

        candidates = self.catalogue.all_tasks() if tasks is None else tuple(tasks)
        for task in candidates:
            task_id = self.catalogue.adapter.task_view(task).task_id
            self._facts.setdefault(task_id, TaskRuntimeFacts(task_id=task_id))

    def ensure(self, task_id: str) -> TaskRuntimeFacts:
        task_id = str(task_id)
        if task_id not in self._facts:
            self.catalogue.get(task_id)
            self._facts[task_id] = TaskRuntimeFacts(task_id=task_id)
        return self._facts[task_id]

    def facts(self, task_id: str) -> TaskRuntimeFacts:
        return self.ensure(task_id)

    def _refresh_ready(self, facts: TaskRuntimeFacts) -> TaskRuntimeFacts:
        if facts.state != PENDING_DEPENDENCY:
            return facts
        if facts.permit_granted_at_ns is None or facts.source_payload_ready_at_ns is None:
            return facts
        ready_at = max(int(facts.permit_granted_at_ns), int(facts.source_payload_ready_at_ns))
        return replace(facts, state=READY_UNCOMMITTED, ready_at_ns=ready_at)

    def note_permit(self, task_id: str, *, at_ns: int) -> TaskRuntimeFacts:
        at_ns = _time_ns(at_ns)
        facts = self.ensure(task_id)
        if facts.permit_granted_at_ns is not None and facts.permit_granted_at_ns != at_ns:
            raise AuthorityError(f"task {task_id} received conflicting permit timestamps")
        facts = replace(facts, permit_granted_at_ns=at_ns)
        facts = self._refresh_ready(facts)
        self._facts[str(task_id)] = facts
        return facts

    def note_source_payload_ready(self, task_id: str, *, at_ns: int) -> TaskRuntimeFacts:
        at_ns = _time_ns(at_ns)
        facts = self.ensure(task_id)
        if facts.source_payload_ready_at_ns is not None and facts.source_payload_ready_at_ns != at_ns:
            raise AuthorityError(f"task {task_id} received conflicting source-ready timestamps")
        facts = replace(facts, source_payload_ready_at_ns=at_ns)
        facts = self._refresh_ready(facts)
        self._facts[str(task_id)] = facts
        return facts

    def _transition(
        self,
        task_id: str,
        *,
        expected: tuple[str, ...],
        target: str,
        at_ns: int,
    ) -> TaskRuntimeFacts:
        at_ns = _time_ns(at_ns)
        facts = self.ensure(task_id)
        if facts.state not in expected:
            raise AuthorityError(
                f"illegal task transition {task_id}: {facts.state} -> {target}; expected {expected}"
            )
        changes: dict[str, Any] = {"state": target}
        if target == COMMITTED:
            changes["committed_at_ns"] = at_ns
        elif target == RUNNING:
            changes["running_at_ns"] = at_ns
        elif target == COMPLETED:
            changes["completed_at_ns"] = at_ns
        facts = replace(facts, **changes)
        self._facts[str(task_id)] = facts
        return facts

    def mark_committed_many(self, task_ids: Iterable[str], *, at_ns: int) -> tuple[TaskRuntimeFacts, ...]:
        """Atomically transition a prepared batch into COMMITTED.

        Every task is validated before any logical state is changed.  This is
        the scheduler half of prepare -> apply receipt -> confirm and prevents a
        partially applied batch if one task is stale or otherwise invalid.
        """

        at_ns = _time_ns(at_ns)
        ids = tuple(str(task_id) for task_id in task_ids)
        if not ids:
            raise AuthorityError("cannot commit an empty task set")
        if len(set(ids)) != len(ids):
            raise AuthorityError("commit task set contains duplicates")
        current = tuple(self.ensure(task_id) for task_id in ids)
        invalid = tuple(facts.task_id for facts in current if facts.state != READY_UNCOMMITTED)
        if invalid:
            raise AuthorityError(
                f"atomic commit requires READY_UNCOMMITTED tasks; invalid={invalid}"
            )
        updated = tuple(
            replace(facts, state=COMMITTED, committed_at_ns=at_ns)
            for facts in current
        )
        self._facts.update({facts.task_id: facts for facts in updated})
        return updated

    def mark_committed(self, task_id: str, *, at_ns: int) -> TaskRuntimeFacts:
        return self.mark_committed_many((task_id,), at_ns=at_ns)[0]

    def mark_running(self, task_id: str, *, at_ns: int) -> TaskRuntimeFacts:
        return self._transition(task_id, expected=(COMMITTED,), target=RUNNING, at_ns=at_ns)

    def mark_completed(self, task_id: str, *, at_ns: int) -> TaskRuntimeFacts:
        # Formal transport always emits TransferStarted before TransferCompleted.
        return self._transition(task_id, expected=(RUNNING,), target=COMPLETED, at_ns=at_ns)

    def task_ids_in_state(self, state: str, *, phase_key: Any | None = None) -> tuple[str, ...]:
        if phase_key is None:
            candidate_ids = tuple(self._facts)
        else:
            candidate_ids = self.catalogue.task_ids_for_phase(phase_key)
        return tuple(task_id for task_id in candidate_ids if self.ensure(task_id).state == str(state))

    def snapshot_digest(self) -> str:
        payload = []
        for task_id in sorted(self._facts):
            facts = self._facts[task_id]
            payload.append(
                {
                    "task_id": facts.task_id,
                    "state": facts.state,
                    "permit_granted_at_ns": facts.permit_granted_at_ns,
                    "source_payload_ready_at_ns": facts.source_payload_ready_at_ns,
                    "ready_at_ns": facts.ready_at_ns,
                    "committed_at_ns": facts.committed_at_ns,
                    "running_at_ns": facts.running_at_ns,
                    "completed_at_ns": facts.completed_at_ns,
                }
            )
        return stable_digest(payload)
