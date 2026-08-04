from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from rs.runtime.guards import InvariantFailure, RouterSenseInvariantError
from rs.scheduling.validation import stable_hash

from rs.runtime.online.megatron_ep.target_planning.contracts import PlanVersionLineage


@dataclass(frozen=True)
class ReleaseBatchTask:
    task_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    plan_version: int = 0
    sender_offset: int = 0
    receiver_offset: int = 0
    tensor_role: str = ""
    peer_sequence: int = 0
    dependency_ids: tuple[str, ...] = ()
    plan_digest: str = ""
    state: str = "pending"


@dataclass
class ReleaseBatchFrontier:
    tasks: list[ReleaseBatchTask]
    max_inflight_release_batches: int = 1
    release_epoch: int = 0
    lineage: list[PlanVersionLineage] = field(default_factory=list)

    def ready_batch(self, *, limit: int) -> list[ReleaseBatchTask]:
        completed = {task.task_id for task in self.tasks if task.state == "completed"}
        in_flight = sum(1 for task in self.tasks if task.state in {"committed", "in_flight"})
        if in_flight >= max(1, int(self.max_inflight_release_batches)):
            return []
        ready: list[ReleaseBatchTask] = []
        for task in self.tasks:
            if task.state not in {"pending", "planned"}:
                continue
            if any(dep not in completed for dep in task.dependency_ids):
                continue
            ready.append(task)
            if len(ready) >= max(0, int(limit)):
                break
        return ready

    def commit_batch(self, *, limit: int) -> list[ReleaseBatchTask]:
        batch = []
        for task in self.ready_batch(limit=limit):
            updated = replace(task, state="committed")
            self._replace(updated)
            batch.append(updated)
        if batch:
            self.release_epoch += 1
        return batch

    def mark_in_flight(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            task = self._get(task_id)
            if task.state != "committed":
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-TRANSPORT-RB-001",
                        stage="release_frontier",
                        message="only committed task can enter in_flight",
                        actual={"task_id": task_id, "state": task.state},
                    )
                )
            self._replace(replace(task, state="in_flight"))

    def mark_completed(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            task = self._get(task_id)
            if task.state not in {"committed", "in_flight"}:
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-TRANSPORT-RB-002",
                        stage="release_frontier",
                        message="only committed/in_flight task can complete",
                        actual={"task_id": task_id, "state": task.state},
                    )
                )
            self._replace(replace(task, state="completed"))

    def immutable_prefix_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.state in {"committed", "in_flight", "completed"})

    def replaceable_suffix_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks if task.state in {"pending", "planned"})

    def immutable_prefix(self) -> tuple[ReleaseBatchTask, ...]:
        return tuple(task for task in self.tasks if task.state in {"committed", "in_flight", "completed"})

    def replaceable_suffix(self) -> tuple[ReleaseBatchTask, ...]:
        return tuple(task for task in self.tasks if task.state in {"pending", "planned"})

    def pending_count(self) -> int:
        return sum(1 for task in self.tasks if task.state in {"pending", "planned"})

    def frontier_digest(self) -> str:
        payload = [
            (
                task.task_id,
                task.state,
                task.plan_version,
                task.peer_sequence,
                task.sender_offset,
                task.receiver_offset,
                task.row_count,
            )
            for task in self.tasks
        ]
        return stable_hash(payload)

    def apply_late_suffix(
        self,
        *,
        new_plan_version: int,
        suffix_tasks: list[ReleaseBatchTask],
        plan_origin: str,
        parent_plan_version: int,
        agreement_token: dict[str, Any],
    ) -> PlanVersionLineage:
        if not bool((agreement_token or {}).get("agreed", False)):
            raise RouterSenseInvariantError(
                InvariantFailure(
                    error_code="RS-TRANSPORT-RB-005",
                    stage="release_frontier",
                    message="late suffix requires positive agreement token",
                    actual=dict(agreement_token or {}),
                )
            )
        immutable = [task for task in self.tasks if task.state in {"committed", "in_flight", "completed"}]
        frontier_digest = self.frontier_digest()
        replacement = [replace(task, plan_version=int(new_plan_version), state="pending") for task in suffix_tasks]
        self.tasks = immutable + replacement
        suffix_digest = stable_hash([(task.task_id, task.plan_version) for task in replacement])
        lineage = PlanVersionLineage(
            old_version=int(parent_plan_version),
            new_version=int(new_plan_version),
            plan_origin=str(plan_origin),
            parent_plan_version=int(parent_plan_version),
            frontier_digest=str(frontier_digest),
            replacement_suffix_digest=str(suffix_digest),
            switch_epoch=int(self.release_epoch),
            all_rank_agreement=bool(agreement_token.get("agreed", False)),
        )
        self.lineage.append(lineage)
        return lineage

    def _get(self, task_id: str) -> ReleaseBatchTask:
        for task in self.tasks:
            if str(task.task_id) == str(task_id):
                return task
        raise RouterSenseInvariantError(
            InvariantFailure(
                error_code="RS-TRANSPORT-RB-003",
                stage="release_frontier",
                message="release frontier task missing",
                actual={"task_id": task_id},
            )
        )

    def _replace(self, updated: ReleaseBatchTask) -> None:
        for idx, task in enumerate(self.tasks):
            if str(task.task_id) == str(updated.task_id):
                self.tasks[idx] = updated
                return
        raise RouterSenseInvariantError(
            InvariantFailure(
                error_code="RS-TRANSPORT-RB-004",
                stage="release_frontier",
                message="release frontier replace task missing",
                actual={"task_id": updated.task_id},
            )
        )
