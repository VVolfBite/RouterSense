from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.contracts.factories import make_authority_stamp
from rs_sim.contracts.schema import AuthorityStamp

from rs_sim.scheduler.planning.catalogue import TaskCatalogue
from rs_sim.scheduler.errors import AuthorityError
from rs_sim.scheduler.planning.schema_api import SharedSchemaAdapter
from rs_sim.scheduler.stable import stable_digest, stable_id, stable_json
from rs_sim.scheduler.execution.state import COMPLETED, COMMITTED, RUNNING, TaskRuntimeIndex


@dataclass(frozen=True)
class AuthorityToken:
    phase_token: str
    plan_id: str
    epoch: int




@dataclass(frozen=True, slots=True)
class WindowAuthorityProposal:
    source_window_key: Any
    target_phase_key: Any
    ordered_task_ids: tuple[str, ...]
    proposed_at_ns: int
    proposal_digest: str


@dataclass(frozen=True, slots=True)
class AuthoritySupersessionRecord:
    source_window_key: Any
    target_phase_key: Any
    old_plan_id: str | None
    new_plan_id: str
    phase_plan_epoch: int
    frozen_task_digest: str
    superseded_at_ns: int
    record_digest: str


class PhaseAuthorityManager:
    """Single active authority and immutable PlanVersion lifecycle per phase."""

    def __init__(
        self,
        *,
        adapter: SharedSchemaAdapter,
        catalogue: TaskCatalogue,
        runtime: TaskRuntimeIndex,
    ) -> None:
        self.adapter = adapter
        self.catalogue = catalogue
        self.runtime = runtime
        self._records: dict[str, Any] = {}
        self._plans: dict[str, Any] = {}
        self._plan_phase_tokens: dict[str, str] = {}
        self._window_versions: dict[str, int] = defaultdict(int)
        self._draft_order: dict[str, tuple[str, ...]] = {}
        self._latest_proposal_key_by_phase: dict[str, tuple[int, str, str]] = {}
        self._supersession_records: list[AuthoritySupersessionRecord] = []
        self._completed_once: set[str] = set()
        self._duplicate_execution_count = 0
        self._phase_token_cache: dict[int, tuple[Any, str]] = {}
        self._window_token_cache: dict[int, tuple[Any, str]] = {}
        self._record_view_cache: dict[str, tuple[Any, Any]] = {}
        self._record_catalogue_revision: dict[str, int] = {}

    @staticmethod
    def _identity_cached_token(
        cache: dict[int, tuple[Any, str]], value: Any, payload: Any
    ) -> str:
        cache_key = id(value)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        token = stable_json(payload)
        cache[cache_key] = (value, token)
        return token

    def _phase_token(self, phase_key: Any) -> str:
        return self._identity_cached_token(
            self._phase_token_cache, phase_key, self.adapter.phase_payload(phase_key)
        )

    def _window_token(self, window_key: Any) -> str:
        return self._identity_cached_token(
            self._window_token_cache, window_key, self.adapter.window_payload(window_key)
        )

    def _set_record(
        self, token: str, record: Any, *, catalogue_revision: int | None = None
    ) -> Any:
        self._records[token] = record
        self._record_view_cache.pop(token, None)
        if catalogue_revision is not None:
            self._record_catalogue_revision[token] = int(catalogue_revision)
        return record

    def _view_for_token(self, token: str, record: Any) -> Any:
        cached = self._record_view_cache.get(token)
        if cached is not None and cached[0] is record:
            return cached[1]
        view = self.adapter.phase_record_view(record)
        self._record_view_cache[token] = (record, view)
        return view

    def ensure_phase(self, phase_key: Any, *, registered_window_keys: Iterable[Any] = ()) -> Any:
        token = self._phase_token(phase_key)
        catalogue_revision = self.catalogue.phase_revision(phase_key)
        task_ids = self.catalogue.task_ids_for_phase(phase_key)
        digest = self.catalogue.phase_digest(phase_key)
        if token in self._records:
            record = self._records[token]
            view = self._view_for_token(token, record)
            old_ids = tuple(view.canonical_task_ids)
            new_ids = tuple(task_ids)
            if old_ids != new_ids[: len(old_ids)]:
                raise AuthorityError("PhaseExecutionRecord canonical task catalogue was removed or reordered")
            catalogue_changed = old_ids != new_ids or str(view.task_catalogue_digest) != str(digest)
            windows = list(view.registered_window_keys)
            known = {stable_json(self.adapter.window_payload(item)) for item in windows}
            changed = False
            for item in registered_window_keys:
                item_token = stable_json(self.adapter.window_payload(item))
                if item_token not in known:
                    windows.append(item)
                    known.add(item_token)
                    changed = True
            if changed or catalogue_changed:
                record = self.adapter.replace_phase_record(
                    record,
                    canonical_task_ids=new_ids,
                    task_catalogue_digest=str(digest),
                    registered_window_keys=tuple(windows),
                )
                self._set_record(
                    token, record, catalogue_revision=catalogue_revision
                )
            else:
                self._record_catalogue_revision[token] = catalogue_revision
            return record

        record = self.adapter.make_phase_record(
            phase_key=phase_key,
            canonical_task_ids=tuple(task_ids),
            task_catalogue_digest=str(digest),
            active_plan_id=None,
            phase_plan_epoch=0,
            committed_task_ids=(),
            running_task_ids=(),
            completed_task_ids=(),
            registered_window_keys=tuple(registered_window_keys),
        )
        return self._set_record(
            token, record, catalogue_revision=catalogue_revision
        )

    def record(self, phase_key: Any) -> Any:
        token = self._phase_token(phase_key)
        record = self._records.get(token)
        if record is None:
            return self.ensure_phase(phase_key)
        if self._record_catalogue_revision.get(token, -1) != self.catalogue.phase_revision(phase_key):
            return self.ensure_phase(phase_key)
        return record

    def record_view(self, phase_key: Any):
        token = self._phase_token(phase_key)
        record = self.record(phase_key)
        return self._view_for_token(token, record)

    def registered_windows(self, phase_key: Any) -> tuple[Any, ...]:
        """Return every overlapping window registered against one phase authority."""

        return tuple(self.record_view(phase_key).registered_window_keys)

    def phase_authority_snapshot(self, phase_key: Any) -> dict[str, Any]:
        """Stable audit view proving cross-window single-authority ownership."""

        view = self.record_view(phase_key)
        return {
            "phase_token": self._phase_token(phase_key),
            "registered_window_keys": tuple(view.registered_window_keys),
            "active_plan_id": view.active_plan_id,
            "phase_plan_epoch": int(view.phase_plan_epoch),
            "canonical_task_ids": tuple(view.canonical_task_ids),
            "committed_task_ids": tuple(view.committed_task_ids),
            "running_task_ids": tuple(view.running_task_ids),
            "completed_task_ids": tuple(view.completed_task_ids),
        }

    @property
    def supersession_records(self) -> tuple[AuthoritySupersessionRecord, ...]:
        return tuple(self._supersession_records)

    @property
    def supersession_count(self) -> int:
        return sum(1 for item in self._supersession_records if item.old_plan_id is not None)

    @property
    def duplicate_execution_count(self) -> int:
        return int(self._duplicate_execution_count)

    def frozen_task_digest(self, phase_key: Any) -> str:
        view = self.record_view(phase_key)
        return stable_digest(
            {
                "committed_task_ids": tuple(view.committed_task_ids),
                "running_task_ids": tuple(view.running_task_ids),
                "completed_task_ids": tuple(view.completed_task_ids),
                "runtime_states": tuple(
                    (task_id, self.runtime.facts(task_id).state)
                    for task_id in sorted(
                        set(view.committed_task_ids)
                        | set(view.running_task_ids)
                        | set(view.completed_task_ids)
                    )
                ),
            }
        )

    def propose_window_order(
        self,
        *,
        source_window_key: Any,
        target_phase_key: Any,
        ordered_task_ids: Iterable[str],
        proposed_at_ns: int,
    ) -> WindowAuthorityProposal:
        ordered = tuple(str(item) for item in ordered_task_ids)
        semantic = {
            "source_window_key": self.adapter.window_payload(source_window_key),
            "target_phase_key": self.adapter.phase_payload(target_phase_key),
            "ordered_task_ids": ordered,
            "proposed_at_ns": int(proposed_at_ns),
        }
        return WindowAuthorityProposal(
            source_window_key=source_window_key,
            target_phase_key=target_phase_key,
            ordered_task_ids=ordered,
            proposed_at_ns=int(proposed_at_ns),
            proposal_digest=stable_digest(semantic),
        )

    def handoff_window_proposal(self, proposal: WindowAuthorityProposal) -> Any:
        """Deterministically hand a read-only window proposal to unique phase authority.

        Only the uncommitted frontier may change.  Older proposals are rejected
        without mutating authority.  Frozen task identities and states are
        included in every supersession audit record.
        """

        if not isinstance(proposal, WindowAuthorityProposal):
            raise TypeError("proposal must be WindowAuthorityProposal")
        phase_token = self._phase_token(proposal.target_phase_key)
        proposal_key = (
            int(proposal.proposed_at_ns),
            self._window_token(proposal.source_window_key),
            str(proposal.proposal_digest),
        )
        previous_key = self._latest_proposal_key_by_phase.get(phase_token)
        if previous_key is not None and proposal_key <= previous_key:
            raise AuthorityError("stale or non-monotonic window proposal")
        record_before = self.ensure_phase(
            proposal.target_phase_key,
            registered_window_keys=(proposal.source_window_key,),
        )
        view_before = self.adapter.phase_record_view(record_before)
        old_plan_id = None if view_before.active_plan_id is None else str(view_before.active_plan_id)
        frozen_before = self.frozen_task_digest(proposal.target_phase_key)
        active = self.create_and_activate(
            phase_key=proposal.target_phase_key,
            window_key=proposal.source_window_key,
            ordered_task_ids=proposal.ordered_task_ids,
            now_ns=int(proposal.proposed_at_ns),
        )
        view_after = self.record_view(proposal.target_phase_key)
        frozen_after = self.frozen_task_digest(proposal.target_phase_key)
        if frozen_after != frozen_before:
            raise AuthorityError("window handoff mutated frozen task state")
        self._latest_proposal_key_by_phase[phase_token] = proposal_key
        new_plan_id = str(self.adapter.plan_view(active).plan_id)
        payload = {
            "source_window_key": self.adapter.window_payload(proposal.source_window_key),
            "target_phase_key": self.adapter.phase_payload(proposal.target_phase_key),
            "old_plan_id": old_plan_id,
            "new_plan_id": new_plan_id,
            "phase_plan_epoch": int(view_after.phase_plan_epoch),
            "frozen_task_digest": frozen_after,
            "superseded_at_ns": int(proposal.proposed_at_ns),
        }
        self._supersession_records.append(
            AuthoritySupersessionRecord(**payload, record_digest=stable_digest(payload))
        )
        return active

    def plan(self, plan_id: str) -> Any:
        try:
            return self._plans[str(plan_id)]
        except KeyError as exc:
            raise AuthorityError(f"unknown plan_id {plan_id}") from exc

    def active_plan(self, phase_key: Any) -> Any | None:
        view = self.record_view(phase_key)
        return None if view.active_plan_id is None else self.plan(view.active_plan_id)

    def draft_plan(
        self,
        *,
        phase_key: Any,
        window_key: Any,
        ordered_task_ids: Iterable[str],
        created_at_ns: int,
    ) -> Any:
        record = self.ensure_phase(phase_key, registered_window_keys=(window_key,))
        record_view = self.adapter.phase_record_view(record)
        canonical_ids = tuple(record_view.canonical_task_ids)
        ordered = tuple(str(item) for item in ordered_task_ids)
        if len(set(ordered)) != len(ordered):
            raise AuthorityError("plan order contains duplicate task IDs")
        unknown = set(ordered) - set(canonical_ids)
        if unknown:
            raise AuthorityError(f"plan order contains non-canonical tasks: {sorted(unknown)}")

        frozen = set(record_view.committed_task_ids) | set(record_view.running_task_ids) | set(
            record_view.completed_task_ids
        )
        expected_remaining = set(canonical_ids) - frozen
        if set(ordered) != expected_remaining:
            missing = expected_remaining - set(ordered)
            extra = set(ordered) - expected_remaining
            raise AuthorityError(
                f"plan must cover every and only unfrozen task; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        window_token = self._window_token(window_key)
        version = self._window_versions[window_token] + 1
        self._window_versions[window_token] = version
        supersedes: tuple[str, ...] = ()
        committed_history = tuple(
            dict.fromkeys(
                (*record_view.committed_task_ids, *record_view.running_task_ids, *record_view.completed_task_ids)
            )
        )
        if record_view.active_plan_id is not None:
            supersedes = (str(record_view.active_plan_id),)
            active_history = self.adapter.plan_view(self.plan(record_view.active_plan_id)).committed_task_ids
            committed_history = tuple(dict.fromkeys((*active_history, *committed_history)))
        plan_semantics = {
            "phase_key": self.adapter.phase_payload(phase_key),
            "window_key": self.adapter.window_payload(window_key),
            "version": int(version),
            "supersedes_plan_ids": supersedes,
            "committed_task_ids": committed_history,
            "remaining_task_ids": ordered,
            "created_at_ns": int(created_at_ns),
        }
        plan_id = stable_id("plan", plan_semantics)
        plan_digest = stable_digest(plan_semantics)
        plan = self.adapter.make_plan(
            plan_id=plan_id,
            window_key=window_key,
            version=int(version),
            status="DRAFT",
            supersedes_plan_ids=supersedes,
            commit_index=len(committed_history),
            committed_task_ids=committed_history,
            remaining_task_ids=ordered,
            created_at_ns=int(created_at_ns),
            activated_at_ns=None,
            completed_at_ns=None,
            plan_digest=plan_digest,
        )
        self._plans[plan_id] = plan
        self._plan_phase_tokens[plan_id] = self._phase_token(phase_key)
        self._draft_order[plan_id] = ordered
        return plan

    def activate_plan(self, plan_id: str, *, activated_at_ns: int) -> Any:
        plan = self.plan(plan_id)
        view = self.adapter.plan_view(plan)
        if view.status != "DRAFT":
            raise AuthorityError(f"only DRAFT plans may activate, got {view.status}")
        phase_token = self._plan_phase_tokens[str(plan_id)]
        record = self._records[phase_token]
        record_view = self.adapter.phase_record_view(record)

        actual_supersedes: tuple[str, ...] = ()
        if record_view.active_plan_id is not None:
            actual_supersedes = (str(record_view.active_plan_id),)
        if tuple(view.supersedes_plan_ids) != actual_supersedes:
            raise AuthorityError("draft plan was built against stale active authority")
        if record_view.active_plan_id is not None:
            old = self.plan(record_view.active_plan_id)
            old_view = self.adapter.plan_view(old)
            if old_view.status not in ("ACTIVE", "DRAFT"):
                raise AuthorityError(f"cannot supersede plan in state {old_view.status}")
            old = self.adapter.replace_plan(old, status="SUPERSEDED")
            self._plans[old_view.plan_id] = old

        frozen = set(record_view.committed_task_ids) | set(record_view.running_task_ids) | set(
            record_view.completed_task_ids
        )
        ordered = tuple(task_id for task_id in self._draft_order[str(plan_id)] if task_id not in frozen)
        active = self.adapter.replace_plan(
            plan,
            status="ACTIVE",
            commit_index=len(view.committed_task_ids),
            committed_task_ids=tuple(view.committed_task_ids),
            remaining_task_ids=ordered,
            activated_at_ns=int(activated_at_ns),
        )
        self._plans[str(plan_id)] = active
        updated_record = self.adapter.replace_phase_record(
            record,
            active_plan_id=str(plan_id),
            phase_plan_epoch=int(record_view.phase_plan_epoch) + 1,
        )
        self._records[phase_token] = updated_record
        return active


    def reject_plan(self, plan_id: str, *, rejected_at_ns: int) -> Any:
        plan = self.plan(plan_id)
        view = self.adapter.plan_view(plan)
        if view.status not in ("DRAFT", "ACTIVE"):
            raise AuthorityError(f"cannot reject plan in state {view.status}")
        rejected = self.adapter.replace_plan(
            plan, status="REJECTED", completed_at_ns=int(rejected_at_ns)
        )
        self._plans[str(plan_id)] = rejected
        phase_token = self._plan_phase_tokens[str(plan_id)]
        record = self._records[phase_token]
        record_view = self.adapter.phase_record_view(record)
        if record_view.active_plan_id == str(plan_id):
            self._records[phase_token] = self.adapter.replace_phase_record(
                record,
                active_plan_id=None,
                phase_plan_epoch=int(record_view.phase_plan_epoch) + 1,
            )
        return rejected

    def create_and_activate(
        self,
        *,
        phase_key: Any,
        window_key: Any,
        ordered_task_ids: Iterable[str],
        now_ns: int,
    ) -> Any:
        draft = self.draft_plan(
            phase_key=phase_key,
            window_key=window_key,
            ordered_task_ids=ordered_task_ids,
            created_at_ns=int(now_ns),
        )
        return self.activate_plan(self.adapter.plan_view(draft).plan_id, activated_at_ns=int(now_ns))

    def authority_token(self, phase_key: Any) -> AuthorityToken:
        view = self.record_view(phase_key)
        if view.active_plan_id is None:
            raise AuthorityError("phase has no active scheduling authority")
        return AuthorityToken(
            phase_token=self._phase_token(phase_key),
            plan_id=str(view.active_plan_id),
            epoch=int(view.phase_plan_epoch),
        )

    def authority_stamp(self, phase_key: Any) -> AuthorityStamp:
        token = self.authority_token(phase_key)
        return make_authority_stamp(
            phase_token=token.phase_token,
            plan_id=token.plan_id,
            phase_plan_epoch=token.epoch,
        )

    def stamp_is_current(self, phase_key: Any, stamp: AuthorityStamp) -> bool:
        if not isinstance(stamp, AuthorityStamp):
            return False
        expected_phase_token = self._phase_token(phase_key)
        if stamp.phase_token != expected_phase_token:
            return False
        current = self.authority_stamp(phase_key)
        return current == stamp

    def token_is_current(self, token: AuthorityToken) -> bool:
        record = self._records.get(token.phase_token)
        if record is None:
            return False
        view = self.adapter.phase_record_view(record)
        return view.active_plan_id == token.plan_id and int(view.phase_plan_epoch) == int(token.epoch)

    def apply_commit_receipt(self, receipt: Any) -> None:
        """Atomically apply the Scheduler-owned logical commit after transport prepare.

        transport owns only the temporary physical resource reservation represented by
        the receipt.  This method remains the unique transition of canonical
        TaskState into COMMITTED.
        """
        phase_key = receipt.phase_key
        if not self.stamp_is_current(phase_key, receipt.authority_stamp):
            raise AuthorityError("commit receipt authority stamp does not match active plan")
        self.commit_batch(
            phase_key,
            tuple(str(item) for item in receipt.task_ids),
            at_ns=int(receipt.commit_time_ns),
        )

    def commit_batch(self, phase_key: Any, task_ids: Iterable[str], *, at_ns: int) -> None:
        ids = tuple(str(item) for item in task_ids)
        if not ids:
            raise AuthorityError("cannot commit an empty batch")
        if len(set(ids)) != len(ids):
            raise AuthorityError("batch contains duplicate task IDs")
        record = self.record(phase_key)
        view = self.adapter.phase_record_view(record)
        if view.active_plan_id is None:
            raise AuthorityError("cannot commit without active plan")
        active = self.plan(view.active_plan_id)
        active_view = self.adapter.plan_view(active)
        if active_view.status != "ACTIVE":
            raise AuthorityError("active_plan_id does not reference ACTIVE plan")
        remaining = list(active_view.remaining_task_ids)
        if any(task_id not in remaining for task_id in ids):
            raise AuthorityError("batch contains task absent from active remaining order")

        # Construct every immutable shared object before changing canonical
        # TaskState.  After these validations succeed, mark_committed_many is
        # itself atomic and the final dictionary assignments are infallible.
        next_remaining = list(remaining)
        for task_id in ids:
            next_remaining.remove(task_id)
        record_committed = tuple((*view.committed_task_ids, *ids))
        plan_committed_history = tuple(dict.fromkeys((*active_view.committed_task_ids, *ids)))
        updated_active = self.adapter.replace_plan(
            active,
            commit_index=int(active_view.commit_index) + len(ids),
            committed_task_ids=plan_committed_history,
            remaining_task_ids=tuple(next_remaining),
        )
        updated_record = self.adapter.replace_phase_record(
            record, committed_task_ids=record_committed
        )
        self.runtime.mark_committed_many(ids, at_ns=int(at_ns))
        self._plans[active_view.plan_id] = updated_active
        self._records[self._phase_token(phase_key)] = updated_record

    def mark_running(self, phase_key: Any, task_id: str, *, at_ns: int) -> None:
        record = self.record(phase_key)
        view = self.adapter.phase_record_view(record)
        if task_id not in view.committed_task_ids:
            raise AuthorityError("running task is not committed")
        committed = tuple(item for item in view.committed_task_ids if item != task_id)
        running = tuple((*view.running_task_ids, task_id))
        updated_record = self.adapter.replace_phase_record(
            record, committed_task_ids=committed, running_task_ids=running
        )
        self.runtime.mark_running(task_id, at_ns=int(at_ns))
        self._records[self._phase_token(phase_key)] = updated_record

    def mark_completed(self, phase_key: Any, task_id: str, *, at_ns: int) -> None:
        if task_id in self._completed_once:
            self._duplicate_execution_count += 1
            raise AuthorityError("duplicate physical task completion")
        record = self.record(phase_key)
        view = self.adapter.phase_record_view(record)
        if task_id not in view.running_task_ids:
            raise AuthorityError("completed task is not RUNNING")
        committed = tuple(item for item in view.committed_task_ids if item != task_id)
        running = tuple(item for item in view.running_task_ids if item != task_id)
        completed = tuple((*view.completed_task_ids, task_id))
        updated_record = self.adapter.replace_phase_record(
            record,
            committed_task_ids=committed,
            running_task_ids=running,
            completed_task_ids=completed,
        )
        self.runtime.mark_completed(task_id, at_ns=int(at_ns))
        self._completed_once.add(str(task_id))
        self._records[self._phase_token(phase_key)] = updated_record
        updated_view = self.adapter.phase_record_view(updated_record)
        active = self.active_plan(phase_key)
        if active is not None:
            active_view = self.adapter.plan_view(active)
            if (
                active_view.status == "ACTIVE"
                and not active_view.remaining_task_ids
                and not updated_view.committed_task_ids
                and not updated_view.running_task_ids
                and set(updated_view.completed_task_ids) == set(updated_view.canonical_task_ids)
            ):
                self.complete_active_plan(phase_key, completed_at_ns=int(at_ns))

    def complete_active_plan(self, phase_key: Any, *, completed_at_ns: int) -> Any:
        record = self.record(phase_key)
        record_view = self.adapter.phase_record_view(record)
        if record_view.active_plan_id is None:
            raise AuthorityError("phase has no active plan to complete")
        plan = self.plan(record_view.active_plan_id)
        plan_view = self.adapter.plan_view(plan)
        if plan_view.status != "ACTIVE":
            raise AuthorityError(f"only ACTIVE plan may complete, got {plan_view.status}")
        if plan_view.remaining_task_ids or record_view.committed_task_ids or record_view.running_task_ids:
            raise AuthorityError("cannot complete plan while tasks remain or are in flight")
        if set(record_view.completed_task_ids) != set(record_view.canonical_task_ids):
            raise AuthorityError("cannot complete plan before every canonical task completes")
        completed = self.adapter.replace_plan(
            plan, status="COMPLETED", completed_at_ns=int(completed_at_ns)
        )
        self._plans[plan_view.plan_id] = completed
        self._records[self._phase_token(phase_key)] = self.adapter.replace_phase_record(
            record,
            active_plan_id=None,
            phase_plan_epoch=int(record_view.phase_plan_epoch) + 1,
        )
        return completed

    def stable_payload(self) -> dict[str, Any]:
        records = []
        for token in sorted(self._records):
            view = self.adapter.phase_record_view(self._records[token])
            records.append(
                {
                    "phase": self.adapter.phase_payload(view.phase_key),
                    "canonical_task_ids": view.canonical_task_ids,
                    "task_catalogue_digest": view.task_catalogue_digest,
                    "active_plan_id": view.active_plan_id,
                    "phase_plan_epoch": view.phase_plan_epoch,
                    "committed_task_ids": view.committed_task_ids,
                    "running_task_ids": view.running_task_ids,
                    "completed_task_ids": view.completed_task_ids,
                    "registered_window_keys": [
                        self.adapter.window_payload(item) for item in view.registered_window_keys
                    ],
                }
            )
        plans = []
        for plan_id in sorted(self._plans):
            view = self.adapter.plan_view(self._plans[plan_id])
            plans.append(
                {
                    "plan_id": view.plan_id,
                    "window_key": self.adapter.window_payload(view.window_key),
                    "version": view.version,
                    "status": view.status,
                    "supersedes_plan_ids": view.supersedes_plan_ids,
                    "commit_index": view.commit_index,
                    "committed_task_ids": view.committed_task_ids,
                    "remaining_task_ids": view.remaining_task_ids,
                    "created_at_ns": view.created_at_ns,
                    "activated_at_ns": view.activated_at_ns,
                    "completed_at_ns": view.completed_at_ns,
                    "plan_digest": view.plan_digest,
                }
            )
        payload = {"records": records, "plans": plans}
        if self._supersession_records or self._duplicate_execution_count:
            payload["supersession_records"] = self._supersession_records
            payload["duplicate_execution_count"] = self._duplicate_execution_count
        return payload

    def digest(self) -> str:
        return stable_digest(self.stable_payload())
