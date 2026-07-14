from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from rs.runtime.guards import InvariantFailure, RouterSenseInvariantError

from .contracts import PreparationToken, TargetLayerPreparedJointPlan, TargetPlanKey, TargetPlanStateRecord, TargetPlanTerminalRecord


@dataclass
class _StoredTargetPlan:
    plan: TargetLayerPreparedJointPlan
    state: str = "LOGICAL_READY"
    claim_owner: str = ""
    bound_owner: str = ""
    execution_origin: str = ""
    created_at_ns: int = field(default_factory=time.perf_counter_ns)
    updated_at_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass(frozen=True)
class PublishCurrentResult:
    status: str
    published_plan_digest: str = ""


class TargetPlanStore:
    def __init__(self) -> None:
        self._plans: dict[tuple[str, int, str, str], _StoredTargetPlan] = {}
        self._claimed: dict[tuple[str, int, str, str], _StoredTargetPlan] = {}
        self._terminal: dict[tuple[str, int, str, str], TargetPlanTerminalRecord] = {}
        self._publish_tokens: dict[tuple[str, int, str, str], PreparationToken] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(key: TargetPlanKey) -> tuple[str, int, str, str]:
        return (str(key.run_id), int(key.forward_epoch), str(key.microbatch_id), str(key.target_layer_id))

    @staticmethod
    def _transition_table() -> dict[str, set[str]]:
        return {
            "LOGICAL_READY": {"CLAIMED", "FAILED", "EXPIRED", "CANCELLED", "REJECTED"},
            "CLAIMED": {"BOUND", "FAILED", "EXPIRED", "CANCELLED", "REJECTED"},
            "BOUND": {"EXECUTING", "FAILED", "EXPIRED", "CANCELLED", "REJECTED"},
            "EXECUTING": {"COMPLETED", "FAILED"},
            "COMPLETED": set(),
            "FAILED": set(),
            "EXPIRED": set(),
            "CANCELLED": set(),
            "REJECTED": set(),
            "CONSUMED": set(),
        }

    def _raise_invalid_transition(self, *, key: TargetPlanKey, current_state: str, next_state: str, where: str) -> None:
        raise RouterSenseInvariantError(
            InvariantFailure(
                error_code="RS-PLANNING-TP-STATE",
                stage="target_plan_store",
                message=f"illegal target plan transition at {where}",
                actual={
                    "key": key.to_dict(),
                    "current_state": str(current_state),
                    "next_state": str(next_state),
                    "where": str(where),
                },
            )
        )

    def put(self, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> None:
        self.publish_logical(key, plan)

    def register_expected_publication(self, token: PreparationToken) -> None:
        with self._lock:
            self._publish_tokens[self._key(token.target_key)] = token

    def publish_if_current(
        self,
        *,
        token: PreparationToken,
        plan: TargetLayerPreparedJointPlan,
    ) -> PublishCurrentResult:
        with self._lock:
            skey = self._key(token.target_key)
            current = self._publish_tokens.get(skey)
            if current is None:
                terminal = self._terminal.get(skey)
                if terminal is not None:
                    return PublishCurrentResult(status="TERMINAL", published_plan_digest=str(terminal.plan_digest))
                return PublishCurrentResult(status="STALE_TOKEN")
            if int(current.service_session_id) != int(token.service_session_id):
                return PublishCurrentResult(status="STALE_SESSION")
            if int(current.forward_generation) != int(token.forward_generation):
                return PublishCurrentResult(status="CANCELLED_GENERATION")
            if int(current.publish_sequence) != int(token.publish_sequence):
                return PublishCurrentResult(status="SLOT_MISMATCH")
            if int(current.task_version) != int(token.task_version):
                return PublishCurrentResult(status="STALE_TOKEN")
            existing = self._plans.get(skey)
            if existing is not None:
                if str(existing.plan.logical_plan_digest) == str(plan.logical_plan_digest):
                    self._publish_tokens.pop(skey, None)
                    return PublishCurrentResult(status="ALREADY_PUBLISHED_SAME", published_plan_digest=str(plan.logical_plan_digest))
                return PublishCurrentResult(status="CONFLICTING_PLAN", published_plan_digest=str(existing.plan.logical_plan_digest))
            claimed = self._claimed.get(skey)
            if claimed is not None:
                if str(claimed.plan.logical_plan_digest) == str(plan.logical_plan_digest):
                    self._publish_tokens.pop(skey, None)
                    return PublishCurrentResult(status="ALREADY_PUBLISHED_SAME", published_plan_digest=str(plan.logical_plan_digest))
                return PublishCurrentResult(status="CONFLICTING_PLAN", published_plan_digest=str(claimed.plan.logical_plan_digest))
            self._plans[skey] = _StoredTargetPlan(plan=plan, state="LOGICAL_READY")
            self._publish_tokens.pop(skey, None)
            return PublishCurrentResult(status="PUBLISHED", published_plan_digest=str(plan.logical_plan_digest))

    def publish_logical(self, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> None:
        skey = self._key(key)
        with self._lock:
            terminal = self._terminal.get(skey)
            if terminal is not None:
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-004",
                        stage="target_plan_store",
                        message="target plan cannot be re-put after terminal status",
                        actual={"key": key.to_dict(), "terminal": terminal.to_dict()},
                    )
                )
            existing = self._plans.get(skey)
            if existing is not None:
                if str(existing.plan.logical_plan_digest) == str(plan.logical_plan_digest):
                    return
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-005",
                        stage="target_plan_store",
                        message="target plan overwrite with different digest",
                        actual={
                            "key": key.to_dict(),
                            "existing_digest": str(existing.plan.logical_plan_digest),
                            "new_digest": str(plan.logical_plan_digest),
                        },
                    )
                )
            claimed = self._claimed.get(skey)
            if claimed is not None:
                if str(claimed.plan.logical_plan_digest) == str(plan.logical_plan_digest):
                    return
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-012",
                        stage="target_plan_store",
                        message="target plan overwrite while claimed",
                        actual={
                            "key": key.to_dict(),
                            "existing_digest": str(claimed.plan.logical_plan_digest),
                            "new_digest": str(plan.logical_plan_digest),
                        },
                    )
                )
            self._plans[skey] = _StoredTargetPlan(plan=plan, state="LOGICAL_READY")
            self._publish_tokens.pop(skey, None)

    def peek(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan | None:
        with self._lock:
            item = self._plans.get(self._key(key))
            if item is None:
                return None
            if str(item.state) != "LOGICAL_READY":
                return None
            return item.plan

    def claim_for_reconciliation(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan:
        return self.claim(key, claim_owner="reconciliation")

    def claim(self, key: TargetPlanKey, *, claim_owner: str) -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._plans.pop(skey, None)
            if item is None:
                if skey in self._claimed:
                    raise RouterSenseInvariantError(
                        InvariantFailure(
                            error_code="RS-PLANNING-TP-013",
                            stage="target_plan_store",
                            message="target plan already claimed",
                            actual={"key": key.to_dict()},
                        )
                    )
                terminal = self._terminal.get(skey)
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-014",
                        stage="target_plan_store",
                        message="target plan missing at claim_for_reconciliation",
                        actual={"key": key.to_dict(), "terminal": terminal.to_dict() if terminal is not None else None},
                        )
                    )
            if str(item.state) != "LOGICAL_READY":
                self._raise_invalid_transition(key=key, current_state=str(item.state), next_state="CLAIMED", where="claim")
            self._claimed[skey] = _StoredTargetPlan(
                plan=item.plan,
                state="CLAIMED",
                claim_owner=str(claim_owner),
                bound_owner=str(item.bound_owner),
                execution_origin=str(item.execution_origin),
                created_at_ns=int(item.created_at_ns),
                updated_at_ns=int(time.perf_counter_ns()),
            )
            return item.plan

    def consume_once(self, key: TargetPlanKey, *, execution_origin: str = "consumed") -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._claimed.pop(skey, None)
            if item is None:
                item = self._plans.pop(skey, None)
            if item is None:
                terminal = self._terminal.get(skey)
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-001",
                        stage="target_plan_store",
                        message="target plan missing at consume_once",
                        actual={"key": key.to_dict(), "terminal": terminal.to_dict() if terminal is not None else None},
                    )
                )
            plan = item.plan
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=str(plan.logical_plan_digest),
                final_status="CONSUMED",
                execution_origin=str(execution_origin),
                terminal_at_ns=int(time.perf_counter_ns()),
            )
            self._publish_tokens.pop(skey, None)
            return plan

    def bind(self, key: TargetPlanKey, *, bound_owner: str) -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._claimed.get(skey)
            if item is None:
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-015",
                        stage="target_plan_store",
                        message="target plan missing at bind",
                        actual={"key": key.to_dict()},
                    )
                )
            if str(item.state) != "CLAIMED":
                self._raise_invalid_transition(key=key, current_state=str(item.state), next_state="BOUND", where="bind")
            self._claimed[skey] = _StoredTargetPlan(
                plan=item.plan,
                state="BOUND",
                claim_owner=str(item.claim_owner),
                bound_owner=str(bound_owner),
                execution_origin=str(item.execution_origin),
                created_at_ns=int(item.created_at_ns),
                updated_at_ns=int(time.perf_counter_ns()),
            )
            return item.plan

    def start_execution(self, key: TargetPlanKey, *, execution_origin: str, claim_owner: str = "") -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._claimed.get(skey)
            if item is None:
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-016",
                        stage="target_plan_store",
                        message="target plan missing at start_execution",
                        actual={"key": key.to_dict()},
                    )
                )
            if str(item.state) != "BOUND":
                self._raise_invalid_transition(key=key, current_state=str(item.state), next_state="EXECUTING", where="start_execution")
            self._claimed[skey] = _StoredTargetPlan(
                plan=item.plan,
                state="EXECUTING",
                claim_owner=str(claim_owner or item.claim_owner),
                bound_owner=str(item.bound_owner),
                execution_origin=str(execution_origin),
                created_at_ns=int(item.created_at_ns),
                updated_at_ns=int(time.perf_counter_ns()),
            )
            return item.plan

    def complete(self, key: TargetPlanKey, *, execution_origin: str = "completed") -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._claimed.pop(skey, None)
            if item is None:
                raise RouterSenseInvariantError(
                    InvariantFailure(
                        error_code="RS-PLANNING-TP-017",
                        stage="target_plan_store",
                        message="target plan missing at complete",
                        actual={"key": key.to_dict()},
                    )
                )
            if str(item.state) != "EXECUTING":
                self._claimed[skey] = item
                self._raise_invalid_transition(key=key, current_state=str(item.state), next_state="COMPLETED", where="complete")
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=str(item.plan.logical_plan_digest),
                final_status="COMPLETED",
                execution_origin=str(execution_origin),
                terminal_at_ns=int(time.perf_counter_ns()),
            )
            self._publish_tokens.pop(skey, None)
            return item.plan

    def fail(self, key: TargetPlanKey, *, execution_origin: str = "failed") -> None:
        self._finalize(key, final_status="FAILED", execution_origin=execution_origin)

    def reject(self, key: TargetPlanKey, *, execution_origin: str = "rejected") -> None:
        self._finalize(key, final_status="REJECTED", execution_origin=execution_origin)

    def invalidate(self, key: TargetPlanKey) -> None:
        self._finalize(key, final_status="REJECTED", execution_origin="invalidated")

    def expire_key(self, key: TargetPlanKey, *, execution_origin: str = "expired") -> None:
        self._finalize(key, final_status="EXPIRED", execution_origin=execution_origin)

    def expire(self, *, older_than_ns: int) -> int:
        now = time.perf_counter_ns()
        with self._lock:
            expired = [
                skey
                for skey, item in self._plans.items()
                if (now - int(item.created_at_ns)) >= int(older_than_ns)
            ]
            for skey in expired:
                plan = self._plans.pop(skey)
                self._terminal[skey] = TargetPlanTerminalRecord(
                    key=TargetPlanKey(run_id=skey[0], forward_epoch=skey[1], microbatch_id=skey[2], target_layer_id=skey[3]),
                    plan_digest=str(plan.plan.logical_plan_digest),
                    final_status="EXPIRED",
                    execution_origin="expired",
                    terminal_at_ns=int(now),
                )
                self._publish_tokens.pop(skey, None)
            return len(expired)

    def cancel(self, key: TargetPlanKey) -> None:
        self._finalize(key, final_status="CANCELLED", execution_origin="cancelled")

    def cleanup_epoch(self, *, run_id: str, forward_epoch: int, microbatch_id: str) -> None:
        with self._lock:
            doomed = [
                skey
                for skey in tuple(self._plans) + tuple(self._claimed)
                if skey[0] == str(run_id) and int(skey[1]) == int(forward_epoch) and skey[2] == str(microbatch_id)
            ]
            token_doomed = [
                skey
                for skey in tuple(self._publish_tokens)
                if skey[0] == str(run_id) and int(skey[1]) == int(forward_epoch) and skey[2] == str(microbatch_id)
            ]
            for skey in token_doomed:
                self._publish_tokens.pop(skey, None)
        for skey in doomed:
            self.cancel(TargetPlanKey(run_id=skey[0], forward_epoch=skey[1], microbatch_id=skey[2], target_layer_id=skey[3]))

    def shutdown(self) -> None:
        with self._lock:
            doomed = list(self._plans) + [key for key in self._claimed if key not in self._plans]
        for skey in doomed:
            self.cancel(TargetPlanKey(run_id=skey[0], forward_epoch=skey[1], microbatch_id=skey[2], target_layer_id=skey[3]))

    def close_key_if_unclaimed(
        self,
        key: TargetPlanKey,
        *,
        final_status: str = "EXPIRED",
        execution_origin: str = "too_late_no_effect",
    ) -> None:
        skey = self._key(key)
        with self._lock:
            if skey in self._terminal:
                return
            if skey in self._claimed:
                return
            item = self._plans.pop(skey, None)
            plan_digest = str(item.plan.logical_plan_digest) if item is not None else ""
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=plan_digest,
                final_status=str(final_status),
                execution_origin=str(execution_origin),
                terminal_at_ns=int(time.perf_counter_ns()),
            )
            self._publish_tokens.pop(skey, None)

    def get_terminal_record(self, key: TargetPlanKey) -> TargetPlanTerminalRecord | None:
        with self._lock:
            return self._terminal.get(self._key(key))

    def get_state_record(self, key: TargetPlanKey) -> TargetPlanStateRecord | None:
        skey = self._key(key)
        with self._lock:
            item = self._plans.get(skey)
            if item is None:
                item = self._claimed.get(skey)
            if item is None:
                terminal = self._terminal.get(skey)
                if terminal is None:
                    return None
                return TargetPlanStateRecord(
                    key=key,
                    plan_digest=str(terminal.plan_digest),
                    state=str(terminal.final_status),
                    execution_origin=str(terminal.execution_origin),
                    updated_at_ns=int(terminal.terminal_at_ns),
                )
            return TargetPlanStateRecord(
                key=key,
                plan_digest=str(item.plan.logical_plan_digest),
                state=str(item.state),
                claim_owner=str(item.claim_owner),
                bound_owner=str(item.bound_owner),
                execution_origin=str(item.execution_origin),
                updated_at_ns=int(item.updated_at_ns),
            )

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            for key, item in self._plans.items():
                rows.append(
                    {
                        "key": {
                            "run_id": key[0],
                            "forward_epoch": key[1],
                            "microbatch_id": key[2],
                            "target_layer_id": key[3],
                        },
                        "logical_plan_digest": str(item.plan.logical_plan_digest),
                        "status": str(item.state),
                    }
                )
            for key, item in self._claimed.items():
                rows.append(
                    {
                        "key": {
                            "run_id": key[0],
                            "forward_epoch": key[1],
                            "microbatch_id": key[2],
                            "target_layer_id": key[3],
                        },
                        "logical_plan_digest": str(item.plan.logical_plan_digest),
                        "status": str(item.state),
                        "claim_owner": str(item.claim_owner),
                        "bound_owner": str(item.bound_owner),
                        "execution_origin": str(item.execution_origin),
                    }
                )
            for record in self._terminal.values():
                rows.append({"key": record.key.to_dict(), "logical_plan_digest": record.plan_digest, "status": record.final_status, "execution_origin": record.execution_origin})
        return rows

    def _finalize(self, key: TargetPlanKey, *, final_status: str, execution_origin: str) -> None:
        skey = self._key(key)
        with self._lock:
            item = self._plans.pop(skey, None)
            if item is None:
                item = self._claimed.pop(skey, None)
            if item is None:
                if skey in self._terminal:
                    return
                return
            self._publish_tokens.pop(skey, None)
            allowed = self._transition_table().get(str(item.state), set())
            if str(final_status) not in allowed:
                if str(item.state) in {"FAILED", "EXPIRED", "CANCELLED", "REJECTED", "COMPLETED", "CONSUMED"}:
                    return
                if item.state in {"LOGICAL_READY", "CLAIMED", "BOUND", "EXECUTING"}:
                    self._raise_invalid_transition(key=key, current_state=str(item.state), next_state=str(final_status), where="finalize")
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=str(item.plan.logical_plan_digest),
                final_status=str(final_status),
                execution_origin=str(execution_origin),
                terminal_at_ns=int(time.perf_counter_ns()),
            )
