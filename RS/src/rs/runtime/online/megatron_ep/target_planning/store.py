from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from rs.runtime.guards import InvariantFailure, RouterSenseInvariantError

from .contracts import TargetLayerPreparedJointPlan, TargetPlanKey, TargetPlanTerminalRecord


@dataclass
class _StoredTargetPlan:
    plan: TargetLayerPreparedJointPlan
    created_at_ns: int = field(default_factory=time.perf_counter_ns)


class TargetPlanStore:
    def __init__(self) -> None:
        self._plans: dict[tuple[str, int, str, str], _StoredTargetPlan] = {}
        self._terminal: dict[tuple[str, int, str, str], TargetPlanTerminalRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(key: TargetPlanKey) -> tuple[str, int, str, str]:
        return (str(key.run_id), int(key.forward_epoch), str(key.microbatch_id), str(key.target_layer_id))

    def put(self, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> None:
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
            self._plans[skey] = _StoredTargetPlan(plan=plan)

    def peek(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan | None:
        with self._lock:
            item = self._plans.get(self._key(key))
            if item is None:
                return None
            return item.plan

    def consume_once(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan:
        skey = self._key(key)
        with self._lock:
            item = self._plans.get(skey)
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
            self._plans.pop(skey, None)
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=str(plan.logical_plan_digest),
                final_status="CONSUMED",
                execution_origin="consumed",
                terminal_at_ns=int(time.perf_counter_ns()),
            )
            return plan

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
            return len(expired)

    def cancel(self, key: TargetPlanKey) -> None:
        self._finalize(key, final_status="CANCELLED", execution_origin="cancelled")

    def cleanup_epoch(self, *, run_id: str, forward_epoch: int, microbatch_id: str) -> None:
        with self._lock:
            doomed = [
                skey
                for skey in self._plans
                if skey[0] == str(run_id) and int(skey[1]) == int(forward_epoch) and skey[2] == str(microbatch_id)
            ]
        for skey in doomed:
            self.cancel(TargetPlanKey(run_id=skey[0], forward_epoch=skey[1], microbatch_id=skey[2], target_layer_id=skey[3]))

    def shutdown(self) -> None:
        with self._lock:
            doomed = list(self._plans)
        for skey in doomed:
            self.cancel(TargetPlanKey(run_id=skey[0], forward_epoch=skey[1], microbatch_id=skey[2], target_layer_id=skey[3]))

    def get_terminal_record(self, key: TargetPlanKey) -> TargetPlanTerminalRecord | None:
        with self._lock:
            return self._terminal.get(self._key(key))

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
                        "status": "AVAILABLE",
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
                if skey in self._terminal:
                    return
                return
            self._terminal[skey] = TargetPlanTerminalRecord(
                key=key,
                plan_digest=str(item.plan.logical_plan_digest),
                final_status=str(final_status),
                execution_origin=str(execution_origin),
                terminal_at_ns=int(time.perf_counter_ns()),
            )
