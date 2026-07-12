from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rs.runtime.guards import InvariantFailure, RouterSenseInvariantError

from .contracts import TargetLayerPreparedJointPlan, TargetPlanKey


@dataclass
class _StoredTargetPlan:
    plan: TargetLayerPreparedJointPlan
    consumed: bool = False
    cancelled: bool = False
    invalidated: bool = False
    expired: bool = False
    created_at_ns: int = field(default_factory=time.perf_counter_ns)


class TargetPlanStore:
    def __init__(self) -> None:
        self._plans: dict[tuple[str, int, str, str], _StoredTargetPlan] = {}

    @staticmethod
    def _key(key: TargetPlanKey) -> tuple[str, int, str, str]:
        return (str(key.run_id), int(key.forward_epoch), str(key.microbatch_id), str(key.target_layer_id))

    def put(self, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> None:
        skey = self._key(key)
        self._plans[skey] = _StoredTargetPlan(plan=plan)

    def peek(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan | None:
        item = self._plans.get(self._key(key))
        if item is None or item.cancelled or item.invalidated or item.expired:
            return None
        return item.plan

    def consume_once(self, key: TargetPlanKey) -> TargetLayerPreparedJointPlan:
        item = self._plans.get(self._key(key))
        if item is None:
            raise RouterSenseInvariantError(
                InvariantFailure(
                    error_code="RS-PLANNING-TP-001",
                    stage="target_plan_store",
                    message="target plan missing at consume_once",
                    actual=key.to_dict(),
                )
            )
        if item.cancelled or item.invalidated or item.expired:
            raise RouterSenseInvariantError(
                InvariantFailure(
                    error_code="RS-PLANNING-TP-002",
                    stage="target_plan_store",
                    message="target plan not consumable",
                    actual={"key": key.to_dict(), "cancelled": item.cancelled, "invalidated": item.invalidated, "expired": item.expired},
                )
            )
        if item.consumed:
            raise RouterSenseInvariantError(
                InvariantFailure(
                    error_code="RS-PLANNING-TP-003",
                    stage="target_plan_store",
                    message="target plan double consume",
                    actual=key.to_dict(),
                )
            )
        item.consumed = True
        return item.plan

    def invalidate(self, key: TargetPlanKey) -> None:
        item = self._plans.get(self._key(key))
        if item is not None:
            item.invalidated = True

    def expire(self, *, older_than_ns: int) -> int:
        now = time.perf_counter_ns()
        count = 0
        for item in self._plans.values():
            if not item.expired and (now - item.created_at_ns) >= int(older_than_ns):
                item.expired = True
                count += 1
        return count

    def cancel(self, key: TargetPlanKey) -> None:
        item = self._plans.get(self._key(key))
        if item is not None:
            item.cancelled = True

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
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
                    "consumed": bool(item.consumed),
                    "cancelled": bool(item.cancelled),
                    "invalidated": bool(item.invalidated),
                    "expired": bool(item.expired),
                }
            )
        return rows

