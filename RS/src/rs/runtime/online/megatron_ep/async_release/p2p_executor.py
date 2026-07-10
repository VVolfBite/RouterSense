"""Experimental AR1 P2P executor contract.

This module defines deterministic peer-op ordering and a fake-backend execution
path. Real collectives remain disabled unless explicit flags are enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import AsyncReleaseExecutionPlan


@dataclass(frozen=True)
class P2POp:
    op_kind: str
    peer_rank: int
    task_id: str
    byte_count: int
    global_order_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncReleaseP2PExecutorConfig:
    enabled: bool = False
    allow_real_collectives: bool = False
    backend: str = "p2p_experimental"
    timeout_ms: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsyncReleaseP2PExecutor:
    def __init__(self, *, config: AsyncReleaseP2PExecutorConfig) -> None:
        self.config = config

    def ordered_ops(self, plan: AsyncReleaseExecutionPlan) -> tuple[P2POp, ...]:
        ops: list[P2POp] = []
        for task in sorted(plan.phase_tasks, key=lambda item: (int(item.get("global_order_index", 0)), str(item["task_id"]))):
            ops.append(
                P2POp(
                    op_kind="recv",
                    peer_rank=int(task.get("dst_rank", -1)),
                    task_id=str(task["task_id"]),
                    byte_count=int(task.get("byte_count", 0)),
                    global_order_index=int(task.get("global_order_index", 0)),
                )
            )
            ops.append(
                P2POp(
                    op_kind="send",
                    peer_rank=int(task.get("dst_rank", -1)),
                    task_id=str(task["task_id"]),
                    byte_count=int(task.get("byte_count", 0)),
                    global_order_index=int(task.get("global_order_index", 0)),
                )
            )
        return tuple(ops)

    def execute(self, plan: AsyncReleaseExecutionPlan) -> dict[str, Any]:
        ops = self.ordered_ops(plan)
        return {
            "enabled": bool(self.config.enabled),
            "allow_real_collectives": bool(self.config.allow_real_collectives),
            "backend": str(self.config.backend),
            "real_collectives_executed": False,
            "fallback_to_phase_sync": True,
            "fallback_reason": "p2p_executor_experimental_not_validated"
            if self.config.enabled
            else "p2p_executor_disabled",
            "ordered_ops": [op.to_dict() for op in ops],
            "op_count": len(ops),
        }


__all__ = ["AsyncReleaseP2PExecutor", "AsyncReleaseP2PExecutorConfig", "P2POp"]
