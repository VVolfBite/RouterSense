"""Backend-owned logical rank state and deterministic transition history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rs_sim.backend.core.errors import BackendContractError, IllegalTransitionError
from rs_sim.backend.core.internal import RankState
from rs_sim.backend.core.util import require_time_ns


@dataclass(frozen=True, slots=True)
class RankTransition:
    at_ns: int
    state: RankState
    phase_key: Any
    reason: str


@dataclass(slots=True)
class RankActor:
    """One logical rank; simulation time remains owned by the Kernel."""

    rank_id: int
    node_id: int | None = None
    state: RankState = RankState.WAIT_DISPATCH
    last_transition_at_ns: int = 0
    history: list[RankTransition] = field(default_factory=list)

    def transition(
        self,
        *,
        state: RankState,
        phase_key: Any,
        at_ns: int,
        reason: str,
    ) -> None:
        at_ns = require_time_ns(at_ns, field="rank_transition.at_ns")
        if at_ns < self.last_transition_at_ns:
            raise IllegalTransitionError("rank transition moved backward in time")
        if not reason:
            raise BackendContractError("rank transition reason must be non-empty")
        self.state = state
        self.last_transition_at_ns = at_ns
        self.history.append(
            RankTransition(
                at_ns=at_ns,
                state=state,
                phase_key=phase_key,
                reason=reason,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "rank_id": self.rank_id,
            "node_id": self.node_id,
            "state": self.state.value,
            "last_transition_at_ns": self.last_transition_at_ns,
            "history": [
                {
                    "at_ns": row.at_ns,
                    "state": row.state.value,
                    "phase_key": row.phase_key,
                    "reason": row.reason,
                }
                for row in self.history
            ],
        }
