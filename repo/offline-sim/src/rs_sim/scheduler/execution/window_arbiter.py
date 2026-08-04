from __future__ import annotations

"""Structural port for future window-level ORDER_ONLY arbitration.

This module freezes only the boundary between independent per-PhaseKey
Authorities and a future window arbiter.  It intentionally provides no
scheduling policy implementation and never creates mixed-phase physical
batches.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rs_sim.contracts.schema import AuthorityStamp

from rs_sim.scheduler.stable import canonical_data, stable_digest


def _domain_digest(domain: str, payload: Any) -> str:
    return stable_digest({"domain": str(domain), "payload": payload})


def _ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be tuple")
    normalized = tuple(str(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} contains an empty ID")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class PhaseFrontier:
    """One active phase's READY_UNCOMMITTED frontier in active-plan order."""

    phase_key: Any
    authority_stamp: AuthorityStamp
    ready_task_ids: tuple[str, ...]
    frontier_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority_stamp, AuthorityStamp):
            raise TypeError("authority_stamp must be AuthorityStamp")
        object.__setattr__(self, "ready_task_ids", _ids(self.ready_task_ids, "ready_task_ids"))
        if not isinstance(self.frontier_digest, str) or not self.frontier_digest:
            raise ValueError("frontier_digest must be non-empty")
        expected = _domain_digest(
            "SCHEDULER_PHASE_FRONTIER",
            {
                "phase_key": canonical_data(self.phase_key),
                "authority_stamp": self.authority_stamp,
                "ready_task_ids": self.ready_task_ids,
            },
        )
        if self.frontier_digest != expected:
            raise ValueError("frontier_digest does not match frontier semantics")

    @classmethod
    def build(
        cls,
        *,
        phase_key: Any,
        authority_stamp: AuthorityStamp,
        ready_task_ids: tuple[str, ...],
    ) -> "PhaseFrontier":
        normalized = _ids(ready_task_ids, "ready_task_ids")
        digest = _domain_digest(
            "SCHEDULER_PHASE_FRONTIER",
            {
                "phase_key": canonical_data(phase_key),
                "authority_stamp": authority_stamp,
                "ready_task_ids": normalized,
            },
        )
        return cls(
            phase_key=phase_key,
            authority_stamp=authority_stamp,
            ready_task_ids=normalized,
            frontier_digest=digest,
        )


@dataclass(frozen=True, slots=True)
class WindowArbitrationContext:
    window_key: Any
    frontiers: tuple[PhaseFrontier, ...]
    transport_snapshot_digest: str
    observed_at_ns: int
    context_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.frontiers, tuple) or not all(
            isinstance(frontier, PhaseFrontier) for frontier in self.frontiers
        ):
            raise TypeError("frontiers must be tuple[PhaseFrontier, ...]")
        phase_tokens = tuple(frontier.authority_stamp.phase_token for frontier in self.frontiers)
        if len(set(phase_tokens)) != len(phase_tokens):
            raise ValueError("window context contains duplicate phase authorities")
        all_tasks = tuple(
            task_id for frontier in self.frontiers for task_id in frontier.ready_task_ids
        )
        if len(set(all_tasks)) != len(all_tasks):
            raise ValueError("one canonical task appears in multiple phase frontiers")
        if not isinstance(self.transport_snapshot_digest, str) or not self.transport_snapshot_digest:
            raise ValueError("transport_snapshot_digest must be non-empty")
        if not isinstance(self.observed_at_ns, int) or isinstance(self.observed_at_ns, bool) or self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be a non-negative int")
        if not isinstance(self.context_digest, str) or not self.context_digest:
            raise ValueError("context_digest must be non-empty")
        expected = _domain_digest(
            "SCHEDULER_WINDOW_ARBITRATION_CONTEXT",
            {
                "window_key": canonical_data(self.window_key),
                "frontier_digests": tuple(frontier.frontier_digest for frontier in self.frontiers),
                "transport_snapshot_digest": self.transport_snapshot_digest,
                "observed_at_ns": self.observed_at_ns,
            },
        )
        if self.context_digest != expected:
            raise ValueError("context_digest does not match arbitration context")

    @classmethod
    def build(
        cls,
        *,
        window_key: Any,
        frontiers: tuple[PhaseFrontier, ...],
        transport_snapshot_digest: str,
        observed_at_ns: int,
    ) -> "WindowArbitrationContext":
        digest = _domain_digest(
            "SCHEDULER_WINDOW_ARBITRATION_CONTEXT",
            {
                "window_key": canonical_data(window_key),
                "frontier_digests": tuple(frontier.frontier_digest for frontier in frontiers),
                "transport_snapshot_digest": str(transport_snapshot_digest),
                "observed_at_ns": int(observed_at_ns),
            },
        )
        return cls(
            window_key=window_key,
            frontiers=frontiers,
            transport_snapshot_digest=str(transport_snapshot_digest),
            observed_at_ns=int(observed_at_ns),
            context_digest=digest,
        )


@dataclass(frozen=True, slots=True)
class WindowArbitrationDecision:
    """Select tasks from exactly one phase frontier for the next compile pass."""

    context_digest: str
    selected_phase_token: str | None
    selected_task_ids: tuple[str, ...]
    reason: str
    decision_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.context_digest, str) or not self.context_digest:
            raise ValueError("context_digest must be non-empty")
        if self.selected_phase_token is not None and not self.selected_phase_token:
            raise ValueError("selected_phase_token must be non-empty when present")
        object.__setattr__(self, "selected_task_ids", _ids(self.selected_task_ids, "selected_task_ids"))
        if self.selected_task_ids and self.selected_phase_token is None:
            raise ValueError("selected tasks require a selected phase")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        if not isinstance(self.decision_digest, str) or not self.decision_digest:
            raise ValueError("decision_digest must be non-empty")


@runtime_checkable
class WindowArbiter(Protocol):
    """Policy port for an explicitly supplied window-arbitration implementation."""

    def select(self, context: WindowArbitrationContext) -> WindowArbitrationDecision: ...


def validate_window_decision(
    context: WindowArbitrationContext,
    decision: WindowArbitrationDecision,
) -> None:
    """Fail closed unless a decision selects a prefix of one current frontier."""

    if decision.context_digest != context.context_digest:
        raise ValueError("window decision was produced for another context")
    by_phase = {
        frontier.authority_stamp.phase_token: frontier
        for frontier in context.frontiers
    }
    if decision.selected_phase_token is None:
        if decision.selected_task_ids:
            raise ValueError("empty phase selection cannot contain tasks")
    else:
        try:
            frontier = by_phase[decision.selected_phase_token]
        except KeyError as exc:
            raise ValueError("window decision selected an unknown phase authority") from exc
        selected = decision.selected_task_ids
        if selected != frontier.ready_task_ids[: len(selected)]:
            raise ValueError("window decision must select a prefix of one phase frontier")
    expected = _domain_digest(
        "SCHEDULER_WINDOW_ARBITRATION_DECISION",
        {
            "context_digest": decision.context_digest,
            "selected_phase_token": decision.selected_phase_token,
            "selected_task_ids": decision.selected_task_ids,
            "reason": decision.reason,
        },
    )
    if decision.decision_digest != expected:
        raise ValueError("decision_digest does not match decision semantics")


def make_window_decision(
    context: WindowArbitrationContext,
    *,
    selected_phase_token: str | None,
    selected_task_ids: tuple[str, ...],
    reason: str,
) -> WindowArbitrationDecision:
    normalized = _ids(selected_task_ids, "selected_task_ids")
    digest = _domain_digest(
        "SCHEDULER_WINDOW_ARBITRATION_DECISION",
        {
            "context_digest": context.context_digest,
            "selected_phase_token": selected_phase_token,
            "selected_task_ids": normalized,
            "reason": str(reason),
        },
    )
    decision = WindowArbitrationDecision(
        context_digest=context.context_digest,
        selected_phase_token=selected_phase_token,
        selected_task_ids=normalized,
        reason=str(reason),
        decision_digest=digest,
    )
    validate_window_decision(context, decision)
    return decision


@dataclass(frozen=True, slots=True)
class PrefixWindowArbiter:
    """Deterministic ORDER_ONLY arbiter over independent phase frontiers.

    The arbiter may prioritize phase authorities and choose a bounded prefix,
    but it never combines tasks from two catalogues in one decision.
    """

    max_prefix_tasks: int = 1
    preferred_phase_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.max_prefix_tasks, int) or isinstance(self.max_prefix_tasks, bool) or self.max_prefix_tasks <= 0:
            raise ValueError("max_prefix_tasks must be a positive int")
        object.__setattr__(self, "preferred_phase_tokens", _ids(self.preferred_phase_tokens, "preferred_phase_tokens"))

    def select(self, context: WindowArbitrationContext) -> WindowArbitrationDecision:
        nonempty = [frontier for frontier in context.frontiers if frontier.ready_task_ids]
        if not nonempty:
            return make_window_decision(
                context, selected_phase_token=None, selected_task_ids=(), reason="NO_READY_FRONTIER"
            )
        preferred_rank = {token: index for index, token in enumerate(self.preferred_phase_tokens)}
        chosen = min(
            nonempty,
            key=lambda frontier: (
                preferred_rank.get(frontier.authority_stamp.phase_token, len(preferred_rank)),
                frontier.authority_stamp.phase_token,
                frontier.frontier_digest,
            ),
        )
        selected = chosen.ready_task_ids[: self.max_prefix_tasks]
        return make_window_decision(
            context,
            selected_phase_token=chosen.authority_stamp.phase_token,
            selected_task_ids=selected,
            reason="ORDER_ONLY_SINGLE_PHASE_PREFIX",
        )


@dataclass(frozen=True, slots=True)
class ReleaseFrontierWindowArbiter:
    """Approved WINDOW_JOINT ORDER_ONLY arbiter over live phase frontiers.

    ``preferred_task_ids`` is the deterministic ReleaseFrontier window order.
    The arbiter projects that order onto independent PhaseAuthorities and emits
    only a prefix of one phase frontier, preserving the frozen single-phase
    ``TransferBatch`` contract while allowing later phases to overtake an
    earlier phase when their release-frontier priority is higher.
    """

    preferred_task_ids: tuple[str, ...]
    max_prefix_tasks: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "preferred_task_ids", _ids(self.preferred_task_ids, "preferred_task_ids")
        )
        if (
            not isinstance(self.max_prefix_tasks, int)
            or isinstance(self.max_prefix_tasks, bool)
            or self.max_prefix_tasks <= 0
        ):
            raise ValueError("max_prefix_tasks must be a positive int")

    def select(self, context: WindowArbitrationContext) -> WindowArbitrationDecision:
        nonempty = tuple(frontier for frontier in context.frontiers if frontier.ready_task_ids)
        if not nonempty:
            return make_window_decision(
                context,
                selected_phase_token=None,
                selected_task_ids=(),
                reason="NO_READY_FRONTIER",
            )
        priority = {task_id: index for index, task_id in enumerate(self.preferred_task_ids)}
        fallback = len(priority)

        def task_priority(task_id: str) -> tuple[int, str]:
            return (priority.get(task_id, fallback), task_id)

        chosen = min(
            nonempty,
            key=lambda frontier: (
                task_priority(frontier.ready_task_ids[0]),
                frontier.authority_stamp.phase_token,
                frontier.frontier_digest,
            ),
        )
        competing_priorities = [
            task_priority(task_id)
            for frontier in nonempty
            if frontier is not chosen
            for task_id in frontier.ready_task_ids
        ]
        competing_best = min(competing_priorities) if competing_priorities else None
        selected: list[str] = []
        for task_id in chosen.ready_task_ids:
            if len(selected) >= self.max_prefix_tasks:
                break
            if competing_best is not None and selected and task_priority(task_id) > competing_best:
                break
            selected.append(task_id)
        return make_window_decision(
            context,
            selected_phase_token=chosen.authority_stamp.phase_token,
            selected_task_ids=tuple(selected),
            reason="RELEASEFRONTIER_WINDOW_ORDER_SINGLE_PHASE_PREFIX",
        )


@dataclass(frozen=True, slots=True)
class ReleaseFrontierWaveArbiter:
    """Project authoritative RSCF task-boundary waves onto phase authorities.

    Residual-flow service segments remain algorithm evidence.  The formal
    executor preserves canonical task boundaries, but it must not discard the
    matching-wave grouping and silently repack the plan as an unrelated flat
    order.  Each decision still selects one phase only, satisfying the frozen
    ``TransferBatch`` contract; another phase from the same conflict-free
    wave may be selected by the next stabilization pass at the same timestamp.
    """

    preferred_waves: tuple[tuple[str, ...], ...]
    preferred_task_ids: tuple[str, ...]
    max_prefix_tasks: int = 1

    def __post_init__(self) -> None:
        waves = tuple(_ids(tuple(wave), "preferred_wave") for wave in self.preferred_waves)
        if any(not wave for wave in waves):
            raise ValueError("preferred_waves cannot contain an empty wave")
        flattened = tuple(task_id for wave in waves for task_id in wave)
        ordered = _ids(self.preferred_task_ids, "preferred_task_ids")
        if flattened != ordered:
            raise ValueError("preferred_waves flattening must equal preferred_task_ids")
        if (
            not isinstance(self.max_prefix_tasks, int)
            or isinstance(self.max_prefix_tasks, bool)
            or self.max_prefix_tasks <= 0
        ):
            raise ValueError("max_prefix_tasks must be a positive int")
        object.__setattr__(self, "preferred_waves", waves)
        object.__setattr__(self, "preferred_task_ids", ordered)

    def select(self, context: WindowArbitrationContext) -> WindowArbitrationDecision:
        nonempty = tuple(frontier for frontier in context.frontiers if frontier.ready_task_ids)
        if not nonempty:
            return make_window_decision(
                context, selected_phase_token=None, selected_task_ids=(), reason="NO_READY_FRONTIER"
            )
        wave_rank: dict[str, tuple[int, int]] = {}
        for wave_index, wave in enumerate(self.preferred_waves):
            for within_wave, task_id in enumerate(wave):
                wave_rank[task_id] = (wave_index, within_wave)
        fallback_wave = len(self.preferred_waves)

        def rank(task_id: str) -> tuple[int, int, str]:
            wave_index, within_wave = wave_rank.get(task_id, (fallback_wave, 0))
            return (wave_index, within_wave, task_id)

        chosen = min(
            nonempty,
            key=lambda frontier: (
                rank(frontier.ready_task_ids[0]),
                frontier.authority_stamp.phase_token,
                frontier.frontier_digest,
            ),
        )
        first_wave = rank(chosen.ready_task_ids[0])[0]
        selected: list[str] = []
        for task_id in chosen.ready_task_ids:
            if len(selected) >= self.max_prefix_tasks:
                break
            if rank(task_id)[0] != first_wave:
                break
            selected.append(task_id)
        return make_window_decision(
            context,
            selected_phase_token=chosen.authority_stamp.phase_token,
            selected_task_ids=tuple(selected),
            reason="RELEASEFRONTIER_TASK_BOUNDARY_WAVE_SINGLE_PHASE_PREFIX",
        )


__all__ = [
    "PhaseFrontier",
    "PrefixWindowArbiter",
    "ReleaseFrontierWaveArbiter",
    "ReleaseFrontierWindowArbiter",
    "WindowArbiter",
    "WindowArbitrationContext",
    "WindowArbitrationDecision",
    "make_window_decision",
    "validate_window_decision",
]
