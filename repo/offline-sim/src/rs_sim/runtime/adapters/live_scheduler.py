from __future__ import annotations

"""Composite over per-window scheduler adapters.

Each phase has exactly one live scheduling authority.  All window adapters share
one global PredictionLine, ControlLine, and ExecutionBindingLine.  The
composite only routes public backend observations and transport resource-release callbacks;
it does not own canonical tasks, plans, receiver state, or transport resources.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from rs_sim.contracts.paper_defaults import PAPER_P0_P1_COMPUTE_END_BARRIER
from rs_sim.scheduler import FormalSchedulingRuntimeAdapter
from rs_sim.scheduler.stable import stable_digest, stable_json


def _phase_token(adapter: Any, phase_key: Any) -> str:
    return stable_json(adapter.phase_payload(phase_key))


@dataclass(frozen=True, slots=True)
class CurrentP12TriggerRoute:
    trigger_phase_key: Any
    target_adapter: FormalSchedulingRuntimeAdapter
    p1_source_ready_duration_ns_by_rank: tuple[int, ...]
    planning_window_digest: str
    p0_p1_compute_end_barrier: bool = PAPER_P0_P1_COMPUTE_END_BARRIER

    def __post_init__(self) -> None:
        if not self.p1_source_ready_duration_ns_by_rank:
            raise ValueError("p1_source_ready_duration_ns_by_rank must be non-empty")
        if any(int(value) < 0 for value in self.p1_source_ready_duration_ns_by_rank):
            raise ValueError("P1 source-ready durations must be non-negative")
        if not isinstance(self.planning_window_digest, str) or not self.planning_window_digest:
            raise ValueError("planning_window_digest must be non-empty")
        if not isinstance(self.p0_p1_compute_end_barrier, bool):
            raise ValueError("p0_p1_compute_end_barrier must be bool")

    def earliest_p1_source_ready_ns(self, *, backend: Any | None, fallback_at_ns: int) -> int:
        """Return the first causal P1 plan-consumption deadline.

        The deadline is anchored at the actual per-rank P0 release time, not at
        descriptor closure.  A conservative closure-relative fallback is kept
        only for adapters used without an attached backend.
        """

        candidates: list[int] = []
        if backend is not None:
            for rank, duration_ns in enumerate(self.p1_source_ready_duration_ns_by_rank):
                released_at = backend.rank_release_at(
                    phase_key=self.trigger_phase_key, rank_id=int(rank)
                )
                if released_at is not None:
                    candidates.append(int(released_at) + int(duration_ns))
        if self.p0_p1_compute_end_barrier:
            if len(candidates) == len(self.p1_source_ready_duration_ns_by_rank):
                return max(candidates)
            return int(fallback_at_ns) + max(
                int(value) for value in self.p1_source_ready_duration_ns_by_rank
            )
        if candidates:
            return min(candidates)
        return int(fallback_at_ns) + min(
            int(value) for value in self.p1_source_ready_duration_ns_by_rank
        )


@dataclass(frozen=True, slots=True)
class CompositeFormalSchedulingMetrics:
    adapter_count: int
    raw_observation_count: int
    coalesced_batch_count: int
    coalesced_observation_savings: int
    pipeline_job_count: int
    activated_plan_count: int
    stale_activation_count: int
    global_seal_count: int
    phase_plan_counts: tuple[tuple[str, int], ...]
    line_metrics: tuple[Any, ...]
    adapter_metric_digests: tuple[str, ...]
    current_p12_template_evidence: tuple[Any, ...]
    frontier_replan_count: int
    metrics_digest: str


class CompositeFormalSchedulingRuntimeAdapter:
    """Route one formal run across non-overlapping live window authorities."""

    def __init__(
        self,
        *,
        adapters: tuple[FormalSchedulingRuntimeAdapter, ...],
        scheduling_stack: Any,
        current_p12_trigger_routes: tuple[CurrentP12TriggerRoute, ...] = (),
    ) -> None:
        if not adapters:
            raise ValueError("at least one formal scheduler adapter is required")
        self.adapters = tuple(adapters)
        self.stack = scheduling_stack
        self.backend: Any | None = None
        self.transport: Any | None = None
        self._adapter_by_phase_token: dict[str, FormalSchedulingRuntimeAdapter] = {}
        for runtime in self.adapters:
            for phase_key in runtime.session.phase_keys:
                token = _phase_token(self.stack.adapter, phase_key)
                if token in self._adapter_by_phase_token:
                    raise ValueError("phase has more than one live scheduling authority")
                self._adapter_by_phase_token[token] = runtime
        self._current_p12_route_by_trigger_token: dict[str, CurrentP12TriggerRoute] = {}
        for route in current_p12_trigger_routes:
            token = _phase_token(self.stack.adapter, route.trigger_phase_key)
            if token in self._current_p12_route_by_trigger_token:
                raise ValueError("P0 phase has more than one Current P12 trigger route")
            if route.target_adapter not in self.adapters:
                raise ValueError("Current P12 trigger route references an unknown adapter")
            self._current_p12_route_by_trigger_token[token] = route

    def _phase_from_payload(self, payload: Mapping[str, Any]) -> Any | None:
        phase_key = payload.get("phase_key")
        if phase_key is not None:
            return phase_key
        expectation = payload.get("expectation")
        if expectation is not None:
            return getattr(expectation, "phase_key", None)
        summary = payload.get("summary")
        if summary is not None:
            return getattr(summary, "phase_key", None)
        permit = payload.get("permit")
        task_id = payload.get("task_id")
        if task_id is None and permit is not None:
            task_id = getattr(permit, "task_id", None)
        if task_id is not None:
            return self.stack.catalogue.view(str(task_id)).phase_key
        return None

    def _adapter_for_phase(self, phase_key: Any) -> FormalSchedulingRuntimeAdapter:
        token = _phase_token(self.stack.adapter, phase_key)
        try:
            return self._adapter_by_phase_token[token]
        except KeyError as exc:
            raise RuntimeError("observation references a phase without live authority") from exc

    def emit(
        self,
        *,
        kind: str,
        at_ns: int,
        payload: Mapping[str, Any],
        hide_until_ns: int | None = None,
    ) -> Any | None:
        phase_key = self._phase_from_payload(payload)
        if phase_key is None:
            # Backend-only audit observations do not need to enter scheduler.
            return None
        result = self._adapter_for_phase(phase_key).emit(
            kind=str(kind),
            at_ns=int(at_ns),
            payload=payload,
            hide_until_ns=hide_until_ns,
        )
        if str(kind) == "PHASE_CLOSURE_SUMMARY_READY":
            route = self._current_p12_route_by_trigger_token.get(
                _phase_token(self.stack.adapter, phase_key)
            )
            if route is not None:
                route.target_adapter.trigger_current_p12(
                    at_ns=int(at_ns),
                    hide_until_ns=route.earliest_p1_source_ready_ns(
                        backend=self.backend, fallback_at_ns=int(at_ns)
                    ),
                    trigger_phase_key=phase_key,
                )
        return result

    def attach_backend(self, backend: Any) -> None:
        self.backend = backend
        for runtime in self.adapters:
            runtime.attach_backend(backend)

    def attach_transport(self, transport: Any) -> None:
        self.transport = transport
        for runtime in self.adapters:
            runtime.attach_transport(transport)

    def notify_transport_resource_release(self, phase_key: Any) -> Any:
        return self._adapter_for_phase(phase_key).notify_transport_resource_release(
            phase_key
        )

    @property
    def permits_by_task_id(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for runtime in self.adapters:
            for task_id, permit in runtime.permits_by_task_id.items():
                existing = result.get(task_id)
                if existing is not None and existing != permit:
                    raise RuntimeError("conflicting permit across scheduler windows")
                result[task_id] = permit
        return result

    def plan_count(self, phase_key: Any) -> int:
        return self._adapter_for_phase(phase_key).plan_count(phase_key)

    def metrics(self) -> CompositeFormalSchedulingMetrics:
        rows = tuple(runtime.metrics() for runtime in self.adapters)
        shared_line_metrics = self.adapters[0].lines.metrics()
        if any(runtime.lines is not self.adapters[0].lines for runtime in self.adapters):
            raise RuntimeError("formal scheduler windows do not share one global three-line service")
        phase_plan_counts = tuple(
            sorted(
                item
                for row in rows
                for item in row.phase_plan_counts
            )
        )
        payload = {
            "adapter_count": len(rows),
            "raw_observation_count": sum(row.raw_observation_count for row in rows),
            "coalesced_batch_count": sum(row.coalesced_batch_count for row in rows),
            "coalesced_observation_savings": sum(
                row.coalesced_observation_savings for row in rows
            ),
            "pipeline_job_count": sum(row.pipeline_job_count for row in rows),
            "activated_plan_count": sum(row.activated_plan_count for row in rows),
            "stale_activation_count": sum(row.stale_activation_count for row in rows),
            "global_seal_count": sum(row.global_seal_count for row in rows),
            "phase_plan_counts": phase_plan_counts,
            "line_metrics": shared_line_metrics,
            "adapter_metric_digests": tuple(row.metrics_digest for row in rows),
            "current_p12_template_evidence": tuple(
                item
                for row in rows
                for item in row.current_p12_template_evidence
            ),
            "frontier_replan_count": sum(row.frontier_replan_count for row in rows),
        }
        return CompositeFormalSchedulingMetrics(
            **payload,
            metrics_digest=stable_digest(payload),
        )


__all__ = [
    "CompositeFormalSchedulingMetrics",
    "CurrentP12TriggerRoute",
    "CompositeFormalSchedulingRuntimeAdapter",
]
