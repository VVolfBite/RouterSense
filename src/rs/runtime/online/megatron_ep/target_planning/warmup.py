from __future__ import annotations

"""Process-level cold-start preparation for deployable P012 planning kernels."""

import threading
import time
from typing import Any

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)
from rs.planning import PlannerRegistry
from rs.planning.validation import validate_window_plan_for_request

_COMPILED_KERNEL_WARMUP_LOCK = threading.Lock()
_COMPILED_KERNEL_WARMUP_STATE: dict[str, Any] = {
    "status": "not_started",
    "duration_us": 0.0,
    "planner_ids": (),
    "error": None,
}


def _compiled_kernel_warmup_request() -> PlanningRequest:
    """Return a small production-shaped request used only for cold-start compilation."""

    return PlanningRequest(
        identity=PlanningIdentity(
            request_id="compiled-kernel-warmup",
            run_id="runtime-attach",
            forward_id="warmup",
            window_id="p012",
            source_layer_id="0",
            target_layer_id="1",
        ),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 3, 1), (2, 0, 1), (1, 2, 0)),
            p1_return_rows=((0, 2, 1), (3, 0, 2), (1, 1, 0)),
        ),
        prediction_hint=PredictionHint(
            predictor_id="compiled_kernel_warmup",
            hint_type="traffic_matrix",
            target_dispatch_rows=((0, 2, 1), (1, 0, 3), (2, 1, 0)),
            confidence=1.0,
            source_layer_id="0",
            target_layer_id="1",
        ),
        topology=PlanningTopology(world_size=3),
        constraints=PlanningConstraints(
            bucket_rows=0,
            max_waves=128,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
        planning_track="runtime_lookahead",
        p2_semantics="advisory_hint",
    )


def ensure_compiled_kernel_warmup() -> dict[str, Any]:
    """Warm production P012 planning and truth-binding kernels once per process.

    Source archives intentionally exclude machine-specific Numba cache files. The
    target planner invokes this synchronously during runtime attach, before it
    accepts asynchronous work. A failed warmup is sticky and fails attach closed.
    """

    with _COMPILED_KERNEL_WARMUP_LOCK:
        status = str(_COMPILED_KERNEL_WARMUP_STATE["status"])
        if status == "passed":
            return dict(_COMPILED_KERNEL_WARMUP_STATE)
        if status == "failed":
            raise RuntimeError(
                "compiled planning kernel warmup previously failed: "
                f"{_COMPILED_KERNEL_WARMUP_STATE['error']}"
            )

        started_ns = time.perf_counter_ns()
        planner_ids = (
            "future:p012:joint:event:rscf",
            "future:p012:joint:global:rscf",
        )
        _COMPILED_KERNEL_WARMUP_STATE.update(
            status="running", planner_ids=planner_ids, error=None
        )
        try:
            request = _compiled_kernel_warmup_request()
            for planner_id in planner_ids:
                planner = PlannerRegistry.create(planner_id, None, usage="runtime")
                plan = planner.plan(request)
                validate_window_plan_for_request(plan, request)
            from rs.scheduling.p012_future._kernel.future import warmup_future_bind_kernel

            warmup_future_bind_kernel()
        except BaseException as exc:
            duration_us = (time.perf_counter_ns() - started_ns) / 1000.0
            _COMPILED_KERNEL_WARMUP_STATE.update(
                status="failed",
                duration_us=float(duration_us),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(
                "compiled planning kernel warmup failed before target planner start"
            ) from exc

        duration_us = (time.perf_counter_ns() - started_ns) / 1000.0
        _COMPILED_KERNEL_WARMUP_STATE.update(
            status="passed", duration_us=float(duration_us), error=None
        )
        return dict(_COMPILED_KERNEL_WARMUP_STATE)


__all__ = ["ensure_compiled_kernel_warmup"]
