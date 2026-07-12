from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rs.scheduling.unified_interface import PolicyOptions, build_policy, build_request_from_replay_window
from rs.scheduling.validation import stable_hash

from .contracts import MatrixRows, TargetLayerPreparedJointPlan, TargetPlanKey
from .predictor import SharedTwoHorizonPredictor, TwoHorizonPredictionBundle
from .store import TargetPlanStore


@dataclass(frozen=True)
class _TargetReplayWindow:
    fixture_id: str
    window_id: str
    layer_id: int
    p0_truth_rows: MatrixRows
    p1_truth_rows: MatrixRows
    p2_truth_rows: MatrixRows
    group_size: int


@dataclass(frozen=True)
class TargetLayerPlanningRequest:
    run_id: str
    forward_epoch: int
    microbatch_id: str
    source_layer_id: str
    target_layer_id: str
    current_p0_rows: MatrixRows
    previous_p0_rows: MatrixRows | None
    predictor_name: str
    policy_id: str
    group_size: int
    bucket_rows: int
    policy_options: PolicyOptions
    topology_digest: str
    bucket_contract_digest: str


@dataclass
class TargetLayerPlannerMetrics:
    queue_wait_us: float = 0.0
    h1_us: float = 0.0
    h2_us: float = 0.0
    target_problem_us: float = 0.0
    raw_u_us: float = 0.0
    paired_b_us: float = 0.0
    safe_selection_us: float = 0.0
    encode_us: float = 0.0
    agreement_us: float = 0.0
    planner_wall_us: float = 0.0


@dataclass
class TargetLayerPlannerService:
    store: TargetPlanStore
    agreement_fn: Callable[[str], str] | None = None
    max_queue_size: int = 16
    _queue: queue.Queue[tuple[float, TargetLayerPlanningRequest] | None] = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _last_error: BaseException | None = field(init=False, default=None)
    _timeline: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=int(self.max_queue_size))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker, name="target-layer-planner", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def enqueue(self, request: TargetLayerPlanningRequest) -> None:
        if self._last_error is not None:
            raise RuntimeError(f"planner service already failed: {self._last_error}") from self._last_error
        self._queue.put((time.perf_counter_ns() / 1000.0, request))

    def timeline(self) -> list[dict[str, Any]]:
        return list(self._timeline)

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                return
            queued_at_us, request = item
            started_ns = time.perf_counter_ns()
            metrics = TargetLayerPlannerMetrics(queue_wait_us=max(0.0, (started_ns / 1000.0) - queued_at_us))
            try:
                bundle, plan = self._build_target_plan(request=request, metrics=metrics)
                key = TargetPlanKey(
                    run_id=request.run_id,
                    forward_epoch=int(request.forward_epoch),
                    microbatch_id=request.microbatch_id,
                    target_layer_id=request.target_layer_id,
                )
                self.store.put(key, plan)
                self._timeline.append(
                    {
                        "event": "target_plan_ready",
                        "target_layer_id": request.target_layer_id,
                        "logical_plan_digest": plan.logical_plan_digest,
                        "h1_digest": bundle.h1.matrix_digest,
                        "h2_digest": bundle.h2.matrix_digest,
                        "planner_wall_us": metrics.planner_wall_us,
                    }
                )
            except BaseException as exc:  # pragma: no cover - surfaced in tests
                self._last_error = exc
                self._timeline.append({"event": "planner_error", "error": f"{type(exc).__name__}: {exc}"})
                return

    def _build_target_plan(
        self,
        *,
        request: TargetLayerPlanningRequest,
        metrics: TargetLayerPlannerMetrics,
    ) -> tuple[TwoHorizonPredictionBundle, TargetLayerPreparedJointPlan]:
        planner_started_ns = time.perf_counter_ns()
        predictor = SharedTwoHorizonPredictor(predictor_name=request.predictor_name)
        bundle = predictor.predict_two_horizon(
            source_layer_id=str(request.source_layer_id),
            current_dispatch_matrix=request.current_p0_rows,
            previous_dispatch_matrix=request.previous_p0_rows,
            history_matrices=(() if request.previous_p0_rows is None else (request.previous_p0_rows,)),
        )
        metrics.h1_us = float(bundle.h1.prediction_us)
        metrics.h2_us = float(bundle.h2.prediction_us)
        target_problem_start = time.perf_counter_ns()
        h1 = bundle.h1.matrix_rows
        p1 = tuple(tuple(int(h1[col][row]) for col in range(len(h1))) for row in range(len(h1))) if h1 else ()
        replay_window = _TargetReplayWindow(
            fixture_id=request.run_id,
            window_id=f"{request.forward_epoch}:{request.microbatch_id}:{request.target_layer_id}",
            layer_id=int(request.target_layer_id) if str(request.target_layer_id).isdigit() else 0,
            p0_truth_rows=h1,
            p1_truth_rows=p1,
            p2_truth_rows=bundle.h2.matrix_rows,
            group_size=int(request.group_size),
        )
        target_problem_end = time.perf_counter_ns()
        metrics.target_problem_us = (target_problem_end - target_problem_start) / 1000.0
        raw_u_start = time.perf_counter_ns()
        scheduling_request = build_request_from_replay_window(
            replay_window=replay_window,
            p2_hint_rows=bundle.h2.matrix_rows,
            hint_type=bundle.h1.predictor,
            confidence=float(bundle.h1.confidence),
            bucket_rows=int(request.bucket_rows),
            policy_options=request.policy_options,
        )
        policy = build_policy(request.policy_id, request.policy_options)
        logical_plan = policy.plan(scheduling_request)
        raw_u_end = time.perf_counter_ns()
        metrics.raw_u_us = (raw_u_end - raw_u_start) / 1000.0
        encode_start = time.perf_counter_ns()
        logical_digest = stable_hash(logical_plan.to_dict())
        target_problem_digest = stable_hash(
            {
                "target_layer_id": request.target_layer_id,
                "h1": [list(row) for row in bundle.h1.matrix_rows],
                "h2": [list(row) for row in bundle.h2.matrix_rows],
                "policy": request.policy_id,
            }
        )
        encode_end = time.perf_counter_ns()
        metrics.encode_us = (encode_end - encode_start) / 1000.0
        agreement_start = time.perf_counter_ns()
        agreed_digest = logical_digest if self.agreement_fn is None else str(self.agreement_fn(logical_digest))
        agreement_end = time.perf_counter_ns()
        metrics.agreement_us = (agreement_end - agreement_start) / 1000.0
        finished_ns = time.perf_counter_ns()
        metrics.planner_wall_us = (finished_ns - planner_started_ns) / 1000.0
        plan = TargetLayerPreparedJointPlan(
            source_layer_id=str(request.source_layer_id),
            target_layer_id=str(request.target_layer_id),
            run_id=request.run_id,
            forward_epoch=int(request.forward_epoch),
            microbatch_id=request.microbatch_id,
            h1_prediction_digest=str(bundle.h1.matrix_digest),
            h2_prediction_digest=str(bundle.h2.matrix_digest),
            target_problem_digest=str(target_problem_digest),
            logical_plan=logical_plan,
            logical_plan_digest=str(agreed_digest),
            policy=str(request.policy_id),
            weights={
                "residual_weight": float(request.policy_options.residual_weight),
                "barrier_weight": float(request.policy_options.barrier_weight),
                "age_weight": float(request.policy_options.age_weight),
                "prediction_weight": float(request.policy_options.prediction_weight),
            },
            bucket_contract_digest=str(request.bucket_contract_digest),
            topology_digest=str(request.topology_digest),
            h1_rows=bundle.h1.matrix_rows,
            derived_p1_rows=p1,
            h2_rows=bundle.h2.matrix_rows,
            created_at_ns=int(planner_started_ns),
            ready_at_ns=int(finished_ns),
            selected_variant="raw_u",
            raw_logical_plan_digest=str(logical_digest),
            paired_b_logical_plan_digest="",
        )
        return bundle, plan

