from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)
from rs.planning import CommonCorePlanEstimator, PlannerPolicyConfig, PlannerRegistry, PlannerSelectionMode, PlannerSelector, PlanningCostModel, SelectedPlan
from rs.planning.api import to_logical_plan
from rs.scheduling.validation import stable_hash
from rs.runtime.online.megatron_ep.public_types import LocalPreparationToken, LocalPublicationCandidate, PublicationSlot

from .contracts import MatrixRows, PreparationToken, TargetLayerPreparedJointPlan, TargetPlanKey
from .predictor import SharedTwoHorizonPredictor, TwoHorizonPredictionBundle
from .store import TargetPlanStore


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
    policy_options: PlannerPolicyConfig
    topology_digest: str
    bucket_contract_digest: str
    raw_u_policy_id: str = ""
    paired_b_policy_id: str = ""
    safe_projection_mode: str = "disabled"


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


class PreparationSubmitStatus(Enum):
    ACCEPTED = "accepted"
    REPLACED_STALE = "replaced_stale"
    DROPPED_OVERLOAD = "dropped_overload"
    REJECTED_EXPIRED = "rejected_expired"
    REJECTED_CLOSED = "rejected_closed"


@dataclass(frozen=True)
class PreparationSubmitResult:
    status: PreparationSubmitStatus
    task_key: str


@dataclass(frozen=True)
class _QueuedPlanningTask:
    queued_at_us: float
    request: TargetLayerPlanningRequest
    task_key: str
    task_version: int
    service_session_id: int
    publish_sequence: int


@dataclass(frozen=True)
class _BuiltPlanningResult:
    key: TargetPlanKey
    task_key: str
    request: TargetLayerPlanningRequest
    bundle: TwoHorizonPredictionBundle
    plan: TargetLayerPreparedJointPlan
    metrics: TargetLayerPlannerMetrics
    token: PreparationToken


@dataclass
class TargetLayerPlannerService:
    store: TargetPlanStore
    planner_factory: Callable[[str, Any | None], Any] = PlannerRegistry.create
    two_horizon_predictor_factory: Callable[[str], SharedTwoHorizonPredictor] | None = None
    max_queue_size: int = 16
    _queue: queue.Queue[str | None] = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _last_error: BaseException | None = field(init=False, default=None)
    _timeline: list[dict[str, Any]] = field(init=False, default_factory=list)
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock)
    _pending_by_key: dict[str, _QueuedPlanningTask] = field(init=False, default_factory=dict)
    _queued_keys: set[str] = field(init=False, default_factory=set)
    _inflight_by_key: dict[str, _QueuedPlanningTask] = field(init=False, default_factory=dict)
    _ready_results: list[_BuiltPlanningResult] = field(init=False, default_factory=list)
    _publication_state_by_slot: dict[str, LocalPublicationCandidate] = field(init=False, default_factory=dict)
    _cancelled_generations: set[tuple[str, int, str]] = field(init=False, default_factory=set)
    _closed: bool = field(init=False, default=False)
    _service_session_id: int = field(init=False, default=0)
    _next_task_version: int = field(init=False, default=0)
    _next_publish_sequence: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=int(self.max_queue_size))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._closed = False
        self._last_error = None
        self._service_session_id += 1
        self._next_task_version = 0
        self._next_publish_sequence = 0
        self._queue = queue.Queue(maxsize=int(self.max_queue_size))
        with self._lock:
            self._pending_by_key.clear()
            self._queued_keys.clear()
            self._inflight_by_key.clear()
            self._ready_results.clear()
            self._publication_state_by_slot.clear()
            self._cancelled_generations.clear()
        self._thread = threading.Thread(target=self._worker, name="target-layer-planner", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        with self._lock:
            self._pending_by_key.clear()
            self._queued_keys.clear()
            self._inflight_by_key.clear()
            self._ready_results.clear()
            self._publication_state_by_slot.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("target-layer-planner did not stop cleanly")
            self._thread = None

    def shutdown(self) -> None:
        self.close()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def _task_key(request: TargetLayerPlanningRequest) -> str:
        return f"{request.run_id}:{int(request.forward_epoch)}:{request.microbatch_id}:{request.target_layer_id}"

    @staticmethod
    def _generation_key(request: TargetLayerPlanningRequest) -> tuple[str, int, str]:
        return (str(request.run_id), int(request.forward_epoch), str(request.microbatch_id))

    def submit(self, request: TargetLayerPlanningRequest) -> PreparationSubmitResult:
        if self._last_error is not None:
            raise RuntimeError(f"planner service already failed: {self._last_error}") from self._last_error
        task_key = self._task_key(request)
        generation_key = self._generation_key(request)
        with self._lock:
            if self._closed or self._stop.is_set():
                return PreparationSubmitResult(status=PreparationSubmitStatus.REJECTED_CLOSED, task_key=task_key)
            if generation_key in self._cancelled_generations:
                return PreparationSubmitResult(status=PreparationSubmitStatus.REJECTED_EXPIRED, task_key=task_key)
            self._next_task_version += 1
            self._next_publish_sequence += 1
            queued = _QueuedPlanningTask(
                queued_at_us=(time.perf_counter_ns() / 1000.0),
                request=request,
                task_key=task_key,
                task_version=int(self._next_task_version),
                service_session_id=int(self._service_session_id),
                publish_sequence=int(self._next_publish_sequence),
            )
            replaced = task_key in self._pending_by_key or task_key in self._inflight_by_key
            needs_queue_slot = task_key not in self._queued_keys
            if needs_queue_slot:
                try:
                    self._queue.put_nowait(task_key)
                except queue.Full:
                    return PreparationSubmitResult(status=PreparationSubmitStatus.DROPPED_OVERLOAD, task_key=task_key)
                self._queued_keys.add(task_key)
            self._pending_by_key[task_key] = queued
            self.store.register_expected_publication(
                PreparationToken(
                    service_session_id=int(queued.service_session_id),
                    forward_generation=int(request.forward_epoch),
                    target_key=TargetPlanKey(
                        run_id=request.run_id,
                        forward_epoch=int(request.forward_epoch),
                        microbatch_id=request.microbatch_id,
                        target_layer_id=request.target_layer_id,
                    ),
                    task_version=int(queued.task_version),
                    publish_sequence=int(queued.publish_sequence),
                )
            )
            slot = PublicationSlot(
                run_id=str(request.run_id),
                forward_generation=int(request.forward_epoch),
                microbatch_id=str(request.microbatch_id),
                source_layer_id=str(request.source_layer_id),
                target_layer_id=str(request.target_layer_id),
                planning_slot=f"{request.source_layer_id}->{request.target_layer_id}",
            )
            self._publication_state_by_slot[str(slot.semantic_digest())] = LocalPublicationCandidate(
                slot=slot,
                planner_id=str(request.policy_id),
                logical_plan_digest="",
                token=LocalPreparationToken(
                    service_session_id=int(queued.service_session_id),
                    forward_generation=int(queued.request.forward_epoch),
                    target_layer_id=str(queued.request.target_layer_id),
                    task_version=int(queued.task_version),
                    publication_slot_digest=str(slot.semantic_digest()),
                ),
                status="BUILDING",
                metadata={"task_key": task_key},
            )
        return PreparationSubmitResult(
            status=PreparationSubmitStatus.REPLACED_STALE if replaced else PreparationSubmitStatus.ACCEPTED,
            task_key=task_key,
        )

    def enqueue(self, request: TargetLayerPlanningRequest) -> None:
        result = self.submit(request)
        if result.status in {PreparationSubmitStatus.REJECTED_CLOSED, PreparationSubmitStatus.REJECTED_EXPIRED, PreparationSubmitStatus.DROPPED_OVERLOAD}:
            raise RuntimeError(f"target_planner_submit_failed:{result.status.value}:{result.task_key}")

    def timeline(self) -> list[dict[str, Any]]:
        return list(self._timeline)

    def cancel_generation(self, *, run_id: str, forward_epoch: int, microbatch_id: str) -> None:
        generation_key = (str(run_id), int(forward_epoch), str(microbatch_id))
        with self._lock:
            self._cancelled_generations.add(generation_key)
            doomed = [task_key for task_key, item in self._pending_by_key.items() if self._generation_key(item.request) == generation_key]
            for task_key in doomed:
                self._pending_by_key.pop(task_key, None)
                self._queued_keys.discard(task_key)
            self._ready_results = [item for item in self._ready_results if self._generation_key(item.request) != generation_key]
            for slot_digest, candidate in list(self._publication_state_by_slot.items()):
                if (
                    str(candidate.slot.run_id),
                    int(candidate.slot.forward_generation),
                    str(candidate.slot.microbatch_id),
                ) == generation_key:
                    self._publication_state_by_slot[slot_digest] = LocalPublicationCandidate(
                        slot=candidate.slot,
                        planner_id=str(candidate.planner_id),
                        logical_plan_digest=str(candidate.logical_plan_digest),
                        token=candidate.token,
                        status="CANCELLED",
                        metadata=dict(candidate.metadata),
                    )

    def cancel_before_generation(self, *, run_id: str, microbatch_id: str, current_generation: int) -> None:
        with self._lock:
            generation_keys = {
                (str(item.request.run_id), int(item.request.forward_epoch), str(item.request.microbatch_id))
                for item in self._pending_by_key.values()
                if str(item.request.run_id) == str(run_id)
                and str(item.request.microbatch_id) == str(microbatch_id)
                and int(item.request.forward_epoch) < int(current_generation)
            }
            generation_keys.update(
                {
                    (str(item.request.run_id), int(item.request.forward_epoch), str(item.request.microbatch_id))
                    for item in self._inflight_by_key.values()
                    if str(item.request.run_id) == str(run_id)
                    and str(item.request.microbatch_id) == str(microbatch_id)
                    and int(item.request.forward_epoch) < int(current_generation)
                }
            )
            generation_keys.update(
                {
                    (str(item.request.run_id), int(item.request.forward_epoch), str(item.request.microbatch_id))
                    for item in self._ready_results
                    if str(item.request.run_id) == str(run_id)
                    and str(item.request.microbatch_id) == str(microbatch_id)
                    and int(item.request.forward_epoch) < int(current_generation)
                }
            )
        for generation_key in sorted(generation_keys, key=lambda item: int(item[1])):
            self.cancel_generation(
                run_id=str(generation_key[0]),
                forward_epoch=int(generation_key[1]),
                microbatch_id=str(generation_key[2]),
            )

    def cancel_slot(self, slot: PublicationSlot, *, final_status: str) -> None:
        slot_digest = str(slot.semantic_digest())
        terminal_status = str(final_status).upper()
        with self._lock:
            candidate = self._publication_state_by_slot.get(slot_digest)
            if candidate is None:
                return
            if str(candidate.status).upper() in {"READY", "BUILDING", "FAILED", "CANCELLED", "EXPIRED", "SLOT_MISMATCH"}:
                self._publication_state_by_slot[slot_digest] = LocalPublicationCandidate(
                    slot=candidate.slot,
                    planner_id=str(candidate.planner_id),
                    logical_plan_digest=str(candidate.logical_plan_digest),
                    token=candidate.token,
                    status=terminal_status,
                    metadata=dict(candidate.metadata),
                )

    def drain_ready_publications(self) -> list[_BuiltPlanningResult]:
        with self._lock:
            ready = list(self._ready_results)
            self._ready_results.clear()
            return ready

    def _worker(self) -> None:
        while True:
            task_key = self._queue.get()
            if task_key is None:
                return
            if self._stop.is_set():
                continue
            with self._lock:
                self._queued_keys.discard(task_key)
                item = self._pending_by_key.pop(task_key, None)
                if item is None:
                    continue
                self._inflight_by_key[task_key] = item
            request = item.request
            started_ns = time.perf_counter_ns()
            metrics = TargetLayerPlannerMetrics(queue_wait_us=max(0.0, (started_ns / 1000.0) - float(item.queued_at_us)))
            try:
                bundle, plan = self._build_target_plan(request=request, metrics=metrics)
                key = TargetPlanKey(
                    run_id=request.run_id,
                    forward_epoch=int(request.forward_epoch),
                    microbatch_id=request.microbatch_id,
                    target_layer_id=request.target_layer_id,
                )
                token = PreparationToken(
                    service_session_id=int(item.service_session_id),
                    forward_generation=int(request.forward_epoch),
                    target_key=key,
                    task_version=int(item.task_version),
                    publish_sequence=int(item.publish_sequence),
                )
                with self._lock:
                    current = self._inflight_by_key.get(task_key)
                    if current is None or current.task_version != item.task_version:
                        continue
                    self._inflight_by_key.pop(task_key, None)
                    newer_pending = self._pending_by_key.get(task_key)
                    if newer_pending is not None and int(newer_pending.task_version) > int(item.task_version):
                        continue
                    if (
                        self._generation_key(request) in self._cancelled_generations
                        or int(item.service_session_id) != int(self._service_session_id)
                    ):
                        continue
                    self._ready_results.append(
                        _BuiltPlanningResult(
                            key=key,
                            task_key=task_key,
                            request=request,
                            bundle=bundle,
                            plan=plan,
                            metrics=metrics,
                            token=token,
                        )
                    )
                    slot_digest = str(PublicationSlot(
                        run_id=str(request.run_id),
                        forward_generation=int(request.forward_epoch),
                        microbatch_id=str(request.microbatch_id),
                        source_layer_id=str(request.source_layer_id),
                        target_layer_id=str(request.target_layer_id),
                        planning_slot=f"{request.source_layer_id}->{request.target_layer_id}",
                    ).semantic_digest())
                    self._publication_state_by_slot[slot_digest] = LocalPublicationCandidate(
                        slot=PublicationSlot(
                            run_id=str(request.run_id),
                            forward_generation=int(request.forward_epoch),
                            microbatch_id=str(request.microbatch_id),
                            source_layer_id=str(request.source_layer_id),
                            target_layer_id=str(request.target_layer_id),
                            planning_slot=f"{request.source_layer_id}->{request.target_layer_id}",
                        ),
                        planner_id=str(plan.policy),
                        logical_plan_digest=str(plan.logical_plan_digest),
                        token=LocalPreparationToken(
                            service_session_id=int(token.service_session_id),
                            forward_generation=int(token.forward_generation),
                            target_layer_id=str(key.target_layer_id),
                            task_version=int(token.task_version),
                            publication_slot_digest=slot_digest,
                        ),
                        status="READY",
                        metadata={
                            "target_key": key.to_dict(),
                            "plan": plan.to_dict(),
                            "h1_digest": str(bundle.h1.matrix_digest),
                            "h2_digest": str(bundle.h2.matrix_digest),
                            "planner_wall_us": float(metrics.planner_wall_us),
                        },
                    )
                self._timeline.append(
                    {
                        "event": "target_plan_built",
                        "target_layer_id": request.target_layer_id,
                        "logical_plan_digest": plan.logical_plan_digest,
                        "h1_digest": bundle.h1.matrix_digest,
                        "h2_digest": bundle.h2.matrix_digest,
                        "planner_wall_us": metrics.planner_wall_us,
                    }
                )
            except BaseException as exc:  # pragma: no cover - surfaced in tests
                key = TargetPlanKey(
                    run_id=request.run_id,
                    forward_epoch=int(request.forward_epoch),
                    microbatch_id=request.microbatch_id,
                    target_layer_id=request.target_layer_id,
                )
                publish_failed = False
                slot = PublicationSlot(
                    run_id=str(request.run_id),
                    forward_generation=int(request.forward_epoch),
                    microbatch_id=str(request.microbatch_id),
                    source_layer_id=str(request.source_layer_id),
                    target_layer_id=str(request.target_layer_id),
                    planning_slot=f"{request.source_layer_id}->{request.target_layer_id}",
                )
                with self._lock:
                    current = self._inflight_by_key.get(task_key)
                    if current is not None and current.task_version == item.task_version:
                        self._inflight_by_key.pop(task_key, None)
                    newer_pending_exists = task_key in self._pending_by_key
                    stale_session = int(item.service_session_id) != int(self._service_session_id)
                    cancelled = self._generation_key(request) in self._cancelled_generations
                    if not newer_pending_exists and not stale_session and not cancelled:
                        publish_failed = True
                        self._publication_state_by_slot[str(slot.semantic_digest())] = LocalPublicationCandidate(
                            slot=slot,
                            planner_id=str(request.policy_id),
                            logical_plan_digest="",
                            token=LocalPreparationToken(
                                service_session_id=int(item.service_session_id),
                                forward_generation=int(request.forward_epoch),
                                target_layer_id=str(request.target_layer_id),
                                task_version=int(item.task_version),
                                publication_slot_digest=str(slot.semantic_digest()),
                            ),
                            status="FAILED",
                            metadata={"error": f"{type(exc).__name__}: {exc}"},
                        )
                    elif cancelled:
                        self._publication_state_by_slot[str(slot.semantic_digest())] = LocalPublicationCandidate(
                            slot=slot,
                            planner_id=str(request.policy_id),
                            logical_plan_digest="",
                            token=LocalPreparationToken(
                                service_session_id=int(item.service_session_id),
                                forward_generation=int(request.forward_epoch),
                                target_layer_id=str(request.target_layer_id),
                                task_version=int(item.task_version),
                                publication_slot_digest=str(slot.semantic_digest()),
                            ),
                            status="CANCELLED",
                            metadata={"error": f"{type(exc).__name__}: {exc}"},
                        )
                if publish_failed:
                    self.store.close_key_if_unclaimed(
                        key,
                        final_status="FAILED",
                        execution_origin=f"task_failure:{type(exc).__name__}",
                    )
                self._timeline.append(
                    {
                        "event": "planner_task_failed",
                        "task_key": task_key,
                        "target_layer_id": request.target_layer_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

    def local_publication_candidate(self, ready: _BuiltPlanningResult) -> LocalPublicationCandidate | None:
        if self._generation_key(ready.request) in self._cancelled_generations:
            return None
        with self._lock:
            if int(ready.token.service_session_id) != int(self._service_session_id):
                return None
        slot = PublicationSlot(
            run_id=str(ready.request.run_id),
            forward_generation=int(ready.request.forward_epoch),
            microbatch_id=str(ready.request.microbatch_id),
            source_layer_id=str(ready.request.source_layer_id),
            target_layer_id=str(ready.request.target_layer_id),
            planning_slot=f"{ready.request.source_layer_id}->{ready.request.target_layer_id}",
        )
        return LocalPublicationCandidate(
            slot=slot,
            planner_id=str(ready.plan.policy),
            logical_plan_digest=str(ready.plan.logical_plan_digest),
            token=LocalPreparationToken(
                service_session_id=int(ready.token.service_session_id),
                forward_generation=int(ready.token.forward_generation),
                target_layer_id=str(ready.key.target_layer_id),
                task_version=int(ready.token.task_version),
                publication_slot_digest=str(slot.semantic_digest()),
            ),
            status="READY",
            metadata={
                "target_key": ready.key.to_dict(),
                "plan": ready.plan.to_dict(),
                "h1_digest": str(ready.bundle.h1.matrix_digest),
                "h2_digest": str(ready.bundle.h2.matrix_digest),
                "planner_wall_us": float(ready.metrics.planner_wall_us),
            },
        )

    def publication_state_for_slot(self, slot: PublicationSlot) -> LocalPublicationCandidate | None:
        with self._lock:
            return self._publication_state_by_slot.get(str(slot.semantic_digest()))

    def _select_candidate_plans(
        self,
        *,
        planning_request: PlanningRequest,
        local_plan,
        joint_plan,
        mode: PlannerSelectionMode,
    ) -> SelectedPlan:
        selector = PlannerSelector(
            local_planner=self.planner_factory("fifo_bucket", None) if local_plan is None else self.planner_factory(getattr(local_plan, "planner_id", "fifo_bucket"), None),
            joint_planner=self.planner_factory("barrier_criticality_joint", None) if joint_plan is None else self.planner_factory(getattr(joint_plan, "planner_id", "barrier_criticality_joint"), None),
            estimator=CommonCorePlanEstimator(),
            cost_model=PlanningCostModel(
                expert_compute_delay=float(planning_request.constraints.expert_compute_delay),
                full_duplex=bool(planning_request.topology.full_duplex),
                max_outgoing_per_rank_per_wave=int(planning_request.topology.max_outgoing_per_rank_per_wave),
                max_incoming_per_rank_per_wave=int(planning_request.topology.max_incoming_per_rank_per_wave),
            ),
        )
        return selector.select_prebuilt(
            request=planning_request,
            local_plan=local_plan,
            joint_plan=joint_plan,
            mode=mode,
        )

    def _build_target_plan(
        self,
        *,
        request: TargetLayerPlanningRequest,
        metrics: TargetLayerPlannerMetrics,
    ) -> tuple[TwoHorizonPredictionBundle, TargetLayerPreparedJointPlan]:
        planner_started_ns = time.perf_counter_ns()
        predictor_factory = self.two_horizon_predictor_factory or (lambda predictor_name: SharedTwoHorizonPredictor(predictor_name=predictor_name))
        predictor = predictor_factory(request.predictor_name)
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
        h1_source_layer_id = str(bundle.h1.source_layer_id)
        h1_target_layer_id = str(bundle.h1.target_layer_id)
        h2_source_layer_id = str(bundle.h2.source_layer_id)
        h2_target_layer_id = str(bundle.h2.target_layer_id)
        safe_projection_mode = str(request.safe_projection_mode)
        planning_request = self._build_planning_request(
            request=request,
            p0_dispatch_rows=h1,
            p1_return_rows=p1,
            p2_hint_rows=bundle.h2.matrix_rows,
            predictor_id=str(bundle.h2.predictor),
            prediction_confidence=float(bundle.h2.confidence),
            source_layer_id=h2_source_layer_id,
            target_layer_id=h2_target_layer_id,
        )
        target_problem_end = time.perf_counter_ns()
        metrics.target_problem_us = (target_problem_end - target_problem_start) / 1000.0
        raw_u_start = time.perf_counter_ns()
        raw_policy_id = str(request.raw_u_policy_id or request.policy_id)
        raw_planner = self.planner_factory(raw_policy_id, None)
        raw_plan = raw_planner.plan(planning_request)
        raw_logical_plan = to_logical_plan(raw_plan)
        raw_u_end = time.perf_counter_ns()
        metrics.raw_u_us = (raw_u_end - raw_u_start) / 1000.0
        paired_b_logical_plan = raw_logical_plan
        estimator = CommonCorePlanEstimator()
        cost_model = PlanningCostModel(
            expert_compute_delay=float(planning_request.constraints.expert_compute_delay),
            full_duplex=bool(planning_request.topology.full_duplex),
            max_outgoing_per_rank_per_wave=int(planning_request.topology.max_outgoing_per_rank_per_wave),
            max_incoming_per_rank_per_wave=int(planning_request.topology.max_incoming_per_rank_per_wave),
        )
        raw_score = estimator.estimate(raw_plan, planning_request, cost_model)
        paired_b_makespan = float(raw_score.estimated_makespan)
        selected_variant = "raw_u"
        selected_plan = raw_plan
        selected_score = raw_score
        safe_selection_us = 0.0
        paired_b_us = 0.0
        if safe_projection_mode == "host_select":
            paired_b_start = time.perf_counter_ns()
            paired_policy_id = str(request.paired_b_policy_id or "")
            if not paired_policy_id:
                raise RuntimeError("safe target planner missing paired_b_policy_id")
            paired_b_planner = self.planner_factory(paired_policy_id, None)
            paired_b_plan = paired_b_planner.plan(planning_request)
            paired_b_logical_plan = to_logical_plan(paired_b_plan)
            paired_b_end = time.perf_counter_ns()
            paired_b_us = (paired_b_end - paired_b_start) / 1000.0
            metrics.paired_b_us = float(paired_b_us)
            safe_start = time.perf_counter_ns()
            selected = self._select_candidate_plans(
                planning_request=planning_request,
                local_plan=paired_b_plan,
                joint_plan=raw_plan,
                mode=PlannerSelectionMode.COMPARE,
            )
            selected_variant = "paired_b" if selected.selected_plan.planner_id == paired_b_plan.planner_id else "raw_u"
            selected_plan = selected.selected_plan
            selected_score = selected.selected_score
            safe_end = time.perf_counter_ns()
            safe_selection_us = (safe_end - safe_start) / 1000.0
            metrics.safe_selection_us = float(safe_selection_us)
            paired_b_makespan = float(selected.local_score.estimated_makespan if selected.local_score is not None else 0.0)
        selected_logical_plan = to_logical_plan(selected_plan)
        encode_start = time.perf_counter_ns()
        logical_digest = stable_hash(selected_logical_plan.to_dict())
        target_problem_digest = stable_hash(
            {
                "target_layer_id": request.target_layer_id,
                "h1": [list(row) for row in bundle.h1.matrix_rows],
                "h2": [list(row) for row in bundle.h2.matrix_rows],
                "policy": raw_policy_id,
                "safe_projection_mode": str(request.safe_projection_mode),
            }
        )
        encode_end = time.perf_counter_ns()
        metrics.encode_us = (encode_end - encode_start) / 1000.0
        finished_ns = time.perf_counter_ns()
        metrics.planner_wall_us = (finished_ns - planner_started_ns) / 1000.0
        paired_b_plan_was_built = safe_projection_mode == "host_select"
        paired_b_plan_was_scored = paired_b_plan_was_built
        paired_b_plan_was_selected = selected_variant == "paired_b"
        raw_u_plan_was_selected = selected_variant == "raw_u"
        plan = TargetLayerPreparedJointPlan(
            source_layer_id=str(request.source_layer_id),
            target_layer_id=str(request.target_layer_id),
            run_id=request.run_id,
            forward_epoch=int(request.forward_epoch),
            microbatch_id=request.microbatch_id,
            h1_prediction_digest=str(bundle.h1.matrix_digest),
            h2_prediction_digest=str(bundle.h2.matrix_digest),
            target_problem_digest=str(target_problem_digest),
            logical_plan=selected_logical_plan,
            logical_plan_digest=str(logical_digest),
            policy=str(raw_policy_id if selected_variant == "raw_u" else (request.paired_b_policy_id or raw_policy_id)),
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
            safe_projection_mode=safe_projection_mode,
            selected_variant=str(selected_variant),
            raw_logical_plan_digest=str(stable_hash(raw_logical_plan.to_dict())),
            paired_b_logical_plan_digest=(
                ""
                if safe_projection_mode != "host_select"
                else str(stable_hash(paired_b_logical_plan.to_dict()))
            ),
            selected_logical_plan_digest=str(logical_digest),
            raw_u_estimated_makespan=float(raw_score.estimated_makespan),
            paired_b_estimated_makespan=0.0 if not paired_b_plan_was_scored else float(paired_b_makespan),
            raw_u_build_us=float(metrics.raw_u_us),
            paired_b_build_us=float(paired_b_us),
            safe_selection_us=float(safe_selection_us),
            raw_u_plan_was_built=True,
            raw_u_plan_was_scored=True,
            raw_u_plan_was_selected=raw_u_plan_was_selected,
            paired_b_plan_was_built=paired_b_plan_was_built,
            paired_b_plan_was_scored=paired_b_plan_was_scored,
            paired_b_plan_was_selected=paired_b_plan_was_selected,
        )
        return bundle, plan

    @staticmethod
    def _build_planning_request(
        *,
        request: TargetLayerPlanningRequest,
        p0_dispatch_rows: MatrixRows,
        p1_return_rows: MatrixRows,
        p2_hint_rows: MatrixRows,
        predictor_id: str,
        prediction_confidence: float,
        source_layer_id: str,
        target_layer_id: str,
    ) -> PlanningRequest:
        return PlanningRequest(
            identity=PlanningIdentity(
                request_id=f"{request.run_id}:{request.forward_epoch}:{request.microbatch_id}:{request.target_layer_id}",
                run_id=request.run_id,
                forward_id=str(request.forward_epoch),
                window_id=f"{request.forward_epoch}:{request.microbatch_id}:{request.target_layer_id}",
                source_layer_id=str(source_layer_id),
                target_layer_id=str(target_layer_id),
            ),
            traffic=PlanningTraffic(
                p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p0_dispatch_rows),
                p1_return_rows=tuple(tuple(int(v) for v in row) for row in p1_return_rows),
            ),
            prediction_hint=PredictionHint(
                predictor_id=str(predictor_id),
                hint_type="traffic_matrix",
                target_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p2_hint_rows),
                confidence=float(prediction_confidence),
                oracle=False,
                source_layer_id=str(source_layer_id),
                target_layer_id=str(target_layer_id),
            ),
            topology=PlanningTopology(world_size=int(request.group_size)),
            constraints=PlanningConstraints(
                bucket_rows=int(request.bucket_rows),
                max_waves=256,
                expert_compute_delay=0.0,
                phase_release_model="p1_return",
            ),
            weights=PlanningWeights(
                p0_weight=float(request.policy_options.p0_weight),
                p1_weight=float(request.policy_options.p1_weight),
                p2_weight=float(request.policy_options.p2_hint_weight),
                residual_weight=float(request.policy_options.residual_weight),
                barrier_weight=float(request.policy_options.barrier_weight),
                age_weight=float(request.policy_options.age_weight),
                prediction_weight=float(request.policy_options.prediction_weight),
                criticality_weight=float(request.policy_options.criticality_weight),
                iteration_budget=request.policy_options.iteration_budget,
            ),
            information_mode="p0_p1_p2",
        )
