from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace
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

from .contracts import MatrixRows, TargetLayerPreparedJointPlan, TargetPlanKey
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


@dataclass
class TargetLayerPlannerService:
    store: TargetPlanStore
    agreement_fn: Callable[[dict[str, Any]], str] | None = None
    planner_factory: Callable[[str, Any | None], Any] = PlannerRegistry.create
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
                agreement_start = time.perf_counter_ns()
                published = self.publish_agreed_plan(key=key, plan=plan)
                agreement_end = time.perf_counter_ns()
                metrics.agreement_us = (agreement_end - agreement_start) / 1000.0
                self._timeline.append(
                    {
                        "event": "target_plan_ready",
                        "target_layer_id": request.target_layer_id,
                        "logical_plan_digest": published.logical_plan_digest,
                        "h1_digest": bundle.h1.matrix_digest,
                        "h2_digest": bundle.h2.matrix_digest,
                        "planner_wall_us": metrics.planner_wall_us,
                    }
                )
            except BaseException as exc:  # pragma: no cover - surfaced in tests
                self._last_error = exc
                self._timeline.append({"event": "planner_error", "error": f"{type(exc).__name__}: {exc}"})
                return

    def publish_agreed_plan(self, *, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> TargetLayerPreparedJointPlan:
        agreement_payload = self._agreement_payload(key=key, plan=plan)
        agreement_start = time.perf_counter_ns()
        agreed_digest = str(plan.logical_plan_digest)
        if self.agreement_fn is not None:
            agreed_digest = str(self.agreement_fn(agreement_payload))
        agreement_end = time.perf_counter_ns()
        published = replace(plan, logical_plan_digest=str(agreed_digest), ready_at_ns=int(time.perf_counter_ns()))
        self.store.put(key, published)
        self._timeline.append(
            {
                "event": "target_plan_agreed_publish",
                "target_layer_id": key.target_layer_id,
                "agreement_us": float((agreement_end - agreement_start) / 1000.0),
                "logical_plan_digest": str(published.logical_plan_digest),
                "h1_digest": str(published.h1_prediction_digest),
                "h2_digest": str(published.h2_prediction_digest),
            }
        )
        return published

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
        h1_source_layer_id = str(bundle.h1.source_layer_id)
        h1_target_layer_id = str(bundle.h1.target_layer_id)
        h2_source_layer_id = str(bundle.h2.source_layer_id)
        h2_target_layer_id = str(bundle.h2.target_layer_id)
        safe_projection_mode = str(request.safe_projection_mode)
        planning_request = PlanningRequest(
            identity=PlanningIdentity(
                request_id=f"{request.run_id}:{request.forward_epoch}:{request.microbatch_id}:{request.target_layer_id}",
                run_id=request.run_id,
                forward_id=str(request.forward_epoch),
                window_id=f"{request.forward_epoch}:{request.microbatch_id}:{request.target_layer_id}",
                source_layer_id=h2_source_layer_id,
                target_layer_id=h2_target_layer_id,
            ),
            traffic=PlanningTraffic(
                p0_dispatch_rows=h1,
                p1_return_rows=p1,
            ),
            prediction_hint=PredictionHint(
                predictor_id=str(bundle.h2.predictor),
                hint_type="traffic_matrix",
                target_dispatch_rows=bundle.h2.matrix_rows,
                confidence=float(bundle.h2.confidence),
                oracle=False,
                source_layer_id=h2_source_layer_id,
                target_layer_id=h2_target_layer_id,
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
    def _agreement_payload(*, key: TargetPlanKey, plan: TargetLayerPreparedJointPlan) -> dict[str, Any]:
        return {
            "key": key.to_dict(),
            "logical_plan_digest": str(plan.logical_plan_digest),
            "selected_variant": str(plan.selected_variant),
            "raw_u_logical_plan_digest": str(plan.raw_logical_plan_digest),
            "paired_b_logical_plan_digest": str(plan.paired_b_logical_plan_digest),
            "policy": str(plan.policy),
            "weights_digest": stable_hash(dict(plan.weights or {})),
            "topology_digest": str(plan.topology_digest),
            "bucket_contract_digest": str(plan.bucket_contract_digest),
            "h1_prediction_digest": str(plan.h1_prediction_digest),
            "h2_prediction_digest": str(plan.h2_prediction_digest),
        }
