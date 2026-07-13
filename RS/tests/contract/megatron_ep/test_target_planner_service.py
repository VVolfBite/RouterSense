from __future__ import annotations

from dataclasses import dataclass
import time

from rs.core.contracts import PlanWave, PlannedFlow, PlanningRequest, WindowPlan
from rs.planning import PlannerPolicyConfig, PlannerSelectionMode
from rs.runtime.online.megatron_ep.target_planning import TargetPlanKey
from rs.runtime.online.megatron_ep.target_planning.planner_service import (
    PreparationSubmitStatus,
    TargetLayerPlannerMetrics,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
)
from rs.runtime.online.megatron_ep.target_planning.predictor import TwoHorizonPredictionBundle
from rs.runtime.online.megatron_ep.target_planning.contracts import TwoHorizonPrediction
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore
from rs.planning import PlannerRegistry
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig


def _request(*, safe_projection_mode: str) -> TargetLayerPlanningRequest:
    return TargetLayerPlanningRequest(
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
        current_p0_rows=((0, 2, 5), (3, 0, 3), (1, 5, 0)),
        previous_p0_rows=((0, 1, 4), (2, 0, 2), (1, 4, 0)),
        predictor_name="copy_current_dispatch",
        policy_id="U_barrier_criticality_global_matching",
        raw_u_policy_id="U_barrier_criticality_global_matching",
        paired_b_policy_id="B_barrier_criticality_core_independent",
        safe_projection_mode=safe_projection_mode,
        group_size=3,
        bucket_rows=0,
        policy_options=PlannerPolicyConfig(),
        topology_digest="topo",
        bucket_contract_digest="dynamic_current",
    )


def _runtime() -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="U_barrier_criticality_global_matching",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
            safe_projection_mode="disabled",
            observation_profile="execution",
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1, 2),
        ep_group_root_global_rank=0,
    )


def test_raw_target_planner_does_not_build_paired_b() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    _bundle, plan = service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert plan.selected_variant == "raw_u"
    assert plan.paired_b_logical_plan_digest == ""
    assert plan.safe_projection_mode == "disabled"
    assert plan.paired_b_plan_was_built is False
    assert plan.paired_b_plan_was_scored is False
    assert plan.safe_selection_us == 0.0
    assert plan.paired_b_build_us == 0.0


def test_safe_target_planner_builds_paired_b_and_records_selection() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    _bundle, plan = service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="host_select"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert plan.raw_logical_plan_digest != ""
    assert plan.paired_b_logical_plan_digest != ""
    assert plan.selected_logical_plan_digest != ""
    assert plan.safe_projection_mode == "host_select"
    assert plan.paired_b_plan_was_built is True
    assert plan.paired_b_plan_was_scored is True
    assert plan.selected_variant in {"raw_u", "paired_b"}
    assert plan.paired_b_build_us >= 0.0
    assert plan.safe_selection_us >= 0.0


@dataclass
class _CountingPlanner:
    planner_id: str
    planner_family: str
    counter: dict[str, int]

    def plan(self, request: PlanningRequest) -> WindowPlan:
        self.counter[self.planner_id] = int(self.counter.get(self.planner_id, 0)) + 1
        return WindowPlan(
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            request_digest=request.semantic_digest(),
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(PlannedFlow(flow_id=f"{self.planner_id}:0", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=2, release_state="ready", executable=True),),
                    estimated_duration=0.0,
                ),
            ),
            metadata={"legacy_policy_name": self.planner_id},
        )


def test_target_planner_service_core_selection_does_not_replan() -> None:
    counter: dict[str, int] = {}

    def planner_factory(planner_id: str, _config) -> _CountingPlanner:
        family = "joint" if "joint" in planner_id or planner_id.startswith("U_") else "local"
        return _CountingPlanner(planner_id=planner_id, planner_family=family, counter=counter)

    service = TargetLayerPlannerService(store=TargetPlanStore(), planner_factory=planner_factory)
    from rs.core.contracts import (
        PlanningConstraints,
        PlanningIdentity,
        PlanningRequest,
        PlanningTopology,
        PlanningTraffic,
        PlanningWeights,
        PredictionHint,
    )

    formal_request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 2), (2, 0)), p1_return_rows=((0, 2), (2, 0))),
        prediction_hint=PredictionHint(predictor_id="copy_current", hint_type="traffic_matrix", target_dispatch_rows=((0, 2), (2, 0)), confidence=1.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    local_planner = planner_factory("fifo_bucket", None)
    joint_planner = planner_factory("barrier_criticality_joint", None)
    local_plan = local_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=local_plan,
        joint_plan=None,
        mode=PlannerSelectionMode.LOCAL,
    )
    assert compare_selector.selected_plan.planner_id == "fifo_bucket"
    assert counter["fifo_bucket"] == 1
    assert counter.get("barrier_criticality_joint", 0) == 0
    counter.clear()
    joint_plan = joint_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=None,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.JOINT,
    )
    assert compare_selector.selected_plan.planner_id == "barrier_criticality_joint"
    assert counter.get("fifo_bucket", 0) == 0
    assert counter["barrier_criticality_joint"] == 1
    counter.clear()
    local_plan = local_planner.plan(formal_request)
    joint_plan = joint_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=local_plan,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.COMPARE,
    )
    assert counter["fifo_bucket"] == 1
    assert counter["barrier_criticality_joint"] == 1


def test_target_planner_build_path_counts_planner_calls_once_per_mode() -> None:
    counter: dict[str, int] = {}

    def planner_factory(planner_id: str, _config) -> _CountingPlanner:
        family = "joint" if "joint" in planner_id or planner_id.startswith("U_") else "local"
        return _CountingPlanner(planner_id=planner_id, planner_family=family, counter=counter)

    service = TargetLayerPlannerService(store=TargetPlanStore(), planner_factory=planner_factory)
    service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert counter["U_barrier_criticality_global_matching"] == 1
    assert counter.get("B_barrier_criticality_core_independent", 0) == 0
    counter.clear()
    service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="host_select"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert counter["U_barrier_criticality_global_matching"] == 1
    assert counter["B_barrier_criticality_core_independent"] == 1


def test_sync_and_preplanned_formal_plan_digests_match_for_same_planning_request() -> None:
    matrix = ((0, 2, 5), (3, 0, 3), (1, 5, 0))
    runtime = _runtime()
    service = TargetLayerPlannerService(store=TargetPlanStore())
    bundle, _built = service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        metrics=TargetLayerPlannerMetrics(),
    )
    runtime_request = runtime._build_formal_planning_request(  # noqa: SLF001
        request_id="same-request",
        source_layer_id="1",
        target_layer_id="2",
        p0_dispatch_rows=bundle.h1.matrix_rows,
        p1_return_rows=tuple(tuple(int(bundle.h1.matrix_rows[col][row]) for col in range(len(bundle.h1.matrix_rows))) for row in range(len(bundle.h1.matrix_rows))),
        p2_hint_rows=bundle.h2.matrix_rows,
        predictor_name="copy_current_dispatch",
        prediction_confidence=float(bundle.h2.confidence),
    )
    service_request = service._build_planning_request(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        p0_dispatch_rows=bundle.h1.matrix_rows,
        p1_return_rows=tuple(tuple(int(bundle.h1.matrix_rows[col][row]) for col in range(len(bundle.h1.matrix_rows))) for row in range(len(bundle.h1.matrix_rows))),
        p2_hint_rows=bundle.h2.matrix_rows,
        predictor_id="copy_current_dispatch",
        prediction_confidence=float(bundle.h2.confidence),
        source_layer_id="1",
        target_layer_id="2",
    )
    assert runtime_request.semantic_digest() == service_request.semantic_digest()
    runtime_plan = PlannerRegistry.create("U_barrier_criticality_global_matching", None).plan(runtime_request)
    service_plan = PlannerRegistry.create("U_barrier_criticality_global_matching", None).plan(service_request)
    assert runtime_plan.semantic_digest() == service_plan.semantic_digest()


def test_target_planner_submit_replaces_stale_request_for_same_target() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=4)
    first = service.submit(_request(safe_projection_mode="disabled"))
    second = service.submit(_request(safe_projection_mode="disabled"))
    assert first.status is PreparationSubmitStatus.ACCEPTED
    assert second.status is PreparationSubmitStatus.REPLACED_STALE
    assert first.task_key == second.task_key


def test_target_planner_cancel_generation_rejects_future_submit() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=4)
    request = _request(safe_projection_mode="disabled")
    service.cancel_generation(run_id=request.run_id, forward_epoch=request.forward_epoch, microbatch_id=request.microbatch_id)
    result = service.submit(request)
    assert result.status is PreparationSubmitStatus.REJECTED_EXPIRED


def test_target_planner_worker_does_not_publish_until_main_thread_drains() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=4)
    request = _request(safe_projection_mode="disabled")
    key = service._task_key(request)  # noqa: SLF001
    agreement_calls: list[dict[str, object]] = []
    service.agreement_fn = lambda payload: agreement_calls.append(payload) or str(payload["logical_plan_digest"])
    service.start()
    result = service.submit(request)
    assert result.status is PreparationSubmitStatus.ACCEPTED
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready = service.drain_ready_publications()
        if ready:
            break
        time.sleep(0.01)
    assert ready, f"worker never produced ready publication for {key}"
    target_key = ready[0].key
    assert service.store.peek(target_key) is None
    assert agreement_calls == []
    published = service.publish_ready_plan(ready[0])
    assert published is not None
    assert service.store.peek(target_key) is not None
    assert len(agreement_calls) == 1
    service.shutdown()


def test_target_planner_cancelled_generation_drops_ready_publication_before_publish() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=4)
    request = _request(safe_projection_mode="disabled")
    service.start()
    result = service.submit(request)
    assert result.status is PreparationSubmitStatus.ACCEPTED
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready = service.drain_ready_publications()
        if ready:
            break
        time.sleep(0.01)
    assert ready
    service.cancel_generation(run_id=request.run_id, forward_epoch=request.forward_epoch, microbatch_id=request.microbatch_id)
    published = service.publish_ready_plan(ready[0])
    assert published is None
    assert service.store.peek(ready[0].key) is None
    service.shutdown()


def test_target_planner_worker_task_failure_does_not_stop_next_task() -> None:
    counter = {"calls": 0}

    def planner_factory(planner_id: str, _config):
        counter["calls"] += 1
        if counter["calls"] == 1:
            raise RuntimeError("boom")
        family = "joint" if "joint" in planner_id or planner_id.startswith("U_") else "local"
        return _CountingPlanner(planner_id=planner_id, planner_family=family, counter={})

    service = TargetLayerPlannerService(store=TargetPlanStore(), planner_factory=planner_factory, max_queue_size=4)
    first = _request(safe_projection_mode="disabled")
    second = TargetLayerPlanningRequest(**{**first.__dict__, "forward_epoch": 2})
    service.start()
    assert service.submit(first).status is PreparationSubmitStatus.ACCEPTED
    assert service.submit(second).status is PreparationSubmitStatus.ACCEPTED
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready = service.drain_ready_publications()
        if ready:
            break
        time.sleep(0.01)
    terminal = service.store.get_terminal_record(
        TargetPlanKey(
            run_id=first.run_id,
            forward_epoch=first.forward_epoch,
            microbatch_id=first.microbatch_id,
            target_layer_id=first.target_layer_id,
        )
    )
    assert terminal is not None
    assert terminal.final_status == "FAILED"
    assert ready
    assert ready[0].request.forward_epoch == 2
    service.shutdown()


def test_target_planner_close_rejects_submit_and_clears_thread() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=2)
    service.start()
    assert service.is_alive() is True
    service.close()
    assert service.is_alive() is False
    result = service.submit(_request(safe_projection_mode="disabled"))
    assert result.status is PreparationSubmitStatus.REJECTED_CLOSED


def test_target_planner_can_restart_after_close() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore(), max_queue_size=2)
    service.start()
    service.close()
    service.start()
    assert service.is_alive() is True
    service.shutdown()


@dataclass
class _CountingTwoHorizonPredictor:
    counter: dict[str, int]

    def predict_two_horizon(
        self,
        *,
        source_layer_id: str,
        current_dispatch_matrix,
        previous_dispatch_matrix=None,
        history_matrices=(),
    ) -> TwoHorizonPredictionBundle:
        self.counter["predict_calls"] = int(self.counter.get("predict_calls", 0)) + 1
        current = tuple(tuple(int(value) for value in row) for row in current_dispatch_matrix)
        next_layer_id = str(int(source_layer_id) + 1) if str(source_layer_id).isdigit() else f"{source_layer_id}+1"
        h2_target = str(int(source_layer_id) + 2) if str(source_layer_id).isdigit() else f"{source_layer_id}+2"
        return TwoHorizonPredictionBundle(
            h1=TwoHorizonPrediction(
                forecast_horizon=1,
                source_layer_id=str(source_layer_id),
                target_layer_id=str(next_layer_id),
                matrix_unit="rows",
                matrix_rows=current,
                matrix_digest=f"h1:{source_layer_id}",
                predictor="copy_current",
                confidence=1.0,
                created_at_ns=1,
                prediction_us=10.0,
            ),
            h2=TwoHorizonPrediction(
                forecast_horizon=2,
                source_layer_id=str(next_layer_id),
                target_layer_id=str(h2_target),
                matrix_unit="rows",
                matrix_rows=current,
                matrix_digest=f"h2:{source_layer_id}",
                predictor="copy_current",
                confidence=1.0,
                created_at_ns=2,
                prediction_us=10.0,
            ),
        )


def test_target_planner_keyed_queue_preserves_latest_same_key_once() -> None:
    counter: dict[str, int] = {}
    service = TargetLayerPlannerService(
        store=TargetPlanStore(),
        max_queue_size=1,
        two_horizon_predictor_factory=lambda _name: _CountingTwoHorizonPredictor(counter),
    )
    service.start()
    for _ in range(100):
        result = service.submit(_request(safe_projection_mode="disabled"))
        assert result.status in {PreparationSubmitStatus.ACCEPTED, PreparationSubmitStatus.REPLACED_STALE}
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready = service.drain_ready_publications()
        if ready:
            break
        time.sleep(0.01)
    assert ready
    assert counter["predict_calls"] == 1
    service.shutdown()


def test_target_planner_queue_full_new_key_drops_without_losing_existing_key() -> None:
    counter: dict[str, int] = {}
    service = TargetLayerPlannerService(
        store=TargetPlanStore(),
        max_queue_size=1,
        two_horizon_predictor_factory=lambda _name: _CountingTwoHorizonPredictor(counter),
    )
    service.start()
    first = _request(safe_projection_mode="disabled")
    second = TargetLayerPlanningRequest(**{**first.__dict__, "target_layer_id": "2"})
    assert service.submit(first).status is PreparationSubmitStatus.ACCEPTED
    assert service.submit(second).status is PreparationSubmitStatus.DROPPED_OVERLOAD
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready = service.drain_ready_publications()
        if ready:
            break
        time.sleep(0.01)
    assert ready
    assert ready[0].request.target_layer_id == "1"
    assert counter["predict_calls"] == 1
    service.shutdown()


def test_target_planner_stale_inflight_version_cannot_publish() -> None:
    counter: dict[str, int] = {}
    release_first = {"value": False}

    @dataclass
    class _SlowFirstPredictor(_CountingTwoHorizonPredictor):
        def predict_two_horizon(self, **kwargs) -> TwoHorizonPredictionBundle:
            call_index = int(self.counter.get("predict_calls", 0))
            if call_index == 0:
                deadline = time.time() + 2.0
                while not release_first["value"] and time.time() < deadline:
                    time.sleep(0.01)
            return super().predict_two_horizon(**kwargs)

    service = TargetLayerPlannerService(
        store=TargetPlanStore(),
        max_queue_size=2,
        two_horizon_predictor_factory=lambda _name: _SlowFirstPredictor(counter),
    )
    request = _request(safe_projection_mode="disabled")
    service.start()
    assert service.submit(request).status is PreparationSubmitStatus.ACCEPTED
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if service._inflight_by_key:  # noqa: SLF001
            break
        time.sleep(0.01)
    assert service.submit(request).status is PreparationSubmitStatus.REPLACED_STALE
    release_first["value"] = True
    deadline = time.time() + 5.0
    ready = []
    while time.time() < deadline:
        ready.extend(service.drain_ready_publications())
        if len(ready) >= 2:
            break
        time.sleep(0.01)
    assert len(ready) >= 2
    first = service.publish_ready_plan(ready[0])
    second = service.publish_ready_plan(ready[1])
    assert first is None
    assert second is not None
    service.shutdown()
