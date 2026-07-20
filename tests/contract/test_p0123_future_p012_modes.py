from __future__ import annotations

import pytest

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningTopology,
    PlanningWeights,
)
from rs.experiments_support.runtime_presets import resolve_strategy_runtime
from rs.planning import PlannerRegistry, PlannerPolicyConfig
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.target_planning.contracts import TargetLayerPreparedJointPlan
from rs.runtime.online.megatron_ep.target_planning.planner_service import (
    TargetLayerPlannerMetrics,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
)
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore


def _transpose(matrix):
    return tuple(tuple(int(matrix[col][row]) for col in range(len(matrix))) for row in range(len(matrix)))


def _planning_request(*, mode: str, p3_weight: float):
    p0 = (
        (0, 0, 4, 4),
        (0, 0, 1, 4),
        (2, 8, 0, 4),
        (0, 4, 0, 0),
    )
    p2 = (
        (0, 2, 1, 4),
        (0, 0, 0, 8),
        (2, 4, 0, 4),
        (2, 2, 8, 0),
    )
    return build_window_planning_request(
        identity=PlanningIdentity(request_id=f"request:{mode}:{p3_weight}"),
        p0_dispatch_rows=p0,
        p1_return_rows=_transpose(p0),
        p2_hint_rows=p2,
        predictor_id="fixture",
        confidence=1.0,
        topology=PlanningTopology(world_size=4),
        constraints=PlanningConstraints(
            bucket_rows=1,
            max_waves=256,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(p3_return_weight=float(p3_weight)),
        information_mode=mode,
    )


def _flow_signature(plan):
    return tuple(
        (flow.phase, flow.src_rank, flow.dst_rank, flow.row_count)
        for wave in plan.waves
        for flow in wave.flows
    )


def test_p0123_derives_transposed_p3_but_executes_only_current_p0_p1() -> None:
    request = _planning_request(mode="p0_p1_p2_p3", p3_weight=1.0)
    assert request.p3_return_rows == _transpose(request.prediction_hint.target_dispatch_rows)
    plan = PlannerRegistry.create("current:p0123:joint:event:rscf", usage="runtime").plan(request)
    phases = {flow.phase for wave in plan.waves for flow in wave.flows}
    assert phases <= {"p0_dispatch", "p1_return", "p2_next_dispatch_forecast"}
    assert "p0_dispatch" in phases and "p1_return" in phases
    assert "p2_next_dispatch_forecast" in phases
    assert "p3_next_return_forecast" not in phases
    assert plan.metadata["execution_horizon"] == "p012"
    assert plan.metadata["p3_advisory_only"] is True


def test_p0123_weight_zero_is_execution_order_equivalent_to_frozen_p012() -> None:
    planner = PlannerRegistry.create("current:p0123:joint:event:rscf", usage="runtime")
    p012 = planner.plan(_planning_request(mode="p0_p1_p2", p3_weight=0.0))
    p0123_zero = planner.plan(_planning_request(mode="p0_p1_p2_p3", p3_weight=0.0))
    assert _flow_signature(p0123_zero) == _flow_signature(p012)


def test_p0123_return_advisory_can_change_current_p0_p1_plan() -> None:
    planner = PlannerRegistry.create("current:p0123:joint:event:rscf", usage="runtime")
    p012 = planner.plan(_planning_request(mode="p0_p1_p2", p3_weight=0.0))
    p0123 = planner.plan(_planning_request(mode="p0_p1_p2_p3", p3_weight=1.0))
    assert p0123.metadata["p3_effective"] is True
    assert p0123.metadata["kernel_artifact_digest"] != p012.metadata["kernel_artifact_digest"]
    assert float(p0123.metadata["kernel_makespan"]) >= 0.0


def test_p0123_rejects_non_transposed_p3() -> None:
    request = _planning_request(mode="p0_p1_p2_p3", p3_weight=1.0)
    malformed = type(request)(**{**request.__dict__, "p3_return_rows": request.prediction_hint.target_dispatch_rows})
    with pytest.raises(ValueError, match="transpose"):
        malformed.validate()


def _runtime(*, timing: str, horizon: str = "p012") -> RouterSenseInjectionRuntime:
    planner_id = (
        "future:p012:joint:global:rscf"
        if timing == "previous_layer"
        else "current:p0123:joint:global:rscf"
        if horizon == "p0123"
        else "current:p012:joint:global:rscf"
    )
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="prepared_priority",
            planner_id=planner_id,
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
            safe_projection_mode="disabled",
            observation_profile="execution",
            planning_horizon=horizon,
            planning_timing=timing,
            p3_return_weight=1.0 if horizon == "p0123" else 0.0,
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


def test_explicit_timing_modes_isolate_on_demand_and_future_p012() -> None:
    assert _runtime(timing="on_demand")._policy_supports_target_layer_preplanning() is False  # noqa: SLF001
    future = _runtime(timing="previous_layer")
    assert future._policy_supports_target_layer_preplanning() is True  # noqa: SLF001
    assert future._planning_information_mode() == "p0_p1_p2"  # noqa: SLF001
    p0123 = _runtime(timing="on_demand", horizon="p0123")
    assert p0123._policy_supports_target_layer_preplanning() is False  # noqa: SLF001
    assert p0123._planning_information_mode() == "p0_p1_p2_p3"  # noqa: SLF001


def _future_request() -> TargetLayerPlanningRequest:
    return TargetLayerPlanningRequest(
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
        current_p0_rows=((0, 2, 5), (3, 0, 3), (1, 5, 0)),
        previous_p0_rows=((0, 1, 4), (2, 0, 2), (1, 4, 0)),
        predictor_name="copy_current_dispatch",
        policy_id="future:p012:joint:global:rscf",
        joint_planner_id="future:p012:joint:global:rscf",
        local_planner_id="future:p012:local:global:rscf",
        safe_projection_mode="disabled",
        group_size=3,
        bucket_rows=0,
        policy_options=PlannerPolicyConfig(),
        topology_digest="topology",
        bucket_contract_digest="dynamic_current",
        information_mode="p0_p1_p2",
        planning_track="runtime_lookahead",
        planning_timing="previous_layer",
    )


def test_future_p012_reuses_exact_on_demand_p012_planner_output() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    built = service._build_target_plan(  # noqa: SLF001
        request=_future_request(),
        metrics=TargetLayerPlannerMetrics(),
    )
    on_demand = PlannerRegistry.create("current:p012:joint:global:rscf", usage="runtime").plan(built.planning_request)
    assert built.prepared_plan.window_plan is not None
    assert _flow_signature(built.prepared_plan.window_plan) == _flow_signature(on_demand)
    assert built.prepared_plan.window_plan.metadata["kernel_plan_digest"] == on_demand.metadata["kernel_plan_digest"]
    assert "future_prepared_order" in built.prepared_plan.window_plan.metadata
    assert built.prepared_plan.planning_horizon == "p012"
    assert built.prepared_plan.planning_timing == "previous_layer"
    assert built.prepared_plan.execution_scope == "target_p0_p1"
    assert built.prepared_plan.ready_at_ns >= built.prepared_plan.created_at_ns


def test_future_p012_metadata_survives_serialization() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    plan = service._build_target_plan(  # noqa: SLF001
        request=_future_request(),
        metrics=TargetLayerPlannerMetrics(),
    ).prepared_plan
    restored = TargetLayerPreparedJointPlan.from_dict(plan.to_dict())
    restored.validate()
    assert restored.planning_horizon == "p012"
    assert restored.planning_timing == "previous_layer"
    assert restored.execution_scope == "target_p0_p1"


def test_public_strategy_presets_keep_three_modes_distinct() -> None:
    p012 = resolve_strategy_runtime(strategy_name="routersense_current_p012_joint_global_rscf_async", runtime_line="async_release")
    p0123 = resolve_strategy_runtime(strategy_name="routersense_current_p0123_joint_global_rscf_async", runtime_line="async_release")
    future = resolve_strategy_runtime(strategy_name="routersense_future_p012_joint_global_rscf_async", runtime_line="async_release")
    assert (p012["planning_horizon"], p012["planning_timing"], p012["p3_return_weight"]) == ("p012", "on_demand", 0.0)
    assert (p0123["planning_horizon"], p0123["planning_timing"], p0123["p3_return_weight"]) == ("p0123", "on_demand", 0.01)
    assert (future["planning_horizon"], future["planning_timing"], future["p3_return_weight"]) == ("p012", "previous_layer", 0.0)
    assert p012["planner_id"] == "current:p012:joint:global:rscf"
    assert p0123["planner_id"] == "current:p0123:joint:global:rscf"
    assert future["planner_id"] == "future:p012:joint:global:rscf"


@pytest.mark.parametrize("core", ["gmwd", "rsbc", "rscf"])
@pytest.mark.parametrize("scope,engine", [("local", "event"), ("joint", "event"), ("joint", "global")])
def test_public_p012_strategy_exposes_formal_axes(core: str, scope: str, engine: str) -> None:
    strategy = resolve_strategy_runtime(
        strategy_name=f"routersense_current_p012_{scope}_{engine}_{core}_async",
        runtime_line="async_release",
    )
    assert strategy["planner_id"] == f"current:p012:{scope}:{engine}:{core}"
