from __future__ import annotations

import pytest

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningTopology,
    PlanningWeights,
)
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.scheduling.families.core import FAMILY_KERNEL_SPECS
from rs.scheduling.p012_future._kernel.axes import (
    PlannerAxes,
    parse_planner_axes,
    planner_axis_matrix,
)
from rs.scheduling.p012_future._kernel.families import FAMILY_SPECS


def _transpose(matrix):
    return tuple(tuple(int(matrix[col][row]) for col in range(len(matrix))) for row in range(len(matrix)))


def _request(*, hint_scale: int = 1, mode: str = "p0_p1_p2"):
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
    hint = tuple(tuple(int(value) * int(hint_scale) for value in row) for row in p2)
    return build_window_planning_request(
        identity=PlanningIdentity(request_id=f"axes:{hint_scale}:{mode}"),
        p0_dispatch_rows=p0,
        p1_return_rows=_transpose(p0),
        p2_hint_rows=hint,
        predictor_id="fixture",
        confidence=1.0 if hint_scale else 0.0,
        topology=PlanningTopology(world_size=4),
        constraints=PlanningConstraints(
            bucket_rows=1,
            max_waves=256,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(p3_return_weight=1.0 if mode == "p0_p1_p2_p3" else 0.0),
        information_mode=mode,
    )


def _flow_signature(plan):
    return tuple(
        (flow.phase, flow.src_rank, flow.dst_rank, flow.row_count)
        for wave in plan.waves
        for flow in wave.flows
    )


def test_axes_parser_preserves_legacy_semantics() -> None:
    assert parse_planner_axes("p012:local:rscf") == PlannerAxes(
        "current", "p012", "local", "event", "rscf"
    )
    assert parse_planner_axes("p012:global:rscf") == PlannerAxes(
        "current", "p012", "joint", "global", "rscf"
    )
    assert parse_planner_axes("future_prepared:event:rsbc") == PlannerAxes(
        "future", "p012", "joint", "event", "rsbc"
    )
    assert parse_planner_axes("current:p012:local:global:gmwd").canonical_id == (
        "current:p012:local:global:gmwd"
    )


def test_axis_matrix_exposes_complete_supported_cross_product() -> None:
    rows = planner_axis_matrix()
    assert len(rows) == 36
    assert PlannerAxes("current", "p012", "local", "global", "rscf") in rows
    assert PlannerAxes("current", "p0123", "joint", "event", "rsbc") in rows
    assert PlannerAxes("future", "p012", "local", "event", "gmwd") in rows
    assert all(not (row.timing == "future" and row.horizon == "p0123") for row in rows)


def test_p012_runtime_uses_shared_family_spec_authority() -> None:
    for core in ("gmwd", "rsbc", "rscf"):
        assert FAMILY_SPECS[core] is FAMILY_KERNEL_SPECS[core]
        assert len(FAMILY_SPECS[core].p012_runtime_weights()) == 4


@pytest.mark.parametrize("core", ["gmwd", "rsbc", "rscf"])
def test_explicit_current_axes_preserve_legacy_plans(core: str) -> None:
    request = _request()
    pairs = (
        (f"p012:local:{core}", f"current:p012:local:event:{core}"),
        (f"p012:event:{core}", f"current:p012:joint:event:{core}"),
        (f"p012:global:{core}", f"current:p012:joint:global:{core}"),
    )
    for legacy, explicit in pairs:
        legacy_plan = PlannerRegistry.create(legacy, usage="runtime").plan(request)
        explicit_plan = PlannerRegistry.create(explicit, usage="runtime").plan(request)
        assert _flow_signature(explicit_plan) == _flow_signature(legacy_plan)
        assert explicit_plan.metadata["planner_axes"] == parse_planner_axes(explicit).to_dict()


def test_local_global_is_explicit_and_prediction_independent() -> None:
    planner = PlannerRegistry.create("current:p012:local:global:rscf", usage="runtime")
    with_hint = planner.plan(_request(hint_scale=1))
    without_hint = planner.plan(_request(hint_scale=0))
    assert _flow_signature(with_hint) == _flow_signature(without_hint)
    assert with_hint.planner_family == "local"
    assert with_hint.metadata["scope"] == "local"
    assert with_hint.metadata["engine"] == "global"


@pytest.mark.parametrize("scope", ["local", "joint"])
@pytest.mark.parametrize("engine", ["event", "global"])
def test_future_is_only_a_timing_wrapper(scope: str, engine: str) -> None:
    request = _request()
    current = PlannerRegistry.create(
        f"current:p012:{scope}:{engine}:rscf", usage="runtime"
    ).plan(request)
    future = PlannerRegistry.create(
        f"future:p012:{scope}:{engine}:rscf", usage="runtime"
    ).plan(request)
    assert _flow_signature(future) == _flow_signature(current)
    assert future.metadata["planning_timing"] == "previous_layer"
    assert future.metadata["planner_axes"]["timing"] == "future"
    assert "future_prepared_order" in future.metadata


@pytest.mark.parametrize("engine", ["event", "global"])
def test_local_p0123_is_strict_baseline_and_does_not_consume_p3(engine: str) -> None:
    p012 = PlannerRegistry.create(
        f"current:p012:local:{engine}:rsbc", usage="runtime"
    ).plan(_request(mode="p0_p1_p2"))
    p0123 = PlannerRegistry.create(
        f"current:p0123:local:{engine}:rsbc", usage="runtime"
    ).plan(_request(mode="p0_p1_p2_p3"))
    assert _flow_signature(p0123) == _flow_signature(p012)
    assert p0123.metadata["p3_effective"] is False


def _runtime_for_axes(planner_id: str, *, timing: str, horizon: str = "p012"):
    from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
    from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime

    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            planner_id=planner_id,
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
            safe_projection_mode="disabled",
            observation_profile="execution",
            planning_horizon=horizon,
            planning_timing=timing,
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


def test_runtime_safe_pair_changes_only_scope() -> None:
    planner_id = "future:p012:joint:global:rscf"
    runtime = _runtime_for_axes(planner_id, timing="previous_layer")
    assert runtime._runtime_safe_joint_pair(planner_id) == (  # noqa: SLF001
        planner_id,
        "future:p012:local:global:rscf",
    )
    assert runtime._current_window_planner_id() == "current:p012:joint:global:rscf"  # noqa: SLF001


def test_explicit_runtime_strategy_resolves_all_axes() -> None:
    from rs.experiments_support.runtime_presets import resolve_strategy_runtime

    row = resolve_strategy_runtime(
        strategy_name="routersense_future_p012_local_global_rsbc_async",
        runtime_line="async_release",
    )
    assert row["planner_id"] == "future:p012:local:global:rsbc"
    assert (row["planner_timing"], row["planner_scope"], row["planner_engine"], row["planner_core"]) == (
        "future",
        "local",
        "global",
        "rsbc",
    )


def test_safe_wrapper_accepts_strict_explicit_pair() -> None:
    planner = PlannerRegistry.create(
        "safe_pair",
        {
            "joint_planner_id": "current:p012:joint:global:rscf",
            "local_planner_id": "current:p012:local:global:rscf",
        },
        usage="runtime",
    )
    plan = planner.plan(_request())
    assert plan.planner_id == "safe_pair"
    diagnostics = dict(plan.metadata)
    assert diagnostics["joint_planner_id"] == "current:p012:joint:global:rscf"
    assert diagnostics["local_planner_id"] == "current:p012:local:global:rscf"


def test_current_local_runtime_strategy_disables_prediction() -> None:
    from rs.experiments_support.runtime_presets import resolve_strategy_runtime

    row = resolve_strategy_runtime(
        strategy_name="routersense_current_p012_local_global_rscf_async",
        runtime_line="async_release",
    )
    assert row["planner_id"] == "current:p012:local:global:rscf"
    assert row["online_p2_predictor"] == "none"
    assert row["p2_hint_mode"] == "none"
    assert row["calibrated_p2"] is False
