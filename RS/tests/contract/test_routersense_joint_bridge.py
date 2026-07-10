from __future__ import annotations

from rs.runtime.online.megatron_ep.control.p2_provider import extract_prepared_plan_priority
from rs.scheduling import ForecastPressure, FlowDemand, FlowWindow, GlobalReadySetOptions, LogicalTopology, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem, PreparedWindowPlan, ReleaseConstraint, resolve_policy
from rs.scheduling.validation import stable_hash
from rs.runtime.online.megatron_ep.async_release import simulate_async_release


def _flows(matrix, phase, release_state, executable):
    result = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            result.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(result)


def _problem() -> MultiPhaseSchedulingProblem:
    p0 = (
        (0, 2, 8, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (8, 0, 0, 0),
    )
    p1 = (
        (0, 0, 0, 0),
        (0, 0, 0, 9),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    p2 = (
        (0, 4, 6, 0),
        (0, 0, 0, 2),
        (0, 0, 0, 0),
        (3, 0, 0, 0),
    )
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, "p0_dispatch", "ready", True),
            blocked_flows=_flows(p1, "p1_return", "blocked", False),
            forecast_pressure=_flows(p2, "p2_next_dispatch_forecast", "advisory_only", False),
        ),
        topology=LogicalTopology(num_gpus=4),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source="copy_current_dispatch",
            digest="f0",
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(4, 4),
            matrix_total_bytes=15,
            matrix=p2,
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def test_joint_bridge_policy_exposes_online_metadata() -> None:
    policy = resolve_policy(policy_name="routersense_joint_priority_phase_sync", bucket_rows=0)
    plan = policy.build_logical_plan(_problem())
    assert plan.diagnostics["policy_name"] == "routersense_joint_priority_phase_sync"
    assert plan.diagnostics["service_model"] == "phase_sync_joint_priority"
    assert plan.diagnostics["p2_source"] == "copy_current_dispatch"
    assert plan.diagnostics["priority_components"]["remote_only_matrix_invariant"] is True


def test_joint_bridge_is_diagonal_invariant() -> None:
    clean = _problem()
    dirty = MultiPhaseSchedulingProblem(
        flow_window=clean.flow_window,
        topology=clean.topology,
        release_model=clean.release_model,
        forecast=clean.forecast,
        options=clean.options,
        p0_dispatch_matrix=clean.p0_dispatch_matrix,
        p1_return_matrix=((99, 0, 0, 0), (0, 77, 0, 9), (0, 0, 55, 0), (0, 0, 0, 44)),
        p2_next_dispatch_forecast_matrix=((88, 4, 6, 0), (0, 66, 0, 2), (0, 0, 33, 0), (3, 0, 0, 22)),
    )
    policy = resolve_policy(policy_name="routersense_joint_priority_phase_sync", bucket_rows=0)
    clean_plan = policy.build_logical_plan(clean)
    dirty_plan = policy.build_logical_plan(dirty)
    assert stable_hash(clean_plan.to_dict()) == stable_hash(dirty_plan.to_dict())
    assert clean_plan.diagnostics["priority_components"]["remote_only_matrix_invariant"] is True
    assert dirty_plan.diagnostics["priority_components"]["remote_only_matrix_invariant"] is True


def test_async_release_joint_bridge_path_can_beat_phase_sync_birkhoff_on_sensitive_case() -> None:
    birkhoff = resolve_policy(policy_name="birkhoff_phase_local", bucket_rows=0).build_logical_plan(_problem())
    sim = simulate_async_release(
        p0_dispatch_matrix=(
            (0, 2, 8, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (8, 0, 0, 0),
        ),
        p1_return_matrix=(
            (0, 0, 0, 0),
            (0, 0, 0, 9),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        predicted_p2_matrix=(
            (0, 4, 6, 0),
            (0, 0, 0, 2),
            (0, 0, 0, 0),
            (3, 0, 0, 0),
        ),
        compute_delay=0.0,
        policy_name="routersense_joint_async_release",
    )
    assert float(sim["completion_time"]) < float(birkhoff.diagnostics["makespan"])


def test_prepared_plan_maps_logical_p2_to_next_layer_runtime_p0() -> None:
    prepared = PreparedWindowPlan(
        window_key="w0",
        forecast_digest="fd0",
        logical_plan=LogicalSchedulePlan(
            policy_name="routersense_multiphase_lookahead:p0_p1_p2",
            waves=(
                LogicalWave(wave_id=0, flows=(FlowDemand("p0_dispatch:0->1", "p0_dispatch", 0, 1, 8, "ready", True),)),
                LogicalWave(wave_id=1, flows=(FlowDemand("p2_next_dispatch:1->0", "p2_next_dispatch", 1, 0, 6, "ready", True),)),
            ),
            diagnostics={},
        ),
        created_at_layer_id="4",
        applies_from_layer_id="5",
        execution_capability_required="phase_sync",
    )
    payload = extract_prepared_plan_priority(prepared)
    assert payload["mapped_p2_edge_count"] == 1
    assert payload["stale_p0_p1_edge_count_ignored"] >= 1
    edge = payload["preferred_edges"][0]
    assert edge["phase"] == "P0"
    assert edge["origin_phase"] == "p2_next_dispatch"
    assert edge["source_layer_id"] == "4"
    assert edge["target_layer_id"] == "5"
