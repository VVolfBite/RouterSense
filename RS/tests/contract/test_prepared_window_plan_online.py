from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import torch

from rs.runtime.online.megatron_ep.contracts import ExecutionSelection, OnlinePolicyParameters, OnlineRuntimeConfig, RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.pending_window_policy import MultiphasePendingWindowPolicy
from rs.runtime.online.megatron_ep.multiphase_pending_window import build_pending_window_shadow
from rs.runtime.online.megatron_ep.observation import digest_text
from rs.runtime.online.megatron_ep.p2_contracts import P2HintRequest
from rs.runtime.online.megatron_ep.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.policy_adapter import compile_prepared_window_phase_plan
from rs.scheduling import resolve_phase_policy
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure, GlobalReadySetOptions, LogicalSchedulePlan, LogicalTopology, LogicalWave, MultiPhaseSchedulingProblem, PreparedWindowPlan, ReleaseConstraint
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.observation_contracts import RankTopologyRecord, RuntimeObservation
from rs.scheduling.phase_execution import FutureDemandHint
from rs.scheduling.validation import stable_hash

from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _request(*, rank: int = 0, layer_id: str = "1") -> P2HintRequest:
    return P2HintRequest(
        plan_key={"layer_id": layer_id, "phase": "P0"},
        layer_id=layer_id,
        phase="P0",
        global_rank=rank,
        local_rank=rank,
        ep_group_ranks=(0, 1),
    )


def _prepared_plan(*, forecast_digest: str = "forecast-abc", created: str = "0"):
    matrix = ((0, 8), (4, 0))
    problem = MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(),
        topology=LogicalTopology(num_gpus=2),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source="unit",
            digest=forecast_digest,
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(2, 2),
            matrix_total_bytes=12,
            matrix=matrix,
        ),
        options=GlobalReadySetOptions(scheduling_mode="runtime_lookahead", information_mode="p0_p1_p2", prediction_confidence=1.0),
        p0_dispatch_matrix=matrix,
        p1_return_matrix=matrix,
        p2_next_dispatch_forecast_matrix=matrix,
    )
    return RouterSenseMultiphaseLookaheadPolicy(
        information_mode="p0_p1_p2",
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    ).build_prepared_window_plan(problem=problem, created_at_layer_id=created, applies_from_layer_id=str(int(created) + 1))


def _manual_prepared_plan_with_priority(*, phase: str = "p0_dispatch", src_rank: int = 2, dst_rank: int = 0) -> PreparedWindowPlan:
    flow = FlowDemand(
        flow_id=f"{phase}:{src_rank}->{dst_rank}",
        phase=phase,
        src_rank=src_rank,
        dst_rank=dst_rank,
        byte_count=16,
        release_state="ready",
        is_executable=True,
    )
    logical_plan = LogicalSchedulePlan(
        policy_name="routersense_multiphase_lookahead:p0_p1_p2",
        waves=(LogicalWave(wave_id=0, flows=(flow,), duration=16.0),),
        diagnostics={"unit": "manual-prepared-plan"},
    )
    return PreparedWindowPlan(
        window_key="window-priority",
        forecast_digest="forecast-priority",
        logical_plan=logical_plan,
        created_at_layer_id="0",
        applies_from_layer_id="1",
        execution_capability_required="multiphase_pending_window",
    )


def _runtime(*, control_mode: str = "sync_before_phase") -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="phase_sync_wave",
            control_mode=control_mode,
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def _observation(*, layer_name: str, phase: str, per_peer_bytes: tuple[int, ...]) -> RuntimeObservation:
    return RuntimeObservation(
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        layer_id="0" if "layers.0" in layer_name else "1",
        layer_name=layer_name,
        global_rank=0,
        local_rank=0,
        node_id="node",
        device="cpu",
        ep_group_ranks=(0, 1),
        ep_group_size=2,
        dispatcher_class="MockDispatcher",
        expert_placement_hash="placement",
        model_revision_hash="model",
        dispatcher_hash="dispatcher",
        ep_group_hash="ep",
        request_table_hash="request",
        run_id_digest=digest_text("run"),
        step_id_digest=digest_text("step"),
        microbatch_id_digest=digest_text("mb"),
        phase=phase,
        per_peer_rows=tuple(1 if value else 0 for value in per_peer_bytes),
        per_peer_bytes=per_peer_bytes,
        local_rows=0,
        remote_rows=sum(1 for value in per_peer_bytes if value),
        topology=RankTopologyRecord(global_rank=0, local_rank=0, node_index=0, hostname_digest="host", device_index=0, ep_group_rank=0),
        input_splits=(0, 1),
        output_splits=(0, 1),
        observation_digest=stable_hash({"layer": layer_name, "phase": phase, "bytes": per_peer_bytes}),
    )


def test_calibrated_artifact_digest_determinism() -> None:
    prepared = _prepared_plan(forecast_digest="forecast-xyz")
    state = {"prepared_plan": prepared, "plan_created_at_us": 123, "plan_source_layer": "model.layers.0.mlp"}
    provider = build_p2_hint_provider("calibrated_artifact", shared_state=state)
    hints = [provider.build_hint(_request(rank=rank, layer_id="1")) for rank in (0, 1)]
    expected = hashlib.sha256(b"forecast-xyz:1").hexdigest()[:16]
    assert {hint.hint_digest for hint in hints} == {expected}
    assert all(hint.hint_mode == "calibrated_artifact" for hint in hints)
    assert all(hint.metadata["window_key"] == prepared.window_key for hint in hints)


def test_calibrated_artifact_exports_prepared_edge_priority() -> None:
    prepared = _manual_prepared_plan_with_priority(phase="p0_dispatch", src_rank=2, dst_rank=0)
    state = {"prepared_plan": prepared, "plan_created_at_us": 123, "plan_source_layer": "model.layers.0.mlp"}
    hint = build_p2_hint_provider("calibrated_artifact", shared_state=state).build_hint(
        P2HintRequest(
            plan_key={"layer_id": "1", "phase": "P0"},
            layer_id="1",
            phase="P0",
            global_rank=0,
            local_rank=0,
            ep_group_ranks=(0, 1, 2),
        )
    )
    assert hint.metadata["preferred_edges"] == [
        {
            "phase": "P0",
            "src_rank": 2,
            "dst_rank": 0,
            "priority": 0,
            "origin_phase": "p0_dispatch",
            "origin_flow_id": "p0_dispatch:2->0",
            "byte_count": 16,
            "wave_id": 0,
        }
    ]
    assert hint.metadata["preferred_wave_count"] == 1


def test_calibrated_artifact_no_plan_fallback() -> None:
    state = {"prepared_plan": None, "plan_created_at_us": 0, "plan_source_layer": ""}
    provider = build_p2_hint_provider("calibrated_artifact", shared_state=state)
    hint = provider.build_hint(_request())
    assert hint.hint_mode == "none"
    assert "no_prepared_plan_available" in hint.hint_source


def test_existing_p2_modes_unchanged() -> None:
    none_hint = build_p2_hint_provider("none").build_hint(_request())
    stub_hint = build_p2_hint_provider("deterministic_stub").build_hint(_request())
    assert none_hint.hint_mode == "none"
    assert none_hint.hint_digest == "none"
    assert stub_hint.hint_mode == "deterministic_stub"
    assert stub_hint.hint_digest == build_p2_hint_provider("deterministic_stub").build_hint(_request()).hint_digest


def test_layer_to_layer_plan_passing() -> None:
    runtime = _runtime()
    layer0 = "model.layers.0.mlp"
    runtime._pending_p0[layer0] = _observation(layer_name=layer0, phase="P0", per_peer_bytes=(0, 16))
    runtime._store_prepared_plan(layer_name=layer0, observation_p1=_observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24)))
    hint = runtime._build_p2_hint(layer_name="model.layers.1.mlp", phase="P0")
    assert hint.hint_mode == "calibrated_artifact"
    assert "model.layers.0.mlp" in hint.hint_source
    assert hint.metadata["source_layer"] == layer0


def test_plan_arrival_status_recording() -> None:
    runtime = _runtime(control_mode="default_continue")
    runtime._prepared_plan_state.update(
        {
            "prepared_plan": _prepared_plan(),
            "plan_created_at_us": int(time.time() * 1e6) - 200,
            "plan_source_layer": "model.layers.0.mlp",
        }
    )
    runtime._record_plan_arrival(layer_name="model.layers.1.mlp", phase="P0")
    records = runtime.export_plan_arrival_records()
    assert records
    assert records[-1]["arrival_status"] in {"before_commit", "in_flight"}
    assert records[-1]["plan_age_us"] >= 0
    assert records[-1]["has_prepared_plan"] is True


def test_p0p1p2_hint_evaluation_eligible_with_calibrated_artifact() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)), p2_hint_mode="none")
    hinted = tuple(
        replace(
            context,
            p2_hint=FutureDemandHint(
                hint_mode="calibrated_artifact",
                hint_digest="calibrated-digest",
                hint_source="calibrated_artifact_from_layer_model.layers.0.mlp",
            ),
        )
        for context in contexts
    )
    policy = resolve_phase_policy(policy_name="routersense_p0p1p2_hint", bucket_rows=0)
    plan = policy.build_plan(local_context=hinted[0], global_contexts=hinted)
    assert plan.metrics["evaluation_eligible"] is True
    assert plan.metrics["p2_hint_source"].startswith("calibrated_artifact_from_layer_")


def test_p0p1p2_hint_orders_by_prepared_edge_priority() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        p2_hint_mode="none",
    )
    prepared = _manual_prepared_plan_with_priority(phase="p0_dispatch", src_rank=2, dst_rank=0)
    state = {"prepared_plan": prepared, "plan_created_at_us": 123, "plan_source_layer": "model.layers.0.mlp"}
    hint = build_p2_hint_provider("calibrated_artifact", shared_state=state).build_hint(
        P2HintRequest(
            plan_key={"layer_id": "1", "phase": "P0"},
            layer_id="1",
            phase="P0",
            global_rank=0,
            local_rank=0,
            ep_group_ranks=(0, 1, 2),
        )
    )
    hinted = tuple(replace(context, p2_hint=hint) for context in contexts)
    plan = resolve_phase_policy(policy_name="routersense_p0p1p2_hint", bucket_rows=0).build_plan(
        local_context=hinted[0],
        global_contexts=hinted,
    )
    assert plan.metrics["ordered_by_prepared_plan"] is True
    assert plan.metrics["hint_edges_available"] == 1
    assert plan.metrics["hint_edges_consumed"] == 1
    assert plan.metrics["bucket_order"][0].startswith("P0:2->0:")


def test_compile_prepared_window_phase_plan_preserves_prepared_edge_order() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        p2_hint_mode="none",
    )
    prepared = _manual_prepared_plan_with_priority(phase="p0_dispatch", src_rank=2, dst_rank=0)
    plan = compile_prepared_window_phase_plan(
        prepared_plan=prepared,
        local_context=contexts[0],
        global_contexts=contexts,
        bucket_rows=0,
        p2_hint_weight=1.0,
    )
    assert plan.metrics["compiled_from_prepared_plan"] is True
    assert plan.metrics["prepared_window_key"] == "window-priority"
    assert plan.metrics["prepared_plan_order_preserved"] is True
    assert plan.metrics["bucket_order"][0].startswith("P0:2->0:")


def test_shared_state_thread_isolation() -> None:
    state_a = {"prepared_plan": _prepared_plan(forecast_digest="a"), "plan_created_at_us": 1, "plan_source_layer": "layer_a"}
    state_b = {"prepared_plan": _prepared_plan(forecast_digest="b"), "plan_created_at_us": 2, "plan_source_layer": "layer_b"}
    hint_a = build_p2_hint_provider("calibrated_artifact", shared_state=state_a).build_hint(_request(layer_id="1"))
    hint_b = build_p2_hint_provider("calibrated_artifact", shared_state=state_b).build_hint(_request(layer_id="1"))
    assert hint_a.hint_digest != hint_b.hint_digest
    assert hint_a.metadata["source_layer"] == "layer_a"
    assert hint_b.metadata["source_layer"] == "layer_b"


def test_attach_formal_online_runtime_enables_calibrated_p2_without_model_wrap() -> None:
    runtime = attach_formal_online_runtime(
        model=torch.nn.Module(),
        runtime_config=OnlineRuntimeConfig(
            policy_name="routersense_p0p1p2_hint",
            execution_mode="native_passthrough",
            control_mode="default_continue",
            execution_selection=ExecutionSelection(),
            policy_parameters=OnlinePolicyParameters(calibrated_p2_enabled=True, p2_hint_mode="none"),
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    assert runtime.config.p2_hint_mode == "calibrated_artifact"
    assert runtime._p2_hint_provider is not None


def test_window_state_and_release_events_are_recorded() -> None:
    runtime = _runtime()
    layer0 = "model.layers.0.mlp"
    p0 = _observation(layer_name=layer0, phase="P0", per_peer_bytes=(0, 16))
    p1 = _observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24))
    runtime._record_window_state(layer_name=layer0, p0_observation=p0)
    runtime._record_release_update(layer_name=layer0, event="p0_dispatch_completed")
    runtime._record_window_state(layer_name=layer0, p1_observation=p1)
    runtime._record_release_update(layer_name=layer0, event="p1_return_materialized")
    runtime._record_release_update(layer_name=layer0, event="p1_return_completed")

    state_rows = runtime.export_window_state_records()
    release_rows = runtime.export_release_events()
    shadow_rows = runtime.export_window_schedule_shadows()

    assert state_rows
    assert release_rows
    assert shadow_rows
    assert state_rows[-1]["has_p0_observation"] is True
    assert state_rows[-1]["has_p1_observation"] is True
    assert release_rows[-1]["event"] == "p1_return_completed"
    assert shadow_rows[-1]["shadow_policy_name"].startswith("routersense_multiphase_lookahead:")


def test_prepared_plan_binds_to_next_layer_shadow_window() -> None:
    runtime = _runtime()
    source_layer = "model.layers.0.mlp"
    target_layer = "model.layers.1.mlp"
    runtime._pending_p0[source_layer] = _observation(layer_name=source_layer, phase="P0", per_peer_bytes=(0, 10))
    runtime._store_prepared_plan(
        layer_name=source_layer,
        observation_p1=_observation(layer_name=source_layer, phase="P1", per_peer_bytes=(0, 20)),
    )
    runtime._record_window_state(
        layer_name=target_layer,
        p0_observation=_observation(layer_name=target_layer, phase="P0", per_peer_bytes=(0, 12)),
    )

    bindings = runtime.export_prepared_plan_bindings()
    shadows = runtime.export_window_schedule_shadows()

    assert bindings
    assert bindings[-1]["target_layer_id"] == "1"
    assert bindings[-1]["source_layer_name"] == source_layer
    assert shadows
    assert shadows[-1]["prepared_plan_bound"] is True
    assert shadows[-1]["prepared_window_key"] == bindings[-1]["window_key"]


def test_prepared_phase_plan_shadow_is_recorded_for_bound_layer() -> None:
    runtime = _runtime()
    source_layer = "model.layers.0.mlp"
    target_layer = "model.layers.1.mlp"
    runtime._pending_p0[source_layer] = _observation(layer_name=source_layer, phase="P0", per_peer_bytes=(0, 10))
    runtime._store_prepared_plan(
        layer_name=source_layer,
        observation_p1=_observation(layer_name=source_layer, phase="P1", per_peer_bytes=(0, 20)),
    )
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)), p2_hint_mode="none")
    runtime._record_prepared_phase_plan_shadow(
        layer_name=target_layer,
        phase="P0",
        local_context=contexts[0],
        global_contexts=contexts,
    )
    rows = runtime.export_prepared_phase_plan_shadows()
    assert rows
    assert rows[-1]["compile_status"] == "ok"
    assert rows[-1]["prepared_plan_order_preserved"] in {True, False}
    assert rows[-1]["prepared_window_key"]


def test_runtime_exports_planning_timing_records() -> None:
    runtime = _runtime()
    source_layer = "model.layers.0.mlp"
    target_layer = "model.layers.1.mlp"
    runtime._pending_p0[source_layer] = _observation(layer_name=source_layer, phase="P0", per_peer_bytes=(0, 10))
    runtime._store_prepared_plan(
        layer_name=source_layer,
        observation_p1=_observation(layer_name=source_layer, phase="P1", per_peer_bytes=(0, 20)),
    )
    runtime._record_plan_arrival(layer_name=target_layer, phase="P0")
    runtime._build_p2_hint(layer_name=target_layer, phase="P0")
    runtime._record_window_state(
        layer_name=target_layer,
        p0_observation=_observation(layer_name=target_layer, phase="P0", per_peer_bytes=(0, 12)),
    )

    rows = runtime.export_planning_timing_records()
    assert rows
    stages = {row["stage"] for row in rows}
    assert "store_prepared_plan" in stages
    assert "build_p2_hint" in stages
    assert "record_window_state" in stages
    assert all(float(row["duration_us"]) >= 0.0 for row in rows)


def test_pending_window_shadow_keeps_p1_blocked_until_p0_release() -> None:
    runtime = _runtime()
    layer0 = "model.layers.0.mlp"
    p0 = _observation(layer_name=layer0, phase="P0", per_peer_bytes=(0, 16))
    p1 = _observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24))
    runtime._record_window_state(layer_name=layer0, p0_observation=p0)
    runtime._record_window_state(layer_name=layer0, p1_observation=p1)
    state = runtime._window_states[layer0]
    snapshot = build_pending_window_shadow(
        state=state,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    )
    assert snapshot["ready_flow_count"] >= 1
    assert any(flow["phase"] == "p1_return" for flow in snapshot["blocked_flows"])
    assert not any(flow["phase"] == "p1_return" for flow in snapshot["ready_flows"])


def test_pending_window_shadow_releases_p1_after_p0_completion_and_materialization() -> None:
    runtime = _runtime()
    layer0 = "model.layers.0.mlp"
    p0 = _observation(layer_name=layer0, phase="P0", per_peer_bytes=(0, 16))
    p1 = _observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24))
    runtime._record_window_state(layer_name=layer0, p0_observation=p0)
    runtime._record_release_update(layer_name=layer0, event="p0_dispatch_completed")
    runtime._record_window_state(layer_name=layer0, p1_observation=p1)
    runtime._record_release_update(layer_name=layer0, event="p1_return_materialized")
    state = runtime._window_states[layer0]
    snapshot = build_pending_window_shadow(
        state=state,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    )
    assert any(flow["phase"] == "p1_return" for flow in snapshot["ready_flows"])
    assert snapshot["first_executable_wave"] is not None
    assert snapshot["first_executable_wave"]["selected_edges"]


def test_pending_window_shadow_never_marks_p2_forecast_executable() -> None:
    runtime = _runtime()
    source_layer = "model.layers.0.mlp"
    target_layer = "model.layers.1.mlp"
    runtime._pending_p0[source_layer] = _observation(layer_name=source_layer, phase="P0", per_peer_bytes=(0, 10))
    runtime._store_prepared_plan(
        layer_name=source_layer,
        observation_p1=_observation(layer_name=source_layer, phase="P1", per_peer_bytes=(0, 20)),
    )
    runtime._record_window_state(
        layer_name=target_layer,
        p0_observation=_observation(layer_name=target_layer, phase="P0", per_peer_bytes=(0, 12)),
    )
    state = runtime._window_states[target_layer]
    snapshot = build_pending_window_shadow(
        state=state,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    )
    assert snapshot["forecast_flow_count"] >= 0
    assert all(flow["phase"] != "p2_next_dispatch_forecast" for flow in snapshot["ready_flows"])
    assert snapshot["execution_capability_required"] == "multiphase_pending_window"


def test_multiphase_pending_window_policy_compiles_current_phase_plan() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    prepared = _prepared_plan(forecast_digest="forecast-online", created="0")
    policy = MultiphasePendingWindowPolicy(
        shared_state={
            "prepared_plan": prepared,
            "plan_created_at_us": 1,
            "plan_source_layer": "model.layers.0.mlp",
        },
        phase_policy_name="routersense_p0p1p2_hint",
        bucket_rows=0,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    )
    plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    assert plan.metrics["compiled_from_pending_window"] is True
    assert plan.metrics["pending_window_policy_name"].startswith("routersense_multiphase_lookahead:")
    assert plan.metrics["pending_window_information_mode"] in {"p0_p1", "p0_p1_p2"}


def test_multiphase_pending_window_policy_uses_prepared_p1_and_p2_matrices() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    flow_p1 = FlowDemand(
        flow_id="p1_return:1->0",
        phase="p1_return",
        src_rank=1,
        dst_rank=0,
        byte_count=11,
        release_state="blocked",
        is_executable=False,
    )
    flow_p2 = FlowDemand(
        flow_id="p2_next_dispatch_forecast:0->1",
        phase="p2_next_dispatch_forecast",
        src_rank=0,
        dst_rank=1,
        byte_count=7,
        release_state="advisory_only",
        is_executable=False,
    )
    prepared = PreparedWindowPlan(
        window_key="window-semantic",
        forecast_digest="forecast-semantic",
        logical_plan=LogicalSchedulePlan(
            policy_name="routersense_multiphase_lookahead:p0_p1_p2",
            waves=(LogicalWave(wave_id=0, flows=(flow_p1, flow_p2), duration=11.0),),
            diagnostics={},
        ),
        created_at_layer_id="0",
        applies_from_layer_id="1",
        execution_capability_required="multiphase_pending_window",
    )
    policy = MultiphasePendingWindowPolicy(
        shared_state={
            "prepared_plan": prepared,
            "plan_created_at_us": 1,
            "plan_source_layer": "model.layers.0.mlp",
        },
        phase_policy_name="routersense_p0p1p2_hint",
        bucket_rows=0,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    )
    plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    assert plan.metrics["pending_window_information_mode"] == "p0_p1_p2"
    assert plan.metrics["pending_window_p0_total_bytes"] == 40
    assert plan.metrics["pending_window_p1_total_bytes"] == 11
    assert plan.metrics["pending_window_p2_total_bytes"] == 7
    assert plan.metrics["pending_window_p1_matrix_source"] == "prepared_window_plan"
    assert plan.metrics["pending_window_p2_matrix_source"] == "prepared_window_plan"


def test_pending_window_driver_records_export_compiled_plan_metadata() -> None:
    runtime = _runtime()
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    prepared = _prepared_plan(forecast_digest="forecast-driver", created="0")
    plan = MultiphasePendingWindowPolicy(
        shared_state={
            "prepared_plan": prepared,
            "plan_created_at_us": 1,
            "plan_source_layer": "model.layers.0.mlp",
        },
        phase_policy_name="routersense_p0p1p2_hint",
        bucket_rows=0,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    ).build_plan(local_context=contexts[0], global_contexts=contexts)
    runtime.config = replace(runtime.config, execution_mode="multiphase_pending_window")
    runtime._record_pending_window_driver(layer_name="model.layers.1.mlp", phase="P0", plan=plan)
    rows = runtime.export_pending_window_driver_records()
    assert rows
    assert rows[-1]["compiled_from_pending_window"] is True
    assert rows[-1]["pending_window_policy_name"].startswith("routersense_multiphase_lookahead:")
    assert rows[-1]["wave_count"] == len(plan.waves)
