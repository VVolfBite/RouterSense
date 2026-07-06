from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import torch

from rs.runtime.online.megatron_ep.contracts import ExecutionSelection, OnlinePolicyParameters, OnlineRuntimeConfig, RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
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
