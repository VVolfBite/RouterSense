from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from rs.runtime.online.megatron_ep.contracts import ExecutionSelection, OnlinePolicyParameters, OnlineRuntimeConfig, RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.control.p2_matrix import build_traffic_matrix_bundle, gather_global_peer_bytes_matrix
from rs.runtime.online.megatron_ep.pending_window import MultiphasePendingWindowAdapter, build_pending_window_shadow
from rs.runtime.online.megatron_ep.observation import digest_text
from rs.runtime.online.megatron_ep.observation.views import scheduled_plan_artifact
from rs.runtime.online.megatron_ep.control.p2_contracts import P2HintRequest
from rs.runtime.online.megatron_ep.control.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.pending_window import compile_prepared_window_phase_plan
from rs.scheduling.validation import validate_phase_execution_plan
from rs.scheduling import resolve_phase_policy
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure, GlobalReadySetOptions, LogicalSchedulePlan, LogicalTopology, LogicalWave, MultiPhaseSchedulingProblem, PreparedWindowPlan, ReleaseConstraint
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.observation_contracts import RankTopologyRecord, RuntimeObservation
from rs.scheduling.phase_execution import FutureDemandHint
from rs.scheduling.phase_local.common import build_transfer_layouts_and_tasks
from rs.scheduling.validation import stable_hash

from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _seed_prediction(runtime: RouterSenseInjectionRuntime, *, source_layer_id: str, target_layer_id: str, matrix: tuple[tuple[int, ...], ...]) -> None:
    payload = {
        "predictor_name": "copy_current_dispatch",
        "predictor_version": "v1",
        "source_layer_id": str(source_layer_id),
        "predicted_layer_id": str(target_layer_id),
        "matrix": [list(row) for row in matrix],
        "matrix_digest": stable_hash([list(row) for row in matrix]),
        "total_bytes": int(sum(sum(int(value) for value in row) for row in matrix)),
        "nonzero_edge_count": int(sum(1 for row in matrix for value in row if int(value) > 0)),
        "confidence": 1.0,
        "is_oracle": False,
        "evaluation_eligible": True,
        "created_at_phase": "P0",
    }
    predicted_dispatch_by_layer = dict(runtime._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
    predicted_dispatch_by_layer[str(target_layer_id)] = payload
    runtime._runtime_state.write("predicted_dispatch_by_layer", predicted_dispatch_by_layer)
    runtime._runtime_state.write(
        "active_next_dispatch_prediction",
        {
            "source_layer_id": str(source_layer_id),
            "target_layer_id": str(target_layer_id),
            "forecast_matrix": [list(row) for row in matrix],
            "matrix_digest": payload["matrix_digest"],
            "predictor_name": "copy_current_dispatch",
            "predictor_version": "v1",
            "confidence": 1.0,
            "evaluation_eligible": True,
            "is_oracle": False,
            "created_at_phase": "P0",
            "created_at_stage": "test_seed",
            "prediction_time_us": 0.0,
            "valid": True,
            "error": "",
        },
    )


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


def _runtime(*, control_mode: str = "sync_before_phase", observation_profile: str = "minimal") -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="phase_sync_wave",
            control_mode=control_mode,
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
            observation_profile=observation_profile,
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


def _async_runtime() -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="calibrated_artifact",
            p2_hint_weight=1.0,
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


def _phase_ctx_and_pretransport(*, matrix: tuple[tuple[int, ...], ...], layer_name: str = "model.layers.0.mlp"):
    contexts = make_contexts_from_matrix(phase="P0", matrix=matrix, p2_hint_mode="none")
    runtime = _async_runtime()
    observation = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    return runtime, contexts[0], observation, tuple(tuple(int(v) for v in row) for row in matrix)


def test_calibrated_artifact_digest_determinism() -> None:
    prepared = _prepared_plan(forecast_digest="forecast-xyz")
    state = {"prepared_plan": prepared, "plan_created_at_us": 123, "plan_source_layer": "model.layers.0.mlp"}
    provider = build_p2_hint_provider("calibrated_artifact", shared_state=state)
    hints = [provider.build_hint(_request(rank=rank, layer_id="1")) for rank in (0, 1)]
    expected = hashlib.sha256(b"forecast-xyz:1").hexdigest()[:16]
    assert {hint.hint_digest for hint in hints} == {expected}
    assert all(hint.hint_mode == "calibrated_artifact" for hint in hints)
    assert all(hint.metadata["window_key"] == prepared.window_key for hint in hints)


def test_forward_epoch_clears_stale_prediction_state() -> None:
    runtime = _runtime()
    runtime.begin_forward(forward_epoch=1)
    runtime._prepared_plan_state["active_next_dispatch_prediction"] = {  # noqa: SLF001
        "source_layer_id": "4",
        "target_layer_id": "5",
        "forecast_matrix": ((0, 3), (4, 0)),
    }
    runtime._prepared_plan_state["prediction_consumption_records"] = [{"consumer_layer": "5"}]  # noqa: SLF001
    runtime.begin_forward(forward_epoch=2)
    summary = runtime.export_prepared_plan_summary()
    assert summary["prediction_source_layer"] == ""
    assert summary["prediction_target_layer"] == ""
    assert summary["consumed_before_p1"] is False
    assert runtime._plan_key("model.layers.0.mlp", "P0")["forward_epoch"] == 2  # noqa: SLF001


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
    assert hint.metadata["preferred_edges"] == []
    assert hint.metadata["stale_p0_p1_edge_count_ignored"] == 1
    assert hint.metadata["stale_prepared_edges"][0]["origin_phase"] == "p0_dispatch"
    assert hint.metadata["preferred_wave_count"] == 0


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


def test_prediction_audit_exports_after_next_dispatch_arrives() -> None:
    runtime = _runtime(observation_profile="debug")
    contexts0 = make_contexts_from_matrix(phase="P0", matrix=((0, 1), (0, 0)), p2_hint_mode="none")
    pre0 = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts0[0])  # noqa: SLF001
    runtime._record_prediction_for_dispatch(
        layer_name="model.layers.0.mlp",
        phase_ctx=contexts0[0],
        observation=pre0,
        actual_p0_full_row_matrix=((0, 1), (0, 0)),
        device=torch.device("cpu"),
    )
    _seed_prediction(runtime, source_layer_id="0", target_layer_id="1", matrix=((0, 1), (0, 0)))
    contexts1 = make_contexts_from_matrix(phase="P0", matrix=((0, 1), (0, 0)), p2_hint_mode="none")
    pre1 = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts1[0])  # noqa: SLF001
    runtime._record_prediction_for_dispatch(
        layer_name="model.layers.1.mlp",
        phase_ctx=contexts1[0],
        observation=pre1,
        actual_p0_full_row_matrix=((0, 1), (0, 0)),
        device=torch.device("cpu"),
    )
    rows = runtime.export_prediction_audits()
    assert rows
    assert rows[-1]["predictor_name"] == "copy_current_dispatch"
    assert "relative_l1_error" in rows[-1]


def test_joint_window_async_p0_stores_joint_plan_and_compiles_local_plan() -> None:
    runtime = _async_runtime()
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 16), (8, 0)), p2_hint_mode="none")
    pre = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    runtime._record_prediction_for_dispatch(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=contexts[0],
        observation=pre,
        actual_p0_full_row_matrix=((0, 16), (8, 0)),
        device=torch.device("cpu"),
    )
    _seed_prediction(runtime, source_layer_id="0", target_layer_id="1", matrix=((0, 16), (8, 0)))
    runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=contexts[0],
        observation_p0=pre,
        actual_p0_full_row_matrix=((0, 16), (8, 0)),
    )
    plan = runtime._compile_async_local_phase_plan(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase="P0",
        local_context=contexts[0],
    )
    assert plan.execution_mode == "joint_window_async_p2p"
    assert plan.metrics["prediction_extra_collective_count"] == 0
    assert runtime.export_prepared_plan_summary()["prediction_target_layer"] == "1"


def test_joint_window_async_p1_reuses_prepared_plan_without_planning_collective() -> None:
    runtime = _async_runtime()
    runtime._prepared_plan_state["prepared_plan"] = _prepared_plan(created="0")  # noqa: SLF001
    runtime._prepared_plan_state["p1_inferred_from_p0"] = [[0, 8], [4, 0]]  # noqa: SLF001
    contexts = make_contexts_from_matrix(phase="P1", matrix=((0, 4), (8, 0)), p2_hint_mode="none")
    plan = runtime._compile_async_local_phase_plan(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase="P1",
        local_context=contexts[0],
    )
    assert plan.execution_mode == "joint_window_async_p2p"
    assert plan.metrics["p1_planning_collective_count"] == 0


def test_pretransport_observation_uses_phase_ready_context_splits() -> None:
    runtime = _async_runtime()
    contexts = make_contexts_from_matrix(phase="P0", matrix=((1, 2), (3, 1)), p2_hint_mode="none")
    observation = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    assert observation.source == "phase_ready_context_dispatcher_splits"
    assert observation.captured_before_transport is True
    assert observation.valid is True
    assert observation.send_splits_rows == contexts[0].send_splits
    assert observation.recv_splits_rows == contexts[0].recv_splits
    assert observation.local_p0_row == contexts[0].send_splits


def test_zero_matrix_with_nonzero_splits_fails_fast(tmp_path: Path, monkeypatch) -> None:
    runtime = _async_runtime()
    runtime.config = replace(runtime.config, executor_heartbeat_path=str(tmp_path))
    runtime.ep_process_group = object()
    contexts = make_contexts_from_matrix(phase="P0", matrix=((1, 2), (3, 1)), p2_hint_mode="none")
    observation = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001

    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle.dist.is_available", lambda: True)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle.dist.is_initialized", lambda: True)

    def _zero_all_gather(out_list, local_tensor, group=None):
        for item in out_list:
            item.zero_()

    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle.dist.all_gather", _zero_all_gather)

    try:
        runtime._gather_actual_p0_full_row_matrix(  # noqa: SLF001
            layer_name="model.layers.0.mlp",
            observation=observation,
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "traffic_source_mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected traffic_source_mismatch")

    artifact = tmp_path / "traffic_source_mismatch_rank0.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["dispatcher_send_splits"] == [1, 2]
    assert payload["phase_ready_context_send_splits"] == [1, 2]
    assert payload["global_p0_matrix"] == [[0, 0], [0, 0]]


def test_transfer_layouts_preserve_receiver_offsets_for_async_p2p() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (1, 0)), p2_hint_mode="none")
    transfer_layouts, tasks = build_transfer_layouts_and_tasks(
        local_context=contexts[0],
        global_contexts=contexts,
        bucket_rows=1,
    )
    remote_tasks = [task for task in tasks if int(task.src_rank) != int(task.dst_rank)]
    recv_offsets = {(int(task.src_rank), int(task.dst_rank), int(task.receiver_offset_rows)) for task in remote_tasks}
    assert (1, 0, 0) in recv_offsets
    assert (0, 1, 0) in recv_offsets or (0, 1, 1) in recv_offsets


def test_perf_profile_suppresses_window_shadow_exports() -> None:
    runtime = _runtime(observation_profile="perf")
    runtime.window_state_records.append({"window": "x"})
    runtime.prepared_plan_bindings.append({"binding": "x"})
    runtime.release_events.append({"event": "x"})
    runtime.window_schedule_shadows.append({"shadow": "x"})
    runtime.prepared_phase_plan_shadows.append({"prepared": "x"})
    assert runtime.export_window_state_records() == []
    assert runtime.export_prepared_plan_bindings() == []
    assert runtime.export_release_events() == []
    assert runtime.export_window_schedule_shadows() == []
    assert runtime.export_prepared_phase_plan_shadows() == []


def test_debug_profile_keeps_window_shadow_exports() -> None:
    runtime = _runtime(observation_profile="debug")
    runtime.window_state_records.append({"window": "x"})
    runtime.prepared_plan_bindings.append({"binding": "x"})
    runtime.release_events.append({"event": "x"})
    runtime.window_schedule_shadows.append({"shadow": "x"})
    runtime.prepared_phase_plan_shadows.append({"prepared": "x"})
    assert runtime.export_window_state_records() == [{"window": "x"}]
    assert runtime.export_prepared_plan_bindings() == [{"binding": "x"}]
    assert runtime.export_release_events() == [{"event": "x"}]
    assert runtime.export_window_schedule_shadows() == [{"shadow": "x"}]
    assert runtime.export_prepared_phase_plan_shadows() == [{"prepared": "x"}]


def test_prepared_plan_summary_exposes_p2_matrix_source() -> None:
    runtime = _runtime(observation_profile="perf")
    runtime._prepared_plan_state.update(
        {
            "prepared_plan": _prepared_plan(),
            "plan_created_at_us": 123,
            "plan_source_layer": "model.layers.0.mlp",
            "p2_matrix_source": "predicted_next_dispatch",
            "p2_matrix_total_bytes": 64,
            "p2_matrix_row_sums": [32, 32],
            "p2_matrix_col_sums": [16, 48],
            "p2_matrix_is_replicated_local_row": False,
            "p2_matrix_shape": [2, 2],
            "p2_matrix_gather_time_us": 12.5,
            "p2_matrix_gather_status": "tensor_all_gather",
            "p2_matrix_gather_call_count": 2,
            "predictor_name": "copy_current_dispatch",
            "prediction_digest": "pred-digest",
        }
    )
    summary = runtime.export_prepared_plan_summary()
    assert summary["has_prepared_plan"] is True
    assert summary["p2_matrix_source"] == "predicted_next_dispatch"
    assert summary["p2_matrix_total_bytes"] == 64
    assert summary["p2_matrix_is_replicated_local_row"] is False
    assert summary["p2_matrix_shape"] == [2, 2]
    assert summary["p2_matrix_gather_time_us"] == 12.5
    assert summary["p2_matrix_gather_status"] == "tensor_all_gather"
    assert summary["p2_matrix_gather_call_count"] == 2
    assert summary["predictor_name"] == "copy_current_dispatch"
    assert summary["prediction_digest"] == "pred-digest"


def test_gather_global_peer_bytes_matrix_falls_back_without_dist(monkeypatch) -> None:
    from rs.runtime.online.megatron_ep.control import p2_matrix as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: False)
    local = mod.build_local_peer_bytes_tensor((0, 24), 2, "cpu")
    matrix, metadata = gather_global_peer_bytes_matrix(local)
    assert tuple(tuple(int(value) for value in row) for row in matrix.tolist()) == ((0, 24), (0, 0))
    assert metadata["matrix_source"] == "replicated_local_row_fallback"
    assert metadata["gather_call_count"] == 0


def test_build_traffic_matrix_bundle_uses_tensor_collective(monkeypatch) -> None:
    from rs.runtime.online.megatron_ep.control import p2_matrix as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mod.dist, "all_gather_object", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("object collective forbidden")))

    def all_gather_into_tensor(output, input_tensor, group=None):
        values = torch.tensor([0, 24, 12, 0], dtype=input_tensor.dtype, device=input_tensor.device)
        output.copy_(values)

    monkeypatch.setattr(mod.dist, "all_gather_into_tensor", all_gather_into_tensor)
    bundle = build_traffic_matrix_bundle(per_peer_bytes=(0, 24), world_size=2, device="cpu", group=None)
    assert bundle.matrix_source == "tensor_all_gather"
    assert bundle.is_global is True
    assert bundle.matrix == ((0, 24), (12, 0))
    assert bundle.total_bytes == 36
    assert bundle.gather_call_count == 1


def test_store_prepared_plan_prefers_predicted_next_dispatch(monkeypatch) -> None:
    runtime = _runtime(observation_profile="perf")
    from rs.runtime.online.megatron_ep.control import p2_matrix as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)

    def all_gather_into_tensor(output, input_tensor, group=None):
        if input_tensor.tolist() == [0, 24]:
            output.copy_(torch.tensor([0, 24, 12, 0], dtype=input_tensor.dtype, device=input_tensor.device))
        else:
            output.copy_(torch.tensor([0, 16, 8, 0], dtype=input_tensor.dtype, device=input_tensor.device))

    monkeypatch.setattr(mod.dist, "all_gather_into_tensor", all_gather_into_tensor)
    layer0 = "model.layers.0.mlp"
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 16), (8, 0)), p2_hint_mode="none")
    pre = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    runtime._record_prediction_for_dispatch(
        layer_name=layer0,
        phase_ctx=contexts[0],
        observation=pre,
        actual_p0_full_row_matrix=((0, 16), (8, 0)),
        device=torch.device("cpu"),
    )
    _seed_prediction(runtime, source_layer_id="0", target_layer_id="1", matrix=((0, 16), (8, 0)))
    runtime._store_prepared_plan(layer_name=layer0, observation_p1=_observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24)))
    summary = runtime.export_prepared_plan_summary()
    assert summary["p2_matrix_source"] == "active_next_dispatch_prediction"
    assert summary["p2_matrix_is_replicated_local_row"] is False
    assert summary["p2_matrix_row_sums"] == [16, 8]
    assert summary["p2_matrix_col_sums"] == [8, 16]
    assert summary["p2_matrix_gather_status"] == "tensor_all_gather"
    assert summary["p2_matrix_gather_call_count"] == 1
    assert summary["p2_matrix_gather_time_us"] >= 0.0
    assert summary["predictor_name"] == "copy_current_dispatch"


def test_runtime_joint_plan_records_host_projected_safe_selection(monkeypatch) -> None:
    runtime = _async_runtime()
    from rs.runtime.online.megatron_ep.control import p2_matrix as mod

    monkeypatch.setattr(mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(mod.dist, "is_initialized", lambda: True)

    def all_gather_into_tensor(output, input_tensor, group=None):
        output.copy_(torch.tensor([0, 16, 8, 0], dtype=input_tensor.dtype, device=input_tensor.device))

    monkeypatch.setattr(mod.dist, "all_gather_into_tensor", all_gather_into_tensor)
    layer0 = "model.layers.0.mlp"
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 16), (8, 0)), p2_hint_mode="none")
    pre = runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    runtime._record_prediction_for_dispatch(  # noqa: SLF001
        layer_name=layer0,
        phase_ctx=contexts[0],
        observation=pre,
        actual_p0_full_row_matrix=((0, 16), (8, 0)),
        device=torch.device("cpu"),
    )
    runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name=layer0,
        phase_ctx=contexts[0],
        observation_p0=pre,
        actual_p0_full_row_matrix=((0, 16), (8, 0)),
    )
    summary = runtime.export_prepared_plan_summary()
    assert summary["safe_selected_policy"] != ""
    assert summary["raw_u_policy_name"].startswith("U_")
    assert summary["paired_b_policy_name"].startswith("B_")
    assert summary["host_projected_estimated_makespan"] >= 0.0


def test_store_prepared_plan_legacy_fallback_is_not_marked_as_predictor_matrix() -> None:
    runtime = _runtime(observation_profile="perf")
    layer0 = "model.layers.0.mlp"
    runtime._prepared_plan_state["actual_dispatch_by_layer"] = {
        "0": {
            "matrix": [[0, 16], [8, 0]],
            "matrix_digest": "dispatch-0",
            "matrix_source": "replicated_local_row_fallback",
            "total_bytes": 24,
            "nonzero_edge_count": 2,
        }
    }
    runtime._store_prepared_plan(layer_name=layer0, observation_p1=_observation(layer_name=layer0, phase="P1", per_peer_bytes=(0, 24)))
    summary = runtime.export_prepared_plan_summary()
    assert summary["p2_matrix_source"] == "copy_current_dispatch_fallback"
    assert summary["predictor_name"] == "copy_current_dispatch"


def test_perf_scheduled_plan_artifact_keeps_task_ids_only() -> None:
    runtime = _runtime(observation_profile="perf")
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)), p2_hint_mode="none")
    policy = resolve_phase_policy(policy_name="birkhoff_phase_local", bucket_rows=0)
    plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    artifact = scheduled_plan_artifact(plan=plan, perf_profile=runtime._is_perf_profile())
    assert artifact["waves"]
    first_wave = artifact["waves"][0]
    assert "task_ids" in first_wave
    assert "bucket_tasks" not in first_wave
    assert isinstance(first_wave["task_ids"], list)


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
    assert plan.metrics["ordered_by_prepared_plan"] is False
    assert plan.metrics["hint_edges_available"] == 0
    assert plan.metrics["hint_edges_consumed"] == 0


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
    assert plan.metrics["prepared_plan_order_preserved"] is False
    assert plan.metrics["bucket_order"][0].startswith("P0:")


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


class _TokenDispatcherStub:
    def __init__(self) -> None:
        self.token_dispatch = lambda *args, **kwargs: ("dispatch", args, kwargs)
        self.token_combine = lambda *args, **kwargs: ("combine", args, kwargs)


class _MoELayerStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_dispatcher = _TokenDispatcherStub()


class _ScopeModelStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_MoELayerStub(), _MoELayerStub(), _MoELayerStub()])


def test_runtime_hook_scope_resolves_selected_prediction_source_and_none() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            selected_layer_ids=("1",),
            schedule_layer_selector="selected",
            safe_projection_mode="disabled",
            online_p2_predictor="copy_current_dispatch",
            observation_profile="perf",
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1, 2, 3),
        ep_group_root_global_rank=0,
    )
    runtime.configure_hook_scope(
        available_layer_names=("model.layers.0.mlp", "model.layers.1.mlp", "model.layers.2.mlp")
    )
    assert runtime.layer_role_for_name("model.layers.0.mlp") == "prediction_source"
    assert runtime.layer_role_for_name("model.layers.1.mlp") == "selected"
    assert runtime.layer_role_for_name("model.layers.2.mlp") == "none"
    summary = runtime.export_prepared_plan_summary()
    assert summary["selected_layer_ids"] == ["1"]
    assert summary["prediction_source_layer_ids"] == ["0"]
    assert summary["none_layer_ids"] == ["2"]


def test_attach_formal_online_runtime_wraps_only_selected_and_prediction_source_layers() -> None:
    model = _ScopeModelStub()
    layer0_dispatch = model.layers[0].token_dispatcher.token_dispatch
    layer0_combine = model.layers[0].token_dispatcher.token_combine
    layer1_dispatch = model.layers[1].token_dispatcher.token_dispatch
    layer1_combine = model.layers[1].token_dispatcher.token_combine
    layer2_dispatch = model.layers[2].token_dispatcher.token_dispatch
    layer2_combine = model.layers[2].token_dispatcher.token_combine
    attach_formal_online_runtime(
        model=model,
        runtime_config=OnlineRuntimeConfig(
            policy_name="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            execution_selection=ExecutionSelection(
                layer_selector="selected",
                selected_layer_ids=("1",),
            ),
            policy_parameters=OnlinePolicyParameters(
                online_p2_predictor="copy_current_dispatch",
                safe_projection_mode="disabled",
            ),
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    assert model.layers[0].token_dispatcher.token_dispatch is not layer0_dispatch
    assert model.layers[0].token_dispatcher.token_combine is layer0_combine
    assert model.layers[1].token_dispatcher.token_dispatch is not layer1_dispatch
    assert model.layers[1].token_dispatcher.token_combine is not layer1_combine
    assert model.layers[2].token_dispatcher.token_dispatch is layer2_dispatch
    assert model.layers[2].token_dispatcher.token_combine is layer2_combine


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


def test_current_plan_build_is_deduplicated_per_rank_epoch_layer() -> None:
    runtime, context, observation, matrix = _phase_ctx_and_pretransport(matrix=((0, 2), (3, 0)))
    runtime.configure_hook_scope(available_layer_names=("model.layers.0.mlp", "model.layers.1.mlp"))
    runtime.begin_forward(forward_epoch=1)
    runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=context,
        observation_p0=observation,
        actual_p0_full_row_matrix=matrix,
        plan_origin="provisional_current_plan",
    )
    try:
        runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
            layer_name="model.layers.0.mlp",
            phase_ctx=context,
            observation_p0=observation,
            actual_p0_full_row_matrix=matrix,
            plan_origin="provisional_current_plan",
        )
    except RuntimeError as exc:
        assert "duplicate current plan build" in str(exc)
    else:
        raise AssertionError("expected duplicate current plan build to fail")
    runtime.begin_forward(forward_epoch=2)
    runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=context,
        observation_p0=observation,
        actual_p0_full_row_matrix=matrix,
        plan_origin="provisional_current_plan",
    )


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


def test_multiphase_pending_window_adapter_compiles_current_phase_plan() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    prepared = _prepared_plan(forecast_digest="forecast-online", created="0")
    adapter = MultiphasePendingWindowAdapter(
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
        fast_path_enabled=False,
    )
    plan = adapter.build_plan(local_context=contexts[0], global_contexts=contexts)
    assert plan.metrics["compiled_from_pending_window"] is True
    assert plan.metrics["pending_window_logical_policy_name"].startswith("routersense_multiphase_lookahead:")
    assert plan.metrics["pending_window_information_mode"] in {"p0_p1", "p0_p1_p2"}


def test_multiphase_pending_window_adapter_uses_prepared_p1_and_p2_matrices() -> None:
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
    adapter = MultiphasePendingWindowAdapter(
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
        fast_path_enabled=False,
    )
    plan = adapter.build_plan(local_context=contexts[0], global_contexts=contexts)
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
    plan = MultiphasePendingWindowAdapter(
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
        fast_path_enabled=False,
    ).build_plan(local_context=contexts[0], global_contexts=contexts)
    runtime.config = replace(runtime.config, execution_mode="multiphase_pending_window")
    runtime._record_pending_window_driver(layer_name="model.layers.1.mlp", phase="P0", plan=plan)
    rows = runtime.export_pending_window_driver_records()
    assert rows
    assert rows[-1]["compiled_from_pending_window"] is True
    assert rows[-1]["pending_window_logical_policy_name"].startswith("routersense_multiphase_lookahead:")
    assert rows[-1]["wave_count"] == len(plan.waves)


def test_multiphase_pending_window_fast_path_skips_logical_build(monkeypatch) -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    prepared = _prepared_plan(forecast_digest="forecast-fast", created="0")

    def _boom(*args, **kwargs):
        raise AssertionError("heavy logical build should not run")

    monkeypatch.setattr(
        "rs.scheduling.multiphase.routersense_lookahead.RouterSenseMultiphaseLookaheadPolicy.build_logical_plan",
        _boom,
    )
    adapter = MultiphasePendingWindowAdapter(
        shared_state={
            "prepared_plan": prepared,
            "plan_created_at_us": 1,
            "plan_source_layer": "model.layers.0.mlp",
            "p2_matrix_source": "replicated_local_row",
            "p2_matrix_is_replicated_local_row": True,
        },
        phase_policy_name="routersense_p0p1p2_hint",
        bucket_rows=0,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
        fast_path_enabled=True,
    )
    plan = adapter.build_plan(local_context=contexts[0], global_contexts=contexts)
    validation = validate_phase_execution_plan(plan)
    assert not validation["errors"]
    assert plan.metrics["routersense_fast_path_enabled"] is True
    assert plan.metrics["routersense_heavy_path_used"] is False
    assert plan.metrics["pending_window_logical_build_time_us"] == 0.0
    assert plan.metrics["fast_path_wave_plan_valid"] is True
    assert len(plan.waves) >= 1


def test_multiphase_pending_window_fast_path_falls_back_without_prepared_plan() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 2), (3, 0)),
        p2_hint_mode="none",
    )
    adapter = MultiphasePendingWindowAdapter(
        shared_state={},
        phase_policy_name="routersense_p0p1p2_hint",
        bucket_rows=0,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
        fast_path_enabled=True,
    )
    plan = adapter.build_plan(local_context=contexts[0], global_contexts=contexts)
    assert plan.metrics["routersense_fast_path_enabled"] is True
    assert plan.metrics["routersense_heavy_path_used"] is False
    assert plan.metrics["fast_path_fallback_reason"] == "no_prepared_plan"
    assert len(plan.waves) >= 1
