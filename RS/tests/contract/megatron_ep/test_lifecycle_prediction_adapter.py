from __future__ import annotations

from types import SimpleNamespace
import time

import torch

import rs.runtime.online.megatron_ep.lifecycle as lifecycle_mod
from rs.runtime.online.megatron_ep.public_types import PublicationPollResult, PublicationPollStatus
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def test_lifecycle_prediction_adapter_records_worker_prediction_without_attribute_error() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            scheduler_mode="disabled",
            online_p2_predictor="copy_current_dispatch",
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="req",
        hostname="host",
    )
    runtime._policy_supports_target_layer_preplanning = lambda: True  # type: ignore[method-assign]
    runtime._layer_id_selected = lambda _layer_id: True  # type: ignore[method-assign]
    runtime._increment_state_counter_map = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._record_planning_timing = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._record_observer = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._runtime_state.write("predicted_dispatch_by_layer", {})
    runtime._runtime_state.write("actual_dispatch_by_layer", {})

    runtime._record_prediction_for_dispatch(  # noqa: SLF001
            layer_name="layers.1",
        phase_ctx=SimpleNamespace(),
        observation=SimpleNamespace(local_p0_row=(2, 3)),
        actual_p0_full_row_matrix=((0, 3), (2, 0)),
        device=torch.device("cpu"),
    )

    deadline = time.time() + 5.0
    while time.time() < deadline:
        runtime._pump_target_planner_publications()  # noqa: SLF001
        active_prediction = runtime._runtime_state.read("active_next_dispatch_prediction")  # noqa: SLF001
        if active_prediction:
            break
        time.sleep(0.01)

    active_prediction = runtime._runtime_state.read("active_next_dispatch_prediction")  # noqa: SLF001
    predicted_dispatch_by_layer = runtime._runtime_state.read("predicted_dispatch_by_layer")  # noqa: SLF001
    latest_digest = runtime._runtime_state.read("latest_prediction_digest")  # noqa: SLF001
    target_layer_id = str(active_prediction["target_layer_id"])
    assert active_prediction["predictor_name"] == "copy_current"
    assert active_prediction["forecast_matrix"] == [[0, 3], [2, 0]]
    assert predicted_dispatch_by_layer[target_layer_id]["predictor_name"] == "copy_current"
    assert latest_digest == active_prediction["matrix_digest"]
    runtime._cleanup_target_plan_runtime()  # noqa: SLF001


def test_publication_slot_retries_after_not_ready() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(scheduler_mode="disabled", online_p2_predictor="copy_current_dispatch"),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="req",
        hostname="host",
    )
    calls = {"count": 0}
    slot = SimpleNamespace(
        semantic_digest=lambda: "slot-digest",
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="1",
        target_layer_id="2",
    )
    runtime._forward_epoch = 1  # noqa: SLF001
    runtime._expected_publication_slots[("run", 1, "mb", "2")] = slot  # noqa: SLF001

    class _Lane:
        def poll(self, _slot, _candidate):
            calls["count"] += 1
            status = PublicationPollStatus.NOT_READY if calls["count"] == 1 else PublicationPollStatus.READY
            return PublicationPollResult(slot=_slot, status=status, root_rank=0, canonical_payload={}, details={})

    runtime.control_communication_lane = _Lane()  # noqa: SLF001
    runtime.target_plan_store = SimpleNamespace(close_key_if_unclaimed=lambda *a, **k: None)  # noqa: SLF001
    runtime._poll_target_plan_slot(target_layer_id="2", safe_point="source_combine_complete")  # noqa: SLF001
    runtime._poll_target_plan_slot(target_layer_id="2", safe_point="target_dispatch_ready")  # noqa: SLF001
    assert calls["count"] == 2


def test_terminal_publication_cleans_store_without_local_ready_candidate() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(scheduler_mode="disabled", online_p2_predictor="copy_current_dispatch"),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="req",
        hostname="host",
    )
    slot = SimpleNamespace(
        semantic_digest=lambda: "slot-digest",
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="1",
        target_layer_id="2",
    )
    runtime._forward_epoch = 1  # noqa: SLF001
    runtime._expected_publication_slots[("run", 1, "mb", "2")] = slot  # noqa: SLF001
    cleared: list[tuple[str, str]] = []

    class _Lane:
        def poll(self, _slot, _candidate):
            return PublicationPollResult(slot=_slot, status=PublicationPollStatus.FAILED, root_rank=0, canonical_payload={}, details={})

        def cancel_before_generation(self, **_kwargs):
            return None

    runtime.control_communication_lane = _Lane()  # noqa: SLF001
    runtime.target_planner_service = SimpleNamespace(  # noqa: SLF001
        cancel_slot=lambda _slot, final_status: cleared.append(("service", final_status)),
        publication_state_for_slot=lambda _slot: None,
    )
    runtime.target_plan_store = SimpleNamespace(  # noqa: SLF001
        clear_expected_publication=lambda key: cleared.append(("clear", key.target_layer_id)),
        close_key_if_unclaimed=lambda key, **_kwargs: cleared.append(("close", key.target_layer_id)),
    )
    runtime._poll_target_plan_slot(target_layer_id="2", safe_point="source_combine_complete")  # noqa: SLF001
    assert ("service", "FAILED") in cleared
    assert ("clear", "2") in cleared
    assert ("close", "2") in cleared


def test_prepared_target_execution_short_circuits_after_cancellation_without_materialization() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(scheduler_mode="disabled", online_p2_predictor="copy_current_dispatch"),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="req",
        hostname="host",
    )
    runtime._policy_supports_target_layer_preplanning = lambda: True  # type: ignore[method-assign]
    runtime._forward_epoch = 1  # noqa: SLF001
    phase_ctx = make_contexts_from_matrix(phase="P0", matrix=((0, 1), (1, 0)), p2_hint_mode="deterministic_stub")[0]

    class _CancelledStore:
        def peek(self, key):
            return None

    prepare_calls = {"count": 0}
    runtime.target_plan_store = _CancelledStore()  # type: ignore[assignment]
    runtime.execution_pipeline = SimpleNamespace(prepare=lambda *_args, **_kwargs: prepare_calls.__setitem__("count", prepare_calls["count"] + 1))

    result = runtime._try_prepared_target_plan_for_p0(  # noqa: SLF001
        layer_name="model.layers.1.mlp",
        phase_ctx=phase_ctx,
        actual_p0_full_row_matrix=((0, 1), (1, 0)),
    )

    assert result is None
    assert prepare_calls["count"] == 0
    assert runtime._runtime_state.read("prepared_plan_found") is False  # noqa: SLF001


def test_materialization_invalid_marks_store_failed_on_prepared_target_path() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(scheduler_mode="disabled", online_p2_predictor="copy_current_dispatch"),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="req",
        hostname="host",
    )
    runtime._policy_supports_target_layer_preplanning = lambda: True  # type: ignore[method-assign]
    runtime._forward_epoch = 1  # noqa: SLF001
    phase_ctx = make_contexts_from_matrix(phase="P0", matrix=((0, 1), (1, 0)), p2_hint_mode="deterministic_stub")[0]
    failed: list[tuple[object, str]] = []
    bound: list[object] = []
    started: list[tuple[object, str, str]] = []
    prepared_plan = SimpleNamespace(
        h1_prediction_digest="h1",
        source_layer_id="0",
        target_layer_id="1",
        h1_rows=((0, 1), (1, 0)),
        selected_variant="raw_u",
        safe_projection_mode="disabled",
    )
    outcome = SimpleNamespace(
        status="exact",
        logical_plan=SimpleNamespace(to_dict=lambda: {"plan": "ok"}),
        logical_plan_digest="logical-digest",
    )

    class _PreparedStore:
        def __init__(self) -> None:
            self.plan = prepared_plan

        def _key(self, key):
            return (str(key.run_id), int(key.forward_epoch), str(key.microbatch_id), str(key.target_layer_id))

        def peek(self, key):
            return self.plan

        def claim_for_reconciliation(self, key):
            return self.plan

        def bind(self, key, *, bound_owner):
            bound.append((key, str(bound_owner)))

        def start_execution(self, key, *, execution_origin, claim_owner):
            started.append((key, str(execution_origin), str(claim_owner)))

        def fail(self, key, *, execution_origin):
            failed.append((key, str(execution_origin)))

    runtime.target_plan_store = _PreparedStore()  # type: ignore[assignment]
    runtime.execution_pipeline = SimpleNamespace(
        prepare=lambda *_args, **_kwargs: SimpleNamespace(validation=SimpleNamespace(valid=False))
    )
    runtime._compile_async_phase_from_logical_plan = lambda **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        metrics={},
        waves=(),
        plan_hash="compiled",
        execution_mode="joint_window_async_p2p",
        transport_mutation=True,
    )
    original_reconcile_once = lifecycle_mod.reconcile_once
    lifecycle_mod.reconcile_once = lambda **_kwargs: outcome
    try:
        key = runtime._target_plan_key(layer_name="model.layers.1.mlp")  # noqa: SLF001
        runtime._execution_plan_cache()[runtime.target_plan_store._key(key)] = object()  # noqa: SLF001
        result = runtime._try_prepared_target_plan_for_p0(  # noqa: SLF001
            layer_name="model.layers.1.mlp",
            phase_ctx=phase_ctx,
            actual_p0_full_row_matrix=((0, 1), (1, 0)),
        )
    finally:
        lifecycle_mod.reconcile_once = original_reconcile_once

    assert result is None
    assert bound
    assert started
    assert len(failed) == 1
    assert failed[0][1] == "materialization_invalid"
    assert runtime._runtime_state.read("execution_origin") == "materialization_invalid"  # noqa: SLF001
