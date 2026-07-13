from __future__ import annotations

from types import SimpleNamespace

import torch

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime


def test_lifecycle_prediction_adapter_returns_typed_result_without_attribute_error() -> None:
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
    runtime._policy_supports_target_layer_preplanning = lambda: False  # type: ignore[method-assign]
    runtime._increment_state_counter_map = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._record_planning_timing = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._record_observer = lambda *args, **kwargs: None  # type: ignore[method-assign]
    runtime._runtime_state.write("predicted_dispatch_by_layer", {})
    runtime._runtime_state.write("actual_dispatch_by_layer", {})

    runtime._record_prediction_for_dispatch(  # noqa: SLF001
        layer_name="1",
        phase_ctx=SimpleNamespace(),
        observation=SimpleNamespace(local_p0_row=(2, 3)),
        actual_p0_full_row_matrix=((0, 3), (2, 0)),
        device=torch.device("cpu"),
    )

    active_prediction = runtime._runtime_state.read("active_next_dispatch_prediction")  # noqa: SLF001
    predicted_dispatch_by_layer = runtime._runtime_state.read("predicted_dispatch_by_layer")  # noqa: SLF001
    latest_digest = runtime._runtime_state.read("latest_prediction_digest")  # noqa: SLF001
    target_layer_id = str(runtime._next_layer_id("1"))  # noqa: SLF001
    assert active_prediction["predictor_name"] == "copy_current"
    assert active_prediction["forecast_matrix"] == [[0, 3], [2, 0]]
    assert predicted_dispatch_by_layer[target_layer_id]["predictor_name"] == "copy_current"
    assert latest_digest == active_prediction["matrix_digest"]
