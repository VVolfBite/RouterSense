from __future__ import annotations

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.runtime import RouterSenseInjectionRuntime


def test_identity_runtime_assertions_can_report_unchanged_buffers() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            scheduler_mode="native_passthrough_identity",
            control_mode="sync_before_phase",
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
    )
    runtime.assertion_state["native_buffers_unchanged"] = True
    assert runtime.export_assertions()["native_buffers_unchanged"] is True
