from __future__ import annotations

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.runtime import RouterSenseInjectionRuntime


def test_default_continue_early_shadow_replace_metadata_contract() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            scheduler_mode="native_passthrough_identity",
            control_mode="default_continue",
            shadow_command_arrival="before_commit",
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
    runtime.control_timeline.append({"event": "shadow_command_replaced_active"})
    runtime.control_commands.append({"status": "applied"})
    assert any(row["event"] == "shadow_command_replaced_active" for row in runtime.export_control_timeline())
    assert any(row["status"] == "applied" for row in runtime.export_control_commands())
