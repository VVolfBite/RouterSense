from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.control.contracts import ControlCommand, PlanExpiry, PlanKey
from rs.runtime.online.megatron_ep.control.validation import validate_command_not_expired


def _plan_key() -> PlanKey:
    return PlanKey(
        run_id_digest="run",
        forward_epoch=3,
        step_id="step",
        microbatch_id="mb",
        layer_id="0",
        phase="P0",
        ep_group_hash="ep",
        ep_group_epoch=1,
        model_revision_hash="model",
        expert_placement_hash="placement",
        request_table_hash="request",
    )


def test_control_command_expiry() -> None:
    command = ControlCommand(
        command_id="cmd-1",
        plan_key=_plan_key(),
        plan_hash="plan",
        policy_name="p0_fifo",
        policy_version="v1",
        control_mode="default_continue",
        phase="P0",
        action="reorder_pending",
        scope="bucket",
        issued_epoch=3,
        expiry=PlanExpiry(expiry_epoch=4, expiry_boundary="phase_end"),
        transport_mutation=False,
        is_shadow_only=True,
    )
    validate_command_not_expired(command, current_epoch=4)
    with pytest.raises(ValueError):
        validate_command_not_expired(command, current_epoch=5)
