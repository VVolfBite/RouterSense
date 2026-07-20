from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rs.runtime.online.megatron_ep.control.plan_agreement import (
    _decode_abstract_plan_values,
    _encode_abstract_plan_tensor,
)
from rs.scheduling.phase_execution import AbstractPhaseExecutionPlan


def _local_context():
    return SimpleNamespace(
        plan_key={"layer_id": "0", "phase": "P0"},
        phase="P0",
        control_mode="sync_before_phase",
        p2_hint=SimpleNamespace(hint_mode="calibrated_artifact"),
    )


def test_prepared_priority_policy_code_roundtrip() -> None:
    plan = AbstractPhaseExecutionPlan.from_dict(
        {
            "plan_key": {"layer_id": "0", "phase": "P0"},
            "phase": "P0",
            "policy_name": "prepared_priority",
            "policy_version": "v1",
            "control_mode": "sync_before_phase",
            "execution_mode": "phase_sync_wave",
            "transport_mutation": False,
            "is_shadow_only": False,
            "future_hint_mode": "calibrated_artifact",
            "root_rank": 0,
            "observation_digest": "0" * 64,
            "plan_hash": "1" * 64,
            "waves": [{"wave_id": 0, "phase": "P0", "task_refs": []}],
            "metrics": {"bucket_rows": 0, "wave_count": 1},
        }
    )
    tensor = _encode_abstract_plan_tensor(plan, device=torch.device("cpu"))
    decoded = _decode_abstract_plan_values([int(value) for value in tensor.tolist()], local_context=_local_context())
    assert decoded.policy_name == "prepared_priority"


def test_unknown_policy_fails_closed() -> None:
    bad = AbstractPhaseExecutionPlan.from_dict(
        {
            "plan_key": {"layer_id": "0", "phase": "P0"},
            "phase": "P0",
            "policy_name": "unknown_policy",
            "policy_version": "v1",
            "control_mode": "sync_before_phase",
            "execution_mode": "phase_sync_wave",
            "transport_mutation": False,
            "is_shadow_only": False,
            "future_hint_mode": "none",
            "root_rank": 0,
            "observation_digest": "0" * 64,
            "plan_hash": "1" * 64,
            "waves": [],
            "metrics": {"bucket_rows": 0, "wave_count": 0},
        }
    )
    with pytest.raises(RuntimeError, match="unknown abstract-plan policy"):
        _encode_abstract_plan_tensor(bad, device=torch.device("cpu"))
    values = [
        1,  # wire
        0,  # phase P0
        999,  # unknown policy code
        0,
        0,
        0,
        0,
        0,
        *([0] * 8),
    ]
    with pytest.raises(RuntimeError, match="unknown abstract-plan policy code"):
        _decode_abstract_plan_values(values, local_context=_local_context())
