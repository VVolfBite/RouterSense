from __future__ import annotations

from rs.runtime.online.megatron_ep.control.contracts import PlanKey


def test_plan_key_epoch_uniqueness() -> None:
    base = dict(
        run_id_digest="run",
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
    key1 = PlanKey(forward_epoch=1, **base)
    key2 = PlanKey(forward_epoch=2, **base)
    assert key1 != key2
    assert key1.to_dict()["forward_epoch"] == 1
    assert key2.to_dict()["forward_epoch"] == 2
