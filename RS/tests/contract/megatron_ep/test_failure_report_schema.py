from __future__ import annotations

from experiments.online.run_policy_correctness import _failure_report


def test_failure_report_schema() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        payload = _failure_report(
            stage="phase_executor_runtime",
            exc=exc,
            rank=0,
            local_rank=0,
            plan_hash="plan-1",
            layer_id="0",
            phase="P0",
            wave_id=1,
            bucket_id="P0:0:1:0",
            tensor_role="hidden_states",
            expected_shape=[4, 8],
            actual_shape=[3, 8],
            expected_dtype="torch.float16",
            actual_dtype="torch.float16",
            expected_splits=[1, 3],
            actual_splits=[1, 2],
        )

    assert payload["first_failure_stage"] == "phase_executor_runtime"
    assert payload["rank"] == 0
    assert payload["phase"] == "P0"
    assert payload["wave_id"] == 1
    assert payload["bucket_id"] == "P0:0:1:0"
    assert payload["tensor_role"] == "hidden_states"
    assert payload["expected_shape"] == [4, 8]
    assert payload["actual_splits"] == [1, 2]
    assert "RuntimeError: boom" in payload["exception_summary"]
    assert "traceback" in payload
