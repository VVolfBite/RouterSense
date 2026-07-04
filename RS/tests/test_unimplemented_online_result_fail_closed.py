from __future__ import annotations

from rs.online import build_online_unimplemented_result


def test_unimplemented_online_result_is_fail_closed() -> None:
    payload = build_online_unimplemented_result(
        run_id="run-0",
        world_size=2,
        transport_backend="online_native_a2a_ep",
    )
    assert payload["execution_mode"] == "unsupported"
    assert payload["trace_origin"] == "not_collected"
    assert payload["claim_scope"] == "unsupported"
    assert payload["correctness_status"] == "unsupported"
