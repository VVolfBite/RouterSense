from __future__ import annotations

from rs.runtime.online.megatron_ep.contracts import NativeEPSummary


def test_native_ep_summary_contract_fields() -> None:
    payload = NativeEPSummary(ep_size=2, dispatcher="alltoall", backend="nccl").to_dict()
    assert payload["pipeline"] == "host_runtime_native_ep"
    assert payload["host_runtime"] == "megatron_core"
    assert payload["ep_size"] == 2
    assert payload["dispatcher"] == "alltoall"
    assert payload["backend"] == "nccl"
    assert "forward_completed" in payload
    assert "remote_dispatch_exercised" in payload
    assert "remote_combine_exercised" in payload
