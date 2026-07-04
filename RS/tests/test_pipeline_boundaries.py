from __future__ import annotations

import pytest

from rs.legacy.trace_replay import LEGACY_TRACE_REPLAY_MODE, build_legacy_trace_replay_result
from rs.offline.calibration import assert_online_native_ep_observation
from rs.offline.router_prediction import build_router_prediction_result
from rs.online.scheduler_bridge import normalize_online_future_hint_mode
from rs.online.transport import transport_backend_realizes_matching
from rs.online.transport.native_alltoall import ONLINE_NATIVE_A2A_EP


def test_legacy_trace_replay_is_marked_legacy() -> None:
    payload = build_legacy_trace_replay_result(
        run_id="legacy-run",
        transport_backend="scheduled_collective_partition_replay",
        correctness_status="not_checked",
    )
    assert payload["pipeline"] == "legacy"
    assert payload["execution_mode"] == LEGACY_TRACE_REPLAY_MODE
    assert payload["trace_origin"] == "legacy_trace_replay"
    assert payload["is_real_ep_runtime"] is False
    assert payload["performance_claim_eligible"] is False


def test_offline_proxy_trace_cannot_drive_calibrated_ep_analysis() -> None:
    payload = build_router_prediction_result(run_id="offline-run")
    with pytest.raises(RuntimeError, match="observed_online_native_ep"):
        assert_online_native_ep_observation(payload)


def test_online_scheduler_rejects_oracle_future_trace() -> None:
    with pytest.raises(RuntimeError, match="oracle_full_trace"):
        normalize_online_future_hint_mode("oracle_full_trace")


def test_all_to_all_backend_is_not_matching_realized() -> None:
    assert transport_backend_realizes_matching(ONLINE_NATIVE_A2A_EP) is False
    assert transport_backend_realizes_matching("scheduled_p2p") is True
