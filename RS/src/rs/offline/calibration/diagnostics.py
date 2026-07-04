from __future__ import annotations

from typing import Any

from ...contracts import FutureInformationMode, TraceOrigin


def assert_online_native_ep_observation(metadata: dict[str, Any]) -> None:
    trace_origin = metadata.get("trace_origin")
    if trace_origin != TraceOrigin.OBSERVED_ONLINE_NATIVE_EP:
        raise RuntimeError(
            "calibrated offline analysis requires trace_origin=observed_online_native_ep; "
            f"received {trace_origin!r}"
        )
    future_mode = metadata.get("future_information_mode")
    if future_mode not in {FutureInformationMode.NONE, FutureInformationMode.PREDICTED, FutureInformationMode.ORACLE_FULL_TRACE}:
        raise RuntimeError(f"unsupported future_information_mode for calibrated analysis: {future_mode!r}")
