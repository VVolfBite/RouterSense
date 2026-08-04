"""Public explicit instrumentation API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .session import CaptureSession

_SESSION: CaptureSession | None = None


def current_capture_session(*, required: bool = False) -> CaptureSession | None:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    config_path = os.environ.get("RS_SIM_CAPTURE_CONFIG")
    if not config_path:
        if required:
            raise RuntimeError("RS_SIM_CAPTURE_CONFIG is not set")
        return None
    _SESSION = CaptureSession(load_pipeline_config(Path(config_path)))
    return _SESSION


def set_capture_context(*, request_id: str | None = None, decode_step: int | None = None) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    session.set_context(request_id=request_id, decode_step=decode_step)



def set_capture_enabled(enabled: bool) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    session.set_enabled(bool(enabled))


def set_capture_performance_qualification(*, eligible: bool, evidence: dict[str, Any] | None = None) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    session.set_performance_qualification(eligible=bool(eligible), evidence=evidence)


def finish_capture_sample(*, decode_step: int) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    session.finish_sample(decode_step=int(decode_step))


def capture_routing(
    *,
    layer_id: int,
    routing_map: Any,
    probs: Any | None = None,
    raw_routing_map: Any | None = None,
    padding_rows: Any | None = None,
    local_expert_indices: Any | None = None,
    drop_and_pad: bool = False,
    metadata: dict[str, Any] | None = None,
    decode_step: int | None = None,
) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    session.record_routing(
        layer_id=layer_id,
        routing_map=routing_map,
        probs=probs,
        raw_routing_map=raw_routing_map,
        explicit_padding_rows=padding_rows,
        local_expert_indices=local_expert_indices,
        drop_and_pad=drop_and_pad,
        metadata=metadata,
        decode_step=decode_step,
    )


def flush_capture() -> None:
    session = current_capture_session(required=False)
    if session is not None:
        session.flush()
