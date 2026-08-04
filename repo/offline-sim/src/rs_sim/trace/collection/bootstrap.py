"""Import-time bootstrap used by the launcher's temporary sitecustomize."""

from __future__ import annotations

import os


def install_from_environment() -> None:
    if os.environ.get("RS_SIM_CAPTURE_DISABLE", "0") == "1":
        return
    if (
        os.environ.get("RS_SIM_CAPTURE_DEFER_TO_DISTRIBUTED_WORKERS", "0") == "1"
        and "RANK" not in os.environ
    ):
        # torchrun itself imports sitecustomize before worker rank metadata is
        # present.  Do not create a rank0/world1 session that can overwrite the
        # real rank-0 worker manifest at launcher exit.
        return
    backend = os.environ.get("RS_SIM_CAPTURE_BACKEND", "MEGATRON_CORE_AUTO").upper()
    if backend == "MEGATRON_CORE_AUTO":
        from .megatron import install_megatron_auto_capture
        install_megatron_auto_capture()
    elif backend in {"EXPLICIT_API", "REPLAY_ONLY"}:
        from .api import current_capture_session
        current_capture_session(required=True)
    else:
        raise RuntimeError(f"unsupported RS_SIM_CAPTURE_BACKEND={backend}")
