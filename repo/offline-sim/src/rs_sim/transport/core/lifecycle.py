from __future__ import annotations

import multiprocessing
import os
import threading
from typing import Any

from rs_sim.contracts.digest import stable_digest


def capture_process_resource_snapshot() -> dict[str, Any]:
    """Best-effort process-global observation for isolated-runner leak checks."""

    threads = tuple(threading.enumerate())
    children = tuple(multiprocessing.active_children())
    fd_count: int | None = None
    fd_root = "/proc/self/fd"
    if os.path.isdir(fd_root):
        try:
            fd_count = len(os.listdir(fd_root))
        except OSError:
            fd_count = None
    return {
        "process_id": int(os.getpid()),
        "thread_count": len(threads),
        "non_daemon_thread_count": sum(not item.daemon for item in threads),
        "child_process_count": len(children),
        "open_file_descriptor_count": fd_count,
    }


def make_process_lifecycle_diagnostics(
    *,
    component: str,
    closed: bool,
    disposed: bool,
    kernel_pending_event_count: int,
    live_receipt_count: int,
    live_transfer_or_request_count: int,
    all_resources_free: bool,
    final_evidence_digest: str | None,
    kernel_callback_registry_disposed: bool,
) -> dict[str, Any]:
    """Stable diagnostics for the official isolated runner.

    transport is deliberately event-driven and single-threaded. It creates no child
    process, thread, executor or file handle. The counters below are ownership
    assertions, not a snapshot of unrelated process-global resources.
    """

    payload: dict[str, Any] = {
        "schema_version": "TRANSPORT_PROCESS_LIFECYCLE_DIAGNOSTICS",
        "component": str(component),
        "closed": bool(closed),
        "disposed": bool(disposed),
        "kernel_pending_event_count": int(kernel_pending_event_count),
        "live_receipt_count": int(live_receipt_count),
        "live_transfer_or_request_count": int(live_transfer_or_request_count),
        "all_resources_free": bool(all_resources_free),
        "kernel_callback_registry_disposed": bool(
            kernel_callback_registry_disposed
        ),
        "transport_spawns_processes": False,
        "transport_uses_threads": False,
        "transport_uses_executors": False,
        "transport_opens_files": False,
        "owned_child_process_count": 0,
        "owned_thread_count": 0,
        "owned_executor_count": 0,
        "owned_file_handle_count": 0,
        "final_evidence_digest": final_evidence_digest,
    }
    payload["diagnostics_digest"] = stable_digest(
        payload, domain="TRANSPORT_PROCESS_LIFECYCLE_DIAGNOSTICS"
    )
    payload["process_resource_snapshot"] = capture_process_resource_snapshot()
    return payload


__all__ = [
    "capture_process_resource_snapshot",
    "make_process_lifecycle_diagnostics",
]
