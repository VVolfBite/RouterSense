from __future__ import annotations

"""Thin WS=2 dispatch wrapper.

The real current implementation lives in `ws2_native_ep.py`. This module exists
only to point callers at the live dispatch path instead of the old
NotImplementedError stub.
"""

from .ws2_native_ep import _run_a2a_with_metadata as ws2_dispatch_alltoall_with_metadata

__all__ = ["ws2_dispatch_alltoall_with_metadata"]
