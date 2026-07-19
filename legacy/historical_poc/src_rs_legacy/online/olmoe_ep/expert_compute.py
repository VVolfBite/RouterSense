from __future__ import annotations

"""Thin wrapper for the current WS=2 owner-rank expert compute path.

The real implementation currently lives in :mod:`ws2_native_ep`.
This module exists so callers do not conclude expert compute is absent.
"""

from .ws2_native_ep import _execute_online_expert_rows as ws2_execute_owner_rank_expert_rows

__all__ = ["ws2_execute_owner_rank_expert_rows"]
