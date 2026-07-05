"""Exact/small-instance reference exports.

Current implementation reuses the oracle module until a stricter exact solver is
separated in round 2.
"""

from __future__ import annotations

from rs.scheduler.oracle import _pairwise_oracle_scipy

__all__ = ["_pairwise_oracle_scipy"]
