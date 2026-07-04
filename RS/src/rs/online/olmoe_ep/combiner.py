from __future__ import annotations

"""Thin wrapper for the current WS=2 inverse-combine path.

The real implementation currently lives in :mod:`ws2_native_ep`.
This module exists so callers do not conclude combine is absent.
"""

from .ws2_native_ep import _scatter_local_output as ws2_scatter_inverse_combine_output

__all__ = ["ws2_scatter_inverse_combine_output"]
