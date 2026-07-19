from __future__ import annotations

import torch.distributed as dist

from . import async_release_backend as _backend

AsyncP2PExecutionResult = _backend.AsyncP2PExecutionResult
AsyncPhasePreflightResult = _backend.AsyncPhasePreflightResult
ReleaseBatchFrontier = _backend.ReleaseBatchFrontier
_digest_sequence_items = _backend._digest_sequence_items
_pair_index = _backend._pair_index
_sequence_entry = _backend._sequence_entry


def validate_async_phase_preflight(*args, **kwargs):
    _backend.dist = dist
    _backend.ReleaseBatchFrontier = ReleaseBatchFrontier
    return _backend.validate_async_phase_preflight(*args, **kwargs)


def execute_async_phase_tensor(*args, **kwargs):
    _backend.dist = dist
    _backend.ReleaseBatchFrontier = ReleaseBatchFrontier
    return _backend.execute_async_phase_tensor(*args, **kwargs)

__all__ = [
    "AsyncP2PExecutionResult",
    "AsyncPhasePreflightResult",
    "ReleaseBatchFrontier",
    "_digest_sequence_items",
    "_pair_index",
    "_sequence_entry",
    "dist",
    "execute_async_phase_tensor",
    "validate_async_phase_preflight",
]
