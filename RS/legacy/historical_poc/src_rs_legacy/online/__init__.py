from __future__ import annotations

"""Online EP runtime namespace.

Phase 1 only establishes the boundary and explicit failure semantics. A real
online EP runtime is not implemented yet.
"""

from .runtime_metadata import (
    build_online_result_envelope,
    build_online_unimplemented_result,
    online_claim_scope_for_world_size,
)

__all__ = [
    "build_online_result_envelope",
    "build_online_unimplemented_result",
    "online_claim_scope_for_world_size",
]
