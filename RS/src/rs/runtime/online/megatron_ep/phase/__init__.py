"""Megatron EP phase interfaces."""

from .context_builder import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)
from .contracts import (
    BucketTask,
    FutureDemandHint,
    IncomingSlot,
    OutgoingSegment,
    PackedTensorDescriptor,
    PayloadSlice,
    PhaseExecutionPlan,
    PhaseHookResult,
    PhaseName,
    PhaseReadyContext,
    PlanWave,
    TransferLayout,
    TransportBundle,
)
from .layout_join import join_transfer_layouts
from .validation import validate_layout_offsets_cover_exactly_once, validate_p0_atomic_bundle

__all__ = [
    "BucketTask",
    "DispatcherSnapshot",
    "FutureDemandHint",
    "IncomingSlot",
    "OutgoingSegment",
    "PackedTensorDescriptor",
    "PayloadSlice",
    "PhaseExecutionPlan",
    "PhaseContextBuildRequest",
    "PhaseHookResult",
    "PhaseName",
    "PhasePayloadContract",
    "PhaseReadyContext",
    "RuntimeIdentity",
    "PlanWave",
    "TransferLayout",
    "TransportBundle",
    "build_phase_ready_context",
    "join_transfer_layouts",
    "validate_layout_offsets_cover_exactly_once",
    "validate_p0_atomic_bundle",
]
