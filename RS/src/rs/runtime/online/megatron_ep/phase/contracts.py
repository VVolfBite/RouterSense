"""Compatibility re-export of shared phase execution contracts."""

from rs.scheduling.phase_execution import (
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

__all__ = [
    "BucketTask",
    "FutureDemandHint",
    "IncomingSlot",
    "OutgoingSegment",
    "PackedTensorDescriptor",
    "PayloadSlice",
    "PhaseExecutionPlan",
    "PhaseHookResult",
    "PhaseName",
    "PhaseReadyContext",
    "PlanWave",
    "TransferLayout",
    "TransportBundle",
]
