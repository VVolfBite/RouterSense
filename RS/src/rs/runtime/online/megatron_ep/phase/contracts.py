"""Phase 执行合同的统一导出层。

这里复用 scheduling/phase_execution 里的共享合同，供在线 runtime 引用。
"""

from rs.scheduling.phase_execution import (
    AbstractPhaseExecutionPlan,
    AbstractPlanWave,
    AbstractTaskRef,
    BucketTask,
    FutureDemandHint,
    IncomingSlot,
    OutgoingSegment,
    PackedTensorDescriptor,
    PayloadSlice,
    PhaseExecutionPlan,
    PhaseHookResult,
    PhaseName,
    PhasePlanningSummary,
    PhaseReadyContext,
    PlanWave,
    TransferLayout,
    TransportBundle,
)

__all__ = [
    "BucketTask",
    "AbstractPhaseExecutionPlan",
    "AbstractPlanWave",
    "AbstractTaskRef",
    "FutureDemandHint",
    "IncomingSlot",
    "OutgoingSegment",
    "PackedTensorDescriptor",
    "PayloadSlice",
    "PhaseExecutionPlan",
    "PhaseHookResult",
    "PhaseName",
    "PhasePlanningSummary",
    "PhaseReadyContext",
    "PlanWave",
    "TransferLayout",
    "TransportBundle",
]
