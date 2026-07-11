"""Phase package exports with lazy import."""

from __future__ import annotations

__all__ = [
    "BucketTask",
    "AbstractPhaseExecutionPlan",
    "AbstractPlanWave",
    "AbstractTaskRef",
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
    "PhasePlanningSummary",
    "PhasePayloadContract",
    "PhaseReadyContext",
    "RuntimeIdentity",
    "PlanWave",
    "TransferLayout",
    "TransportBundle",
    "build_phase_ready_context",
    "reconstruct_global_phase_contexts_from_byte_matrix",
    "join_transfer_layouts",
    "validate_layout_offsets_cover_exactly_once",
    "validate_p0_atomic_bundle",
]


def __getattr__(name: str):
    if name in {
        "DispatcherSnapshot",
        "PhaseContextBuildRequest",
        "PhasePayloadContract",
        "RuntimeIdentity",
        "build_phase_ready_context",
        "reconstruct_global_phase_contexts_from_byte_matrix",
    }:
        from .context_builder import (
            DispatcherSnapshot,
            PhaseContextBuildRequest,
            PhasePayloadContract,
            RuntimeIdentity,
            build_phase_ready_context,
            reconstruct_global_phase_contexts_from_byte_matrix,
        )

        return {
            "DispatcherSnapshot": DispatcherSnapshot,
            "PhaseContextBuildRequest": PhaseContextBuildRequest,
            "PhasePayloadContract": PhasePayloadContract,
            "RuntimeIdentity": RuntimeIdentity,
            "build_phase_ready_context": build_phase_ready_context,
            "reconstruct_global_phase_contexts_from_byte_matrix": reconstruct_global_phase_contexts_from_byte_matrix,
        }[name]
    if name in {
        "AbstractPhaseExecutionPlan",
        "AbstractPlanWave",
        "AbstractTaskRef",
        "BucketTask",
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
    }:
        from .contracts import (
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

        return {
            "AbstractPhaseExecutionPlan": AbstractPhaseExecutionPlan,
            "AbstractPlanWave": AbstractPlanWave,
            "AbstractTaskRef": AbstractTaskRef,
            "BucketTask": BucketTask,
            "FutureDemandHint": FutureDemandHint,
            "IncomingSlot": IncomingSlot,
            "OutgoingSegment": OutgoingSegment,
            "PackedTensorDescriptor": PackedTensorDescriptor,
            "PayloadSlice": PayloadSlice,
            "PhaseExecutionPlan": PhaseExecutionPlan,
            "PhaseHookResult": PhaseHookResult,
            "PhaseName": PhaseName,
            "PhasePlanningSummary": PhasePlanningSummary,
            "PhaseReadyContext": PhaseReadyContext,
            "PlanWave": PlanWave,
            "TransferLayout": TransferLayout,
            "TransportBundle": TransportBundle,
        }[name]
    if name == "join_transfer_layouts":
        from .layout_join import join_transfer_layouts

        return join_transfer_layouts
    if name in {"validate_layout_offsets_cover_exactly_once", "validate_p0_atomic_bundle"}:
        from .validation import validate_layout_offsets_cover_exactly_once, validate_p0_atomic_bundle

        return {
            "validate_layout_offsets_cover_exactly_once": validate_layout_offsets_cover_exactly_once,
            "validate_p0_atomic_bundle": validate_p0_atomic_bundle,
        }[name]
    raise AttributeError(name)
