from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rs_sim.contracts.schema import SubmitOutcome


class RejectionCode(str, Enum):
    INVALID_BATCH_TYPE = "INVALID_BATCH_TYPE"
    INVALID_COMMIT_TIME = "INVALID_COMMIT_TIME"
    BATCH_LIMIT_EXCEEDED = "BATCH_LIMIT_EXCEEDED"
    TOPOLOGY_CONTRACT_MISMATCH = "FATAL_TOPOLOGY_CONTRACT_MISMATCH"
    STALE_AUTHORITY = "STALE_AUTHORITY"
    TASK_LOOKUP_FAILED = "TASK_LOOKUP_FAILED"
    TASK_CONTRACT_MISMATCH = "TASK_CONTRACT_MISMATCH"
    PERMIT_MISSING = "PERMIT_MISSING"
    PERMIT_CONTRACT_MISMATCH = "PERMIT_CONTRACT_MISMATCH"
    LOCAL_DIAGONAL_TASK = "LOCAL_DIAGONAL_TASK"
    DUPLICATE_PHYSICAL_TASK = "DUPLICATE_PHYSICAL_TASK"
    FOOTPRINT_LOOKUP_FAILED = "FOOTPRINT_LOOKUP_FAILED"
    FOOTPRINT_CONTRACT_MISMATCH = "FOOTPRINT_CONTRACT_MISMATCH"
    MIXED_LINK_CLASS = "MIXED_LINK_CLASS"
    INTERNAL_ENDPOINT_CONFLICT = "INTERNAL_ENDPOINT_CONFLICT"
    INTERNAL_NIC_CONFLICT = "INTERNAL_NIC_CONFLICT"
    NO_LEGAL_LANE_ASSIGNMENT = "NO_LEGAL_LANE_ASSIGNMENT"
    RESOURCE_BUSY = "RESOURCE_BUSY"


@dataclass(frozen=True, slots=True)
class TransportRejection:
    outcome: SubmitOutcome
    code: RejectionCode
    detail: str
    batch_id: str = ""


class ReceiptStateError(RuntimeError):
    """Raised only for invalid/non-live receipt operations.

    The formal contract guarantees that confirm_commit is infallible for the
    exact, unconsumed, live receipt returned by prepare_commit.
    """


class UnsupportedFormalExecutionModeError(RuntimeError):
    """Raised when transport is asked to execute a non-live formal mode."""


def validate_formal_execution_mode(execution_mode: str) -> str:
    normalized = str(execution_mode).upper()
    if normalized == "FULL_JOINT":
        raise UnsupportedFormalExecutionModeError(
            "EXPERIMENTAL_BLOCKED_NOT_LIVE: formal transport does not consume "
            "authoritative FULL_JOINT physical sub-batches"
        )
    if normalized != "ORDER_ONLY":
        raise UnsupportedFormalExecutionModeError(
            f"unsupported formal transport execution mode: {normalized}"
        )
    return normalized


__all__ = [
    "ReceiptStateError",
    "RejectionCode",
    "TransportRejection",
    "UnsupportedFormalExecutionModeError",
    "validate_formal_execution_mode",
]
