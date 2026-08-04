"""Exact staging-capacity sensitivity helpers."""

from __future__ import annotations

from rs_sim.backend.core.errors import BackendContractError
from rs_sim.backend.core.util import require_nonnegative_int, require_positive_int

_RATIO_BY_NAME: dict[str, tuple[int, int] | None] = {
    "UNBOUNDED": None,
    "1.0X": (1, 1),
    "0.5X": (1, 2),
    "0.25X": (1, 4),
}


def align_up(value: int, alignment_bytes: int) -> int:
    value = require_nonnegative_int(value, field="value")
    alignment_bytes = require_positive_int(
        alignment_bytes, field="alignment_bytes"
    )
    return ((value + alignment_bytes - 1) // alignment_bytes) * alignment_bytes


def compute_staging_capacity_bytes(
    *,
    receiver_buffer_reference_bytes: int,
    sensitivity: str,
    alignment_bytes: int,
    max_canonical_task_payload_bytes: int,
) -> int | None:
    """Compute the frozen per-destination staging pool.

    ``None`` means the UNBOUNDED overlap upper bound.  Ratios are exact
    rationals, so no floating point enters authoritative simulation state.
    """

    reference = require_nonnegative_int(
        receiver_buffer_reference_bytes,
        field="receiver_buffer_reference_bytes",
    )
    max_task = require_nonnegative_int(
        max_canonical_task_payload_bytes,
        field="max_canonical_task_payload_bytes",
    )
    alignment = require_positive_int(alignment_bytes, field="alignment_bytes")
    key = str(sensitivity).upper()
    if key not in _RATIO_BY_NAME:
        raise BackendContractError(
            "sensitivity must be UNBOUNDED, 1.0x, 0.5x, or 0.25x"
        )
    ratio = _RATIO_BY_NAME[key]
    if ratio is None:
        return None
    numerator, denominator = ratio
    scaled = (reference * numerator + denominator - 1) // denominator
    return max(align_up(scaled, alignment), max_task)


def compute_fixture_staging_capacity_bytes_by_rank(
    *,
    fixture_input: object,
    sensitivity: str,
    alignment_bytes: int,
    max_canonical_task_payload_bytes: int,
) -> dict[int, int | None]:
    """Derive one fixed pool per destination from Trace payload matrices.

    The reference for each rank is the largest expected inbound payload across
    every Dispatch and Combine phase in the fixture.  This preserves one pool
    across active phases and keeps sensitivity independent of scheduling order.
    """

    windows = tuple(getattr(fixture_input, "windows"))
    if not windows:
        raise BackendContractError("fixture_input must contain Trace windows")
    world_size = int(getattr(fixture_input, "world_size"))
    reference = {rank: 0 for rank in range(world_size)}
    largest_remote_edge = {rank: 0 for rank in range(world_size)}
    configured_max_task = require_nonnegative_int(
        max_canonical_task_payload_bytes,
        field="max_canonical_task_payload_bytes",
    )
    for window in windows:
        for phase_kind in ("DISPATCH", "COMBINE"):
            matrix = tuple(tuple(int(v) for v in row) for row in window.payload_matrix(phase_kind))
            if len(matrix) != world_size or any(len(row) != world_size for row in matrix):
                raise BackendContractError("Trace payload matrix does not match world_size")
            for dst_rank in range(world_size):
                # Diagonal traffic uses local assembly and never consumes the
                # receiver staging pool.  Capacity sensitivity must therefore
                # be derived only from remote inbound payload.
                remote_values = tuple(
                    matrix[src_rank][dst_rank]
                    for src_rank in range(world_size)
                    if src_rank != dst_rank
                )
                inbound = sum(remote_values)
                reference[dst_rank] = max(reference[dst_rank], inbound)
                largest_remote_edge[dst_rank] = max(
                    largest_remote_edge[dst_rank], max(remote_values, default=0)
                )
    return {
        rank: compute_staging_capacity_bytes(
            receiver_buffer_reference_bytes=reference[rank],
            sensitivity=sensitivity,
            alignment_bytes=alignment_bytes,
            # The floor is the largest canonical task that this fixture can
            # actually generate, not the configured upper bound.
            max_canonical_task_payload_bytes=min(
                configured_max_task, largest_remote_edge[rank]
            ),
        )
        for rank in range(world_size)
    }
