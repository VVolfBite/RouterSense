from __future__ import annotations

"""Trace Provider adapters for shared schema/backend integration.

This bridge converts trace-owned workload facts into public shared-contract
objects. It never guesses NICs, lanes, or a full NetworkTopology; the
Integration Owner must construct the unique topology from the immutable
``rank_to_node`` input exposed here.
"""

from dataclasses import dataclass
from typing import Any

from rs_sim import (
    PhaseKey,
    PhaseKind,
    WindowKey,
    make_exact_dispatch_row_truth,
    make_exact_row_descriptor,
)
from rs_sim.trace.schema.constants import PHASE_COMBINE, PHASE_DISPATCH
from rs_sim.trace.schema.model import DescriptorMetadataSpec, TraceValidationError


@dataclass(frozen=True, slots=True)
class TraceWindowKeys:
    window_key: WindowKey
    dispatch_phase_key: PhaseKey
    combine_phase_key: PhaseKey


@dataclass(frozen=True, slots=True)
class ContinuousWindowKeys:
    current_window_key: WindowKey
    p0_dispatch_phase_key: PhaseKey
    p1_combine_phase_key: PhaseKey
    next_window_key: WindowKey | None
    p2_next_dispatch_phase_key: PhaseKey | None
    p3_next_combine_phase_key: PhaseKey | None


def keys_for_trace_window(*, run_id: str, trace_window: Any) -> TraceWindowKeys:
    sample_id = f"{trace_window.request_id}:step{trace_window.decode_step}"
    return TraceWindowKeys(
        window_key=WindowKey(
            run_id=str(run_id),
            sample_id=sample_id,
            window_index=int(trace_window.layer_id),
        ),
        dispatch_phase_key=PhaseKey(
            run_id=str(run_id),
            sample_id=sample_id,
            layer_index=int(trace_window.layer_id),
            phase_kind=PhaseKind.DISPATCH,
        ),
        combine_phase_key=PhaseKey(
            run_id=str(run_id),
            sample_id=sample_id,
            layer_index=int(trace_window.layer_id),
            phase_kind=PhaseKind.COMBINE,
        ),
    )


def keys_for_continuous_window(*, run_id: str, fixture: Any, anchor_index: int) -> ContinuousWindowKeys:
    """Return P0/P1 and exact next-layer P2/P3 keys without creating a plan."""

    index = int(anchor_index)
    if index < 0 or index >= len(fixture.windows):
        raise IndexError("anchor_index outside fixture windows")
    current = fixture.windows[index]
    current_keys = keys_for_trace_window(run_id=run_id, trace_window=current)
    next_window = fixture.windows[index + 1] if index + 1 < len(fixture.windows) else None
    if next_window is not None:
        consecutive = (
            next_window.request_id == current.request_id
            and next_window.decode_step == current.decode_step
            and next_window.layer_id == current.layer_id + 1
        )
        if not consecutive:
            next_window = None
    if next_window is None:
        return ContinuousWindowKeys(
            current_window_key=current_keys.window_key,
            p0_dispatch_phase_key=current_keys.dispatch_phase_key,
            p1_combine_phase_key=current_keys.combine_phase_key,
            next_window_key=None,
            p2_next_dispatch_phase_key=None,
            p3_next_combine_phase_key=None,
        )
    next_keys = keys_for_trace_window(run_id=run_id, trace_window=next_window)
    return ContinuousWindowKeys(
        current_window_key=current_keys.window_key,
        p0_dispatch_phase_key=current_keys.dispatch_phase_key,
        p1_combine_phase_key=current_keys.combine_phase_key,
        next_window_key=next_keys.window_key,
        p2_next_dispatch_phase_key=next_keys.dispatch_phase_key,
        p3_next_combine_phase_key=next_keys.combine_phase_key,
    )


def continuous_window_input_for_fixture(*, run_id: str, fixture: Any, anchor_index: int) -> dict[str, Any]:
    """Expose continuous-window truth and provenance, never a plan/timeline."""

    keys = keys_for_continuous_window(run_id=run_id, fixture=fixture, anchor_index=anchor_index)
    current = fixture.windows[int(anchor_index)]
    next_window = (
        fixture.windows[int(anchor_index) + 1]
        if keys.next_window_key is not None
        else None
    )
    payload = {
        "current_window_key": keys.current_window_key,
        "p0_dispatch_phase_key": keys.p0_dispatch_phase_key,
        "p1_combine_phase_key": keys.p1_combine_phase_key,
        "p2_next_dispatch_phase_key": keys.p2_next_dispatch_phase_key,
        "p3_next_combine_phase_key": keys.p3_next_combine_phase_key,
        "current_dispatch_rows": current.dispatch_rows,
        "current_combine_rows": current.combine_rows,
        "current_dispatch_payload_spec": current.dispatch_payload_spec,
        "current_combine_payload_spec": current.combine_payload_spec,
        "current_local_compute": current.local_compute,
        "rank_to_node": rank_to_node_input_for_trace_window(current),
        "descriptor_metadata_spec": current.descriptor_metadata_spec,
        "current_complete_payload_accounting": current.complete_payload_accounting(),
        "current_workload_identity_digest": current.workload_identity_digest(),
        "fixture_provenance_digest": fixture.provenance.identity_digest(),
        "fixture_truth_digest": fixture.truth_digest(),
        "contains_plan": False,
        "contains_transport_timeline": False,
    }
    if next_window is not None:
        payload.update(
            {
                "next_dispatch_rows": next_window.dispatch_rows,
                "next_combine_rows": next_window.combine_rows,
                "next_dispatch_payload_spec": next_window.dispatch_payload_spec,
                "next_combine_payload_spec": next_window.combine_payload_spec,
                "next_workload_identity_digest": next_window.workload_identity_digest(),
            }
        )
    return payload


def payload_bytes_for_phase(trace_window: Any, phase_key: PhaseKey) -> tuple[tuple[int, ...], ...]:
    if phase_key.phase_kind is PhaseKind.DISPATCH:
        return trace_window.payload_matrix(PHASE_DISPATCH)
    if phase_key.phase_kind is PhaseKind.COMBINE:
        return trace_window.payload_matrix(PHASE_COMBINE)
    raise ValueError(f"unsupported phase kind {phase_key.phase_kind}")


def _resolve_descriptor_metadata_spec(
    *,
    trace_window: Any,
    descriptor_metadata_spec: DescriptorMetadataSpec | None,
    descriptor_entry_bytes: int | None,
    descriptor_fixed_header_bytes: int | None,
) -> DescriptorMetadataSpec:
    spec = descriptor_metadata_spec or trace_window.descriptor_metadata_spec
    if not isinstance(spec, DescriptorMetadataSpec):
        raise TypeError("descriptor_metadata_spec must be DescriptorMetadataSpec")
    legacy_values_supplied = descriptor_entry_bytes is not None or descriptor_fixed_header_bytes is not None
    if legacy_values_supplied:
        if descriptor_entry_bytes is None or descriptor_fixed_header_bytes is None:
            raise ValueError("legacy descriptor byte arguments must be supplied together")
        if int(descriptor_entry_bytes) != int(spec.per_destination_entry_bytes):
            raise ValueError("descriptor_entry_bytes conflicts with trace descriptor metadata truth")
        if int(descriptor_fixed_header_bytes) != int(spec.fixed_header_bytes):
            raise ValueError("descriptor_fixed_header_bytes conflicts with trace descriptor metadata truth")
    return spec


def dispatch_row_truth_for_trace_window(
    *,
    trace_window: Any,
    phase_key: PhaseKey,
    src_rank: int,
    descriptor_metadata_spec: DescriptorMetadataSpec | None = None,
    descriptor_entry_bytes: int | None = None,
    descriptor_fixed_header_bytes: int | None = None,
) -> dict[str, Any]:
    """Return formal Backend row truth with phase-specific payload bytes.

    Realized row counts are the common truth. Dispatch and Combine bytes are
    independently derived from their own PayloadSpec. Descriptor fixed-header
    and per-destination-entry bytes are ControlPlane-only and are not added to
    DataPlane payload bytes.
    """

    if phase_key.phase_kind is not PhaseKind.DISPATCH:
        raise ValueError("exact row descriptor truth is Dispatch-only")
    world_size = int(trace_window.mapping.world_size)
    if src_rank < 0 or src_rank >= world_size:
        raise ValueError("src_rank outside trace world_size")
    spec = _resolve_descriptor_metadata_spec(
        trace_window=trace_window,
        descriptor_metadata_spec=descriptor_metadata_spec,
        descriptor_entry_bytes=descriptor_entry_bytes,
        descriptor_fixed_header_bytes=descriptor_fixed_header_bytes,
    )
    dispatch_rows = trace_window.dispatch_rows
    dispatch_payload = trace_window.payload_matrix(PHASE_DISPATCH)
    combine_payload = trace_window.payload_matrix(PHASE_COMBINE)
    realized_row = tuple(int(value) for value in dispatch_rows[src_rank])
    dispatch_row_bytes = tuple(int(value) for value in dispatch_payload[src_rank])
    combine_return_bytes = tuple(
        int(combine_payload[expert_rank][src_rank]) for expert_rank in range(world_size)
    )
    descriptor_payload_bytes = spec.descriptor_payload_bytes(world_size)
    truth = make_exact_dispatch_row_truth(
        phase_key=phase_key,
        src_rank=int(src_rank),
        realized_rows_by_destination=realized_row,
        dispatch_payload_bytes_by_destination=dispatch_row_bytes,
        combine_return_payload_bytes_by_expert=combine_return_bytes,
        dispatch_payload_spec_digest=trace_window.dispatch_payload_spec.digest(),
        combine_payload_spec_digest=trace_window.combine_payload_spec.digest(),
        descriptor_payload_bytes=descriptor_payload_bytes,
    )
    return {
        "phase_key": truth.phase_key,
        "src_rank": truth.src_rank,
        "realized_rows_by_destination": truth.realized_rows_by_destination,
        "dispatch_payload_bytes_by_destination": truth.dispatch_payload_bytes_by_destination,
        "payload_bytes_by_destination": truth.dispatch_payload_bytes_by_destination,
        "combine_return_payload_bytes_by_expert": truth.combine_return_payload_bytes_by_expert,
        "dispatch_payload_spec_digest": truth.dispatch_payload_spec_digest,
        "combine_payload_spec_digest": truth.combine_payload_spec_digest,
        "descriptor_payload_bytes": truth.descriptor_payload_bytes,
        "truth_digest": truth.truth_digest,
    }


def descriptor_metadata_accounting_for_trace_window(trace_window: Any) -> dict[str, Any]:
    spec = trace_window.descriptor_metadata_spec
    return {
        "fixed_header_bytes": int(spec.fixed_header_bytes),
        "per_destination_entry_bytes": int(spec.per_destination_entry_bytes),
        "destination_count": int(trace_window.mapping.world_size),
        "descriptor_payload_bytes": int(spec.descriptor_payload_bytes(trace_window.mapping.world_size)),
        "accounting_plane": spec.accounting_plane,
        "included_in_data_plane_payload": spec.included_in_data_plane_payload,
        "descriptor_metadata_spec_digest": spec.digest(),
    }


def exact_row_descriptor_for_trace_window(
    *,
    trace_window: Any,
    phase_key: PhaseKey,
    src_rank: int,
    published_at_ns: int,
    descriptor_metadata_spec: DescriptorMetadataSpec | None = None,
):
    """Build the public descriptor while preserving content-only identity."""

    row_truth = dispatch_row_truth_for_trace_window(
        trace_window=trace_window,
        phase_key=phase_key,
        src_rank=src_rank,
        descriptor_metadata_spec=descriptor_metadata_spec,
    )
    return make_exact_row_descriptor(
        phase_key=phase_key,
        src_rank=int(src_rank),
        realized_rows_by_destination=row_truth["realized_rows_by_destination"],
        payload_bytes_by_destination=row_truth["dispatch_payload_bytes_by_destination"],
        payload_spec_digest=row_truth["dispatch_payload_spec_digest"],
        published_at_ns=int(published_at_ns),
        descriptor_payload_bytes=int(row_truth["descriptor_payload_bytes"]),
    )


def rank_to_node_input_for_trace_window(trace_window: Any) -> tuple[int, ...]:
    """Return immutable topology input; no NIC/lane topology is guessed here."""

    rank_to_node = tuple(int(value) for value in trace_window.mapping.rank_to_node)
    if len(rank_to_node) != int(trace_window.mapping.world_size):
        raise TraceValidationError("rank_to_node length does not match trace world_size")
    return rank_to_node


def complete_payload_accounting_for_trace_window(trace_window: Any) -> dict[str, Any]:
    return dict(trace_window.complete_payload_accounting())


def phase_payload_accounting(trace_window: Any, phase_key: PhaseKey) -> dict[str, int]:
    if phase_key.phase_kind is PhaseKind.DISPATCH:
        return dict(trace_window.payload_accounting(PHASE_DISPATCH))
    if phase_key.phase_kind is PhaseKind.COMBINE:
        return dict(trace_window.payload_accounting(PHASE_COMBINE))
    raise ValueError(f"unsupported phase kind {phase_key.phase_kind}")


__all__ = [
    "ContinuousWindowKeys",
    "TraceWindowKeys",
    "complete_payload_accounting_for_trace_window",
    "continuous_window_input_for_fixture",
    "descriptor_metadata_accounting_for_trace_window",
    "dispatch_row_truth_for_trace_window",
    "exact_row_descriptor_for_trace_window",
    "keys_for_continuous_window",
    "keys_for_trace_window",
    "payload_bytes_for_phase",
    "phase_payload_accounting",
    "rank_to_node_input_for_trace_window",
]
