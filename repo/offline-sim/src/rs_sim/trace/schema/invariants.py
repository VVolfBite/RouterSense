"""Expected invariant generation and validation.

These invariants describe input truth only.  They intentionally contain no
canonical task IDs, plans, transport start/finish times, event IDs, or global
expected timeline digest.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .canonical import stable_digest
from .constants import PHASE_COMBINE, PHASE_DISPATCH
from .model import FixtureInput, TraceValidationError, TraceWindow


def matrix_sum(matrix: tuple[tuple[int, ...], ...]) -> int:
    return sum(sum(int(value) for value in row) for row in matrix)


def matrix_transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if not matrix:
        return ()
    return tuple(tuple(int(matrix[row][col]) for row in range(len(matrix))) for col in range(len(matrix[0])))


def nonzero_edge_count(matrix: tuple[tuple[int, ...], ...]) -> int:
    return sum(1 for row in matrix for value in row if int(value) > 0)


def window_invariants(window: TraceWindow) -> dict[str, Any]:
    totals = window.routing.totals()
    dispatch = window.dispatch_rows
    combine = window.combine_rows
    dispatch_payload = window.payload_matrix(PHASE_DISPATCH)
    combine_payload = window.payload_matrix(PHASE_COMBINE)
    dispatch_accounting = window.payload_accounting(PHASE_DISPATCH)
    combine_accounting = window.payload_accounting(PHASE_COMBINE)
    return {
        "window_id": window.window_id,
        "layer_id": window.layer_id,
        "request_id": window.request_id,
        "decode_step": window.decode_step,
        "is_bootstrap_p0": window.is_bootstrap_p0,
        "raw_equals_kept_plus_dropped": totals["raw_selected_rows"] == totals["kept_rows"] + totals["dropped_rows"],
        "realized_equals_kept_plus_padding": totals["realized_transfer_rows"] == totals["kept_rows"] + totals["padding_rows"],
        "totals": totals,
        "dispatch_rows_by_source": [sum(row) for row in dispatch],
        "combine_rows_by_source": [sum(row) for row in combine],
        "dispatch_total_rows": matrix_sum(dispatch),
        "combine_total_rows": matrix_sum(combine),
        "combine_is_dispatch_transpose": combine == matrix_transpose(dispatch),
        "dispatch_nonzero_edge_count": nonzero_edge_count(dispatch),
        "combine_nonzero_edge_count": nonzero_edge_count(combine),
        "dispatch_payload_total_bytes": matrix_sum(dispatch_payload),
        "combine_payload_total_bytes": matrix_sum(combine_payload),
        "dispatch_payload_accounting": dispatch_accounting,
        "combine_payload_accounting": combine_accounting,
        "dispatch_accounting_closes": dispatch_accounting["local_payload_bytes"] + dispatch_accounting["remote_network_payload_bytes"] == dispatch_accounting["total_payload_bytes"],
        "combine_accounting_closes": combine_accounting["local_payload_bytes"] + combine_accounting["remote_network_payload_bytes"] == combine_accounting["total_payload_bytes"],
        "dispatch_payload_digest": stable_digest(dispatch_payload),
        "combine_payload_digest": stable_digest(combine_payload),
        "routing_digest": window.routing.digest(),
        "mapping_digest": window.mapping.digest(),
        "dispatch_payload_spec_digest": window.dispatch_payload_spec.digest(),
        "combine_payload_spec_digest": window.combine_payload_spec.digest(),
        "descriptor_metadata_spec_digest": window.descriptor_metadata_spec.digest(),
        "descriptor_payload_bytes": window.descriptor_payload_bytes(),
        "descriptor_payload_bytes_all_sources": window.descriptor_payload_bytes_all_sources(),
        "descriptor_accounted_only_on_control_plane": not window.descriptor_metadata_spec.included_in_data_plane_payload,
        "complete_payload_accounting": window.complete_payload_accounting(),
        "workload_identity_digest": window.workload_identity_digest(),
        "timing_profile_digest": window.timing_profile_digest(),
        "truth_digest": window.truth_digest(),
    }


def fixture_invariants(windows: tuple[TraceWindow, ...]) -> dict[str, Any]:
    if not windows:
        raise TraceValidationError("cannot generate invariants for empty windows")
    return {
        "scope": "trace_input_truth_only",
        "does_not_include_full_expected_timeline": True,
        "does_not_include_stable_event_ids": True,
        "does_not_include_plan_or_task_ids": True,
        "bootstrap_has_no_predecessor_p1": windows[0].is_bootstrap_p0,
        "only_first_window_is_bootstrap": all(not w.is_bootstrap_p0 for w in windows[1:]),
        "combine_expectation_unique_from_realized_dispatch": True,
        "descriptor_payload_thread_simultaneous_at_local_path_complete": True,
        "windows": [window_invariants(window) for window in windows],
    }


def validate_expected_invariants(fixture: FixtureInput) -> None:
    actual = fixture_invariants(fixture.windows)
    if actual != fixture.expected_invariants:
        raise TraceValidationError(
            "fixture expected_invariants mismatch; regenerate from trace truth. "
            f"expected_digest={stable_digest(fixture.expected_invariants)} actual_digest={stable_digest(actual)}"
        )
