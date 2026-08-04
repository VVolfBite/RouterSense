"""High-level fixture and provenance validators."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .constants import ALLOWED_SPLITS_BY_PURPOSE, PHASE_COMBINE, PHASE_DISPATCH, VALID_DATASET_PURPOSES, VALID_DATASET_SPLITS
from .invariants import matrix_sum, validate_expected_invariants
from .model import FixtureInput, TraceValidationError
from ..io.serialization import load_fixture


def validate_fixture(fixture: FixtureInput) -> dict[str, object]:
    validate_expected_invariants(fixture)
    errors: list[str] = []
    for window in fixture.windows:
        totals = window.routing.totals()
        if totals["raw_selected_rows"] != totals["kept_rows"] + totals["dropped_rows"]:
            errors.append(f"{window.window_id}: raw row closure failed")
        if totals["realized_transfer_rows"] != totals["kept_rows"] + totals["padding_rows"]:
            errors.append(f"{window.window_id}: realized row closure failed")
        dispatch = window.dispatch_rows
        combine = window.combine_rows
        expected_combine = tuple(tuple(dispatch[dst][src] for dst in range(window.mapping.world_size)) for src in range(window.mapping.world_size))
        if combine != expected_combine:
            errors.append(f"{window.window_id}: combine is not uniquely derived transpose")
        # Recompute payload twice to assert determinism and valid alignment.
        for phase in (PHASE_DISPATCH, PHASE_COMBINE):
            first = window.payload_matrix(phase)
            second = window.payload_matrix(phase)
            if first != second:
                errors.append(f"{window.window_id}: {phase} payload bytes are non-deterministic")
            spec = window.dispatch_payload_spec if phase == PHASE_DISPATCH else window.combine_payload_spec
            rows = dispatch if phase == PHASE_DISPATCH else combine
            for src, row in enumerate(first):
                for dst, value in enumerate(row):
                    if rows[src][dst] == 0 and value != 0:
                        errors.append(f"{window.window_id}: zero edge has nonzero {phase} payload")
                    if rows[src][dst] > 0 and spec.padding_rule != "NONE" and value % spec.alignment_bytes != 0:
                        errors.append(f"{window.window_id}: {phase} edge ({src},{dst}) is not aligned")
        for phase in (PHASE_DISPATCH, PHASE_COMBINE):
            accounting = window.payload_accounting(phase)
            if accounting["local_payload_bytes"] + accounting["remote_network_payload_bytes"] != accounting["total_payload_bytes"]:
                errors.append(f"{window.window_id}: {phase} local/remote payload accounting does not close")
            if accounting["local_rows"] + accounting["remote_rows"] != accounting["total_rows"]:
                errors.append(f"{window.window_id}: {phase} local/remote row accounting does not close")
        if window.descriptor_metadata_spec.included_in_data_plane_payload:
            errors.append(f"{window.window_id}: descriptor metadata double-counted in DataPlane")
        if window.is_bootstrap_p0:
            if window.window_id != fixture.initial_state.bootstrap_window_id:
                errors.append(f"{window.window_id}: bootstrap id mismatch")
    if fixture.initial_state.predecessor_p1_exists:
        errors.append("bootstrap P0 fabricates predecessor P1")
    if errors:
        raise TraceValidationError("; ".join(errors))
    return {
        "fixture_id": fixture.fixture_id,
        "world_size": fixture.world_size,
        "window_count": len(fixture.windows),
        "split": fixture.provenance.split,
        "fixture_truth_digest": fixture.truth_digest(),
        "total_dispatch_rows": sum(matrix_sum(window.dispatch_rows) for window in fixture.windows),
        "total_combine_rows": sum(matrix_sum(window.combine_rows) for window in fixture.windows),
        "status": "PASS",
    }


def validate_fixture_paths(paths: Iterable[Path], *, require_all_splits: bool = False) -> dict[str, object]:
    fixtures = [load_fixture(path) for path in paths]
    results = [validate_fixture(fixture) for fixture in fixtures]
    digests_by_split: dict[str, set[str]] = defaultdict(set)
    digest_to_splits: dict[str, set[str]] = defaultdict(set)
    for fixture in fixtures:
        digest = fixture.provenance.identity_digest()
        split = fixture.provenance.split
        digests_by_split[split].add(digest)
        digest_to_splits[digest].add(split)
    collisions = {digest: sorted(splits) for digest, splits in digest_to_splits.items() if len(splits) > 1}
    if collisions:
        raise TraceValidationError(f"train/validation/test provenance digest collision: {collisions}")
    present_splits = set(digests_by_split)
    if require_all_splits and present_splits != set(VALID_DATASET_SPLITS):
        raise TraceValidationError(
            f"required split set {sorted(VALID_DATASET_SPLITS)}, found {sorted(present_splits)}"
        )
    return {
        "status": "PASS",
        "fixture_count": len(fixtures),
        "present_splits": sorted(present_splits),
        "split_identity_digests": {split: sorted(values) for split, values in sorted(digests_by_split.items())},
        "fixtures": results,
    }


def validate_fixture_usage(fixtures: Iterable[FixtureInput], *, purpose: str) -> dict[str, object]:
    """Fail closed on split leakage into training, tuning, or final evaluation."""
    normalized = str(purpose)
    if normalized not in VALID_DATASET_PURPOSES:
        raise TraceValidationError(f"unsupported dataset purpose={purpose!r}")
    allowed = ALLOWED_SPLITS_BY_PURPOSE[normalized]
    fixtures = tuple(fixtures)
    disallowed = sorted({fixture.provenance.split for fixture in fixtures if fixture.provenance.split not in allowed})
    if disallowed:
        raise TraceValidationError(
            f"dataset split leakage for purpose={normalized}: allowed={sorted(allowed)}, found_disallowed={disallowed}"
        )
    return {
        "status": "PASS",
        "purpose": normalized,
        "allowed_splits": sorted(allowed),
        "fixture_ids": [fixture.fixture_id for fixture in fixtures],
        "splits": sorted({fixture.provenance.split for fixture in fixtures}),
    }
