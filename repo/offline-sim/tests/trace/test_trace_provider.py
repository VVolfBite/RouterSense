from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from rs_sim.trace.adapters.routesense_source_expert_counts import merge_source_expert_count_files
from rs_sim.trace.profiles.calibration import (
    ReceiverCalibrationDataset,
    ReceiverCalibrationProvenance,
    ReceiverCalibrationSample,
    calibration_summary,
    load_receiver_calibration_dataset,
    write_receiver_calibration_dataset,
)
from rs_sim.trace.schema.constants import PHASE_COMBINE, PHASE_DISPATCH
from rs_sim.trace.schema.fixtures import build_builtin_fixtures, write_builtin_fixtures
from rs_sim.trace.schema.invariants import matrix_transpose
from rs_sim.trace.schema.model import (
    LocalComputeProfile,
    PureComputeProvenance,
    TraceValidationError,
)
from rs_sim.trace.build.realization import realized_routing_from_token_assignments
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture, validate_fixture_paths


class TraceProviderTests(unittest.TestCase):
    def test_builtin_fixtures_validate_and_cover_all_splits(self) -> None:
        fixtures = build_builtin_fixtures()
        self.assertEqual(3, len(fixtures))
        self.assertEqual({"train", "validation", "test"}, {f.provenance.split for f in fixtures})
        for fixture in fixtures:
            self.assertEqual(4, fixture.world_size)
            self.assertEqual("PASS", validate_fixture(fixture)["status"])

    def test_combine_is_uniquely_derived_from_dispatch(self) -> None:
        for fixture in build_builtin_fixtures():
            for window in fixture.windows:
                self.assertEqual(matrix_transpose(window.dispatch_rows), window.combine_rows)
                self.assertEqual(
                    sum(sum(row) for row in window.dispatch_rows),
                    sum(sum(row) for row in window.combine_rows),
                )

    def test_realized_row_closure_for_clipping_and_padding(self) -> None:
        fixtures = {f.fixture_id: f for f in build_builtin_fixtures()}
        clipped = fixtures["ep4_validation_skew_clipped"]
        padded = fixtures["ep4_test_padding"]
        self.assertGreater(sum(w.routing.totals()["dropped_rows"] for w in clipped.windows), 0)
        self.assertGreater(sum(w.routing.totals()["padding_rows"] for w in padded.windows), 0)
        for fixture in fixtures.values():
            for window in fixture.windows:
                totals = window.routing.totals()
                self.assertEqual(totals["raw_selected_rows"], totals["kept_rows"] + totals["dropped_rows"])
                self.assertEqual(totals["realized_transfer_rows"], totals["kept_rows"] + totals["padding_rows"])

    def test_phase_specific_payload_is_deterministic_and_aligned(self) -> None:
        for fixture in build_builtin_fixtures():
            for window in fixture.windows:
                dispatch = window.payload_matrix(PHASE_DISPATCH)
                combine = window.payload_matrix(PHASE_COMBINE)
                self.assertEqual(dispatch, window.payload_matrix(PHASE_DISPATCH))
                self.assertEqual(combine, window.payload_matrix(PHASE_COMBINE))
                for row_index, row in enumerate(dispatch):
                    for col_index, value in enumerate(row):
                        if window.dispatch_rows[row_index][col_index] > 0:
                            self.assertEqual(0, value % window.dispatch_payload_spec.alignment_bytes)
                for row_index, row in enumerate(combine):
                    for col_index, value in enumerate(row):
                        if window.combine_rows[row_index][col_index] > 0:
                            self.assertEqual(0, value % window.combine_payload_spec.alignment_bytes)
                self.assertNotEqual(window.dispatch_payload_spec.digest(), window.combine_payload_spec.digest())

    def test_bootstrap_does_not_fabricate_predecessor_p1(self) -> None:
        for fixture in build_builtin_fixtures():
            self.assertTrue(fixture.windows[0].is_bootstrap_p0)
            self.assertFalse(fixture.initial_state.predecessor_p1_exists)
            self.assertTrue(all(not w.is_bootstrap_p0 for w in fixture.windows[1:]))

    def test_pure_compute_rejects_wait_components(self) -> None:
        with self.assertRaises(TraceValidationError):
            LocalComputeProfile(
                combine_release_to_router_ready_ns=(1, 1, 1, 1),
                router_and_pack_ns=(1, 1, 1, 1),
                dispatch_local_postprocess_ns=(1, 1, 1, 1),
                dispatch_release_to_combine_source_ready_ns=(1, 1, 1, 1),
                bootstrap_router_and_pack_ns=(1, 1, 1, 1),
                provenance=PureComputeProvenance(
                    measurement_method="bad",
                    source_artifact_digest="sha256:bad",
                    included_components=("network_wait",),
                ),
            )

    def test_round_trip_and_split_digest_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_builtin_fixtures(Path(temp_dir))
            report = validate_fixture_paths(paths, require_all_splits=True)
            self.assertEqual("PASS", report["status"])
            loaded = load_fixture(paths[0])
            self.assertEqual("PASS", validate_fixture(loaded)["status"])
            digest_sets = report["split_identity_digests"]
            flattened = [value for values in digest_sets.values() for value in values]
            self.assertEqual(len(flattened), len(set(flattened)))




    def test_receiver_calibration_provenance_round_trip_without_model_fit(self) -> None:
        provenance = ReceiverCalibrationProvenance(
            dataset_id="calibration-test",
            source_digest="sha256:source",
            environment_digest="sha256:environment",
            capture_id="capture-test",
            collector_version="0.1.0",
            measurement_method="isolated-segment",
        )
        samples = (
            ReceiverCalibrationSample(
                sample_id="posting-0", component="receiver_posting_service", payload_bytes=4096,
                observed_service_ns=100, repetition_index=0, src_rank=0, dst_rank=1,
                node_relation="same_node", metadata={},
            ),
            ReceiverCalibrationSample(
                sample_id="drain-0", component="receiver_drain", payload_bytes=4096,
                observed_service_ns=200, repetition_index=0, src_rank=0, dst_rank=1,
                node_relation="same_node", metadata={},
            ),
        )
        dataset = ReceiverCalibrationDataset(provenance=provenance, samples=samples)
        self.assertFalse(calibration_summary(dataset)["model_fit_performed"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            write_receiver_calibration_dataset(path, dataset)
            loaded = load_receiver_calibration_dataset(path)
            self.assertEqual(dataset.digest(), loaded.digest())

    def test_receiver_calibration_rejects_wait_contamination(self) -> None:
        with self.assertRaises(TraceValidationError):
            ReceiverCalibrationProvenance(
                dataset_id="bad", source_digest="sha256:bad", environment_digest="sha256:bad-env",
                capture_id="bad", collector_version="0.1.0", measurement_method="bad",
                queue_wait_included=True,
            )

    def test_token_assignment_converter_preserves_explicit_keep_drop_truth(self) -> None:
        routing = realized_routing_from_token_assignments(
            selected_experts_by_source=[
                [[0, 2], [0, 3]],
                [[1, 2]],
                [[4, 6]],
                [[5, 7]],
            ],
            kept_mask_by_source=[
                [[True, False], [True, True]],
                [[False, True]],
                [[True, True]],
                [[True, False]],
            ],
            num_experts=8,
            padding_rows_by_source_expert=[
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
        )
        totals = routing.totals()
        self.assertEqual(10, totals["raw_selected_rows"])
        self.assertEqual(7, totals["kept_rows"])
        self.assertEqual(3, totals["dropped_rows"])
        self.assertEqual(3, totals["padding_rows"])
        self.assertEqual(10, totals["realized_transfer_rows"])

    def test_local_path_formula_and_bootstrap_formula(self) -> None:
        fixture = build_builtin_fixtures()[0]
        compute = fixture.windows[0].local_compute
        regular = compute.source_local_path_times(rank=2, p1_local_complete_ns=10_000)
        self.assertEqual(10_000 + compute.combine_release_to_router_ready_ns[2], regular["post_combine_local_path_complete_ns"])
        self.assertEqual(regular["post_combine_local_path_complete_ns"] + compute.router_and_pack_ns[2], regular["local_path_complete_ns"])
        self.assertEqual(regular["local_path_complete_ns"], regular["source_descriptor_ready_ns"])
        self.assertEqual(regular["local_path_complete_ns"], regular["source_payload_ready_ns"])
        self.assertEqual(regular["local_path_complete_ns"], regular["destination_dispatch_thread_ready_ns"])
        bootstrap = compute.bootstrap_source_ready_times(rank=2, bootstrap_start_ns=5_000)
        self.assertEqual(5_000 + compute.bootstrap_router_and_pack_ns[2], bootstrap["bootstrap_local_path_complete_ns"])
        self.assertEqual(0, bootstrap["predecessor_p1_exists"])
        self.assertEqual(20_000 + compute.dispatch_release_to_combine_source_ready_ns[2], compute.combine_source_ready_at_ns(rank=2, dispatch_release_ns=20_000))


    def test_fixture_invariants_do_not_claim_integration_owned_timeline(self) -> None:
        forbidden_fragments = (
            "task_id", "plan_id", "stable_event_id", "transport_start",
            "transport_finish", "golden_timeline", "final_expected_digest",
        )

        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    lowered = str(key).lower()
                    if not lowered.startswith("does_not_include_"):
                        for fragment in forbidden_fragments:
                            self.assertNotIn(fragment, lowered)
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        for fixture in build_builtin_fixtures():
            walk(fixture.expected_invariants)
            self.assertTrue(fixture.expected_invariants["does_not_include_full_expected_timeline"])
            self.assertTrue(fixture.expected_invariants["does_not_include_stable_event_ids"])
            self.assertTrue(fixture.expected_invariants["does_not_include_plan_or_task_ids"])

    def test_fixture_serialization_is_stable_across_100_builds(self) -> None:
        digests = []
        for _ in range(100):
            digests.append(tuple(f.truth_digest() for f in build_builtin_fixtures()))
        self.assertEqual(1, len(set(digests)))

    def test_routesense_adapter_merges_rank_rows_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for rank in range(4):
                counts = [[0] * 8 for _ in range(4)]
                counts[rank][rank * 2] = rank + 1
                payload = {
                    "layer_id": 0,
                    "world_size": 4,
                    "num_experts": 8,
                    "rank": rank,
                    "source_rank": rank,
                    "counts": counts,
                    "expert_to_rank_map": [0, 0, 1, 1, 2, 2, 3, 3],
                }
                path = Path(temp_dir) / f"rank{rank}_source_expert_counts.jsonl"
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                paths.append(path)
            merged = merge_source_expert_count_files(paths, rank_to_node=(0, 0, 1, 1))
            mapping, routing = merged[0]
            self.assertEqual(4, mapping.world_size)
            self.assertEqual("routesense_already_realized_source_expert_counts", routing.realization_origin)
            self.assertEqual(routing.raw_selected_rows, routing.kept_rows)
            self.assertEqual(0, routing.totals()["dropped_rows"])
            self.assertEqual(0, routing.totals()["padding_rows"])


if __name__ == "__main__":
    unittest.main()


def test_local_remote_payload_accounting_closes_for_all_builtin_fixtures():
    from rs_sim import PhaseKey, PhaseKind
    from rs_sim.runtime.adapters.trace import phase_payload_accounting

    for fixture in build_builtin_fixtures():
        for window in fixture.windows:
            sample_id = f"{window.request_id}:step{window.decode_step}"
            for phase_kind in (PhaseKind.DISPATCH, PhaseKind.COMBINE):
                phase_key = PhaseKey(
                    run_id="audit",
                    sample_id=sample_id,
                    layer_index=window.layer_id,
                    phase_kind=phase_kind,
                )
                accounting = phase_payload_accounting(window, phase_key)
                assert accounting["local_payload_bytes"] + accounting["remote_network_payload_bytes"] == accounting["total_payload_bytes"]


def test_dispatch_and_combine_payload_truth_are_independently_derived():
    from rs_sim import PhaseKey, PhaseKind
    from rs_sim.runtime.adapters.trace import dispatch_row_truth_for_trace_window

    found_difference = False
    for fixture in build_builtin_fixtures():
        for window in fixture.windows:
            phase_key = PhaseKey(
                run_id="audit",
                sample_id=f"{window.request_id}:step{window.decode_step}",
                layer_index=window.layer_id,
                phase_kind=PhaseKind.DISPATCH,
            )
            for src_rank in range(window.mapping.world_size):
                truth = dispatch_row_truth_for_trace_window(
                    trace_window=window,
                    phase_key=phase_key,
                    src_rank=src_rank,
                    descriptor_entry_bytes=8,
                    descriptor_fixed_header_bytes=16,
                )
                expected_dispatch = tuple(window.payload_matrix(PHASE_DISPATCH)[src_rank])
                expected_combine = tuple(
                    window.payload_matrix(PHASE_COMBINE)[expert_rank][src_rank]
                    for expert_rank in range(window.mapping.world_size)
                )
                assert truth["dispatch_payload_bytes_by_destination"] == expected_dispatch
                assert truth["combine_return_payload_bytes_by_expert"] == expected_combine
                if expected_dispatch != expected_combine:
                    found_difference = True
    assert found_difference


def test_descriptor_metadata_is_control_plane_only_and_fixed_plus_per_entry():
    from rs_sim.runtime.adapters.trace import descriptor_metadata_accounting_for_trace_window

    window = build_builtin_fixtures()[0].windows[0]
    accounting = descriptor_metadata_accounting_for_trace_window(window)
    assert accounting["accounting_plane"] == "CONTROL_PLANE"
    assert accounting["included_in_data_plane_payload"] is False
    assert accounting["descriptor_payload_bytes"] == (
        accounting["fixed_header_bytes"]
        + window.mapping.world_size * accounting["per_destination_entry_bytes"]
    )
    for phase in (PHASE_DISPATCH, PHASE_COMBINE):
        for row_count in (0, 1, 7):
            spec = window.dispatch_payload_spec if phase == PHASE_DISPATCH else window.combine_payload_spec
            layout = spec.edge_payload_layout(row_count)
            assert layout["metadata_accounting_plane"] == "DATA_PLANE"
            assert layout["control_plane_descriptor_bytes"] == 0


def test_descriptor_identity_is_independent_of_publication_and_compute_time():
    from dataclasses import replace

    from rs_sim import PhaseKind
    from rs_sim.runtime.adapters.trace import exact_row_descriptor_for_trace_window, keys_for_trace_window

    window = build_builtin_fixtures()[0].windows[0]
    phase_key = keys_for_trace_window(run_id="descriptor-time-independence", trace_window=window).dispatch_phase_key
    first = exact_row_descriptor_for_trace_window(
        trace_window=window, phase_key=phase_key, src_rank=0, published_at_ns=100
    )
    second = exact_row_descriptor_for_trace_window(
        trace_window=window, phase_key=phase_key, src_rank=0, published_at_ns=99_999
    )
    changed_compute = replace(
        window,
        local_compute=replace(
            window.local_compute,
            router_and_pack_ns=tuple(value + 1_000_000 for value in window.local_compute.router_and_pack_ns),
        ),
    )
    third = exact_row_descriptor_for_trace_window(
        trace_window=changed_compute, phase_key=phase_key, src_rank=0, published_at_ns=500
    )
    assert phase_key.phase_kind is PhaseKind.DISPATCH
    assert first.published_at_ns != second.published_at_ns
    assert first.descriptor_digest == second.descriptor_digest == third.descriptor_digest


def test_local_diagonal_truth_is_retained_but_excluded_from_remote_network_bytes():
    for fixture in build_builtin_fixtures():
        for window in fixture.windows:
            for phase in (PHASE_DISPATCH, PHASE_COMBINE):
                rows = window.dispatch_rows if phase == PHASE_DISPATCH else window.combine_rows
                payload = window.payload_matrix(phase)
                accounting = window.payload_accounting(phase)
                assert accounting["local_rows"] == sum(rows[rank][rank] for rank in range(window.mapping.world_size))
                assert accounting["local_payload_bytes"] == sum(payload[rank][rank] for rank in range(window.mapping.world_size))
                assert accounting["remote_network_payload_bytes"] == sum(
                    payload[src][dst]
                    for src in range(window.mapping.world_size)
                    for dst in range(window.mapping.world_size)
                    if src != dst
                )
                assert accounting["local_payload_bytes"] + accounting["remote_network_payload_bytes"] == accounting["total_payload_bytes"]


def test_split_usage_policy_prevents_test_leakage_into_training_or_tuning():
    from rs_sim.trace.schema.validation import validate_fixture_usage

    fixtures = {fixture.provenance.split: fixture for fixture in build_builtin_fixtures()}
    assert validate_fixture_usage((fixtures["train"],), purpose="training")["status"] == "PASS"
    assert validate_fixture_usage((fixtures["validation"],), purpose="tuning")["status"] == "PASS"
    assert validate_fixture_usage((fixtures["test"],), purpose="final_evaluation")["status"] == "PASS"
    with pytest.raises(TraceValidationError):
        validate_fixture_usage((fixtures["test"],), purpose="training")
    with pytest.raises(TraceValidationError):
        validate_fixture_usage((fixtures["test"],), purpose="tuning")


def test_trace_bridge_exposes_only_immutable_rank_to_node_input_not_guessed_topology():
    from rs_sim.runtime.adapters import trace as trace_bridge

    window = build_builtin_fixtures()[0].windows[0]
    assert trace_bridge.rank_to_node_input_for_trace_window(window) == (0, 0, 1, 1)
    assert not hasattr(trace_bridge, "topology_for_trace_window")


def test_checked_in_receiver_calibration_example_is_format_only():
    from rs_sim.trace.profiles.calibration import calibration_summary, load_receiver_calibration_dataset

    path = Path("fixtures/trace/calibration/receiver_posting_drain_format_example.json")
    summary = calibration_summary(load_receiver_calibration_dataset(path))
    assert summary["status"] == "PASS"
    assert summary["model_fit_performed"] is False
    assert summary["performance_eligible"] is False
    assert summary["hardware_profile_calibrated"] is False
