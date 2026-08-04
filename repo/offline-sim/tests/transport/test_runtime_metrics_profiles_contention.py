from __future__ import annotations

import json

import pytest

from rs_sim import (
    LinkClass,
    SimulationKernel,
    SubmitOutcome,
    make_control_plane_profile,
    make_hardware_profile,
)
from rs_sim.transport import (
    PROFILE_KIND_CALIBRATED,
    StaticTransportProfileProvider,
    build_formal_transport_runtime_driver,
    load_transport_profile_bundle_json,
    make_calibrated_transport_profile_bundle,
    make_synthetic_profile_sensitivity_set,
    visible_fabric_equal_share_sensitivity,
    write_transport_profile_bundle_json,
)

from .conftest import ControlLog, build_harness
from .test_control_plane import profile as control_profile
from .test_control_plane import request


def _run(kernel: SimulationKernel) -> None:
    while kernel.has_events():
        kernel.run_next_timestamp()


def _zero_latency_profile(*, profile_id: str):
    return make_hardware_profile(
        profile_id=profile_id,
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        max_batch_tasks=4,
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 0),
            (LinkClass.INTER_NODE, 0),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 0),
            (LinkClass.INTER_NODE, 0),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, 1_000_000_000),
            (LinkClass.INTER_NODE, 1_000_000_000),
        ),
    )


def test_data_and_control_runtime_metrics_reconcile_exactly():
    harness = build_harness()
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("t0", "t1"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    _run(harness.kernel)
    harness.transport.assert_terminal()
    harness.transport.assert_statistics_reconcile()

    stats = harness.transport.statistics()
    assert stats["schema_version"] == "TRANSPORT_DATA_PLANE_STATISTICS"
    assert stats["total_launch_count"] == 1
    assert stats["total_launch_cost_ns"] == 3
    assert stats["physical_started_record_count"] == 2
    assert stats["physical_completed_record_count"] == 2
    assert stats["physical_completed_bytes"] == 24
    assert stats["statistics_reconciliation"]["reconciled"] is True
    assert len(stats["transfer_timeline"]) == 2
    assert all(row[10] == 1 for row in stats["transfer_timeline"])
    assert stats["bandwidth_contention_mode"] == "FIXED_PER_LANE"
    assert stats["contention_affected_task_count"] == 0
    runtime_metrics = harness.transport.formal_runtime_metrics()
    assert runtime_metrics["terminal_resource_evidence"]["terminal"] is True
    assert runtime_metrics["runtime_metrics_digest"] == (
        harness.transport.formal_runtime_metrics()["runtime_metrics_digest"]
    )

    kernel = SimulationKernel()
    sink = ControlLog()
    from rs_sim.transport import FormalControlPlaneTransport

    control = FormalControlPlaneTransport(
        kernel=kernel, profile=control_profile(), delivery_sink=sink
    )
    control.publish_row(request(src_rank=0, published_at_ns=0, payload_bytes=10))
    control.publish_row(request(src_rank=1, published_at_ns=0, payload_bytes=20))
    _run(kernel)
    control.assert_terminal()
    control.assert_statistics_reconcile()
    control_stats = control.statistics()
    assert control_stats["schema_version"] == "TRANSPORT_CONTROL_PLANE_STATISTICS"
    assert control_stats["delivered_request_count"] == 2
    assert control_stats["delivered_payload_bytes"] == 30
    assert control_stats["statistics_reconciliation"]["reconciled"] is True
    assert len(control_stats["delivery_timeline"]) == 2
    control_runtime = control.formal_runtime_metrics()
    assert control_runtime["terminal_resource_evidence"]["terminal"] is True
    assert control_runtime["runtime_metrics_digest"] == (
        control.formal_runtime_metrics()["runtime_metrics_digest"]
    )


def test_explicit_visible_fabric_contention_changes_only_service_time():
    task_specs = (("a", 0, 2, 10), ("b", 1, 3, 10))
    fixed = build_harness(
        task_specs=task_specs,
        profile=_zero_latency_profile(profile_id="fixed"),
    )
    sensitive = build_harness(
        task_specs=task_specs,
        profile=_zero_latency_profile(profile_id="sensitive"),
        bandwidth_contention=visible_fabric_equal_share_sensitivity(),
    )
    for harness in (fixed, sensitive):
        outcome, receipt = harness.transport.prepare_commit(
            harness.batch("a", "b"), 0
        )
        assert outcome is SubmitOutcome.PREPARED and receipt is not None
        harness.transport.confirm_commit(receipt)
        _run(harness.kernel)
        harness.transport.assert_terminal()

    fixed_records = fixed.transport.physical_records()
    sensitive_records = sensitive.transport.physical_records()
    assert tuple(record.task_id for record in fixed_records) == tuple(
        record.task_id for record in sensitive_records
    )
    assert sum(record.payload_bytes for record in fixed_records) == 20
    assert sum(record.payload_bytes for record in sensitive_records) == 20
    assert {record.complete_at_ns for record in fixed_records} == {10}
    assert {record.complete_at_ns for record in sensitive_records} == {20}

    fixed_stats = fixed.transport.statistics()
    sensitive_stats = sensitive.transport.statistics()
    assert fixed_stats["contention_affected_task_count"] == 0
    assert fixed_stats["contention_extra_service_ns"] == 0
    assert sensitive_stats["contention_affected_task_count"] == 2
    assert sensitive_stats["contention_extra_service_ns"] == 20
    assert sensitive_stats["max_contention_share_count"] == 2
    manifest = sensitive.transport.manifest_fragment()
    assert manifest["bandwidth_contention_enabled"] is True
    assert manifest["bandwidth_contention_adds_hidden_resources"] is False
    assert manifest["bandwidth_contention_changes_admission"] is False
    assert sensitive.transport.terminal_state()["internal_wait_queue_depth"] == 0


def test_synthetic_profile_bundle_round_trip_and_provider_handoff(tmp_path):
    profile_set = make_synthetic_profile_sensitivity_set(
        profile_set_id="r25-synthetic",
        profile_provenance="SYNTHETIC_ONLY",
        max_batch_tasks=8,
        bandwidth_contention=visible_fabric_equal_share_sensitivity(),
    )
    bundle = profile_set.as_profile_bundle()
    path = tmp_path / "synthetic_transport_profile_bundle.json"
    write_transport_profile_bundle_json(path, bundle)
    loaded = load_transport_profile_bundle_json(path)
    assert loaded == bundle
    assert loaded.performance_eligible is False
    assert loaded.manifest_fragment()["transport_profile_kind"] == "SYNTHETIC"
    assert loaded.manifest_fragment()["bandwidth_contention_enabled"] is True

    source = build_harness()
    driver = build_formal_transport_runtime_driver(
        kernel=SimulationKernel(),
        task_lookup=source.transport.task_lookup,
        permit_lookup=source.transport.permit_lookup,
        authority_validation=source.transport.authority_validation,
        resource_resolver=source.resolver,
        completion_sink=source.completion,
        resource_release_sink=source.release,
        control_delivery_sink=ControlLog(),
        profile_provider=StaticTransportProfileProvider(loaded),
    )
    manifest = driver.manifest_fragment()
    assert manifest["transport_profile_bundle_id"] == "r25-synthetic"
    assert manifest["transport_profile_bundle_digest"] == loaded.bundle_digest
    assert manifest["transport_performance_eligible"] is False
    assert manifest["bandwidth_contention_enabled"] is True


def test_calibrated_profile_bundle_requires_external_sources_and_preserves_eligibility(
    tmp_path,
):
    hardware = make_hardware_profile(
        profile_id="measured-data-fit-v1",
        profile_provenance="MEASURED_CALIBRATION_RUN_42",
        performance_eligible=True,
        max_batch_tasks=8,
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 12),
            (LinkClass.INTER_NODE, 34),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 56),
            (LinkClass.INTER_NODE, 78),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, 90_000_000_000),
            (LinkClass.INTER_NODE, 20_000_000_000),
        ),
    )
    control = make_control_plane_profile(
        profile_id="measured-control-fit-v1",
        profile_provenance="MEASURED_CALIBRATION_RUN_42",
        performance_eligible=True,
        fixed_latency_ns=44,
        bandwidth_bytes_per_second=8_000_000_000,
    )
    bundle = make_calibrated_transport_profile_bundle(
        bundle_id="calibrated-r42",
        profile_provenance="MEASURED_CALIBRATION_RUN_42",
        source_digests=("sha256:data-source", "sha256:control-source"),
        hardware_profile=hardware,
        control_profile=control,
        assumptions=(
            "EXTERNAL_FIT_SUPPLIED_BY_CALIBRATION_PIPELINE",
            "QUEUE_WAIT_EXCLUDED",
        ),
    )
    assert bundle.profile_kind == PROFILE_KIND_CALIBRATED
    assert bundle.performance_eligible is True
    path = tmp_path / "calibrated_transport_profile_bundle.json"
    write_transport_profile_bundle_json(path, bundle)
    loaded = load_transport_profile_bundle_json(path)
    assert loaded == bundle
    assert loaded.manifest_fragment()["transport_profile_source_digests"] == (
        "sha256:control-source",
        "sha256:data-source",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle_digest"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle JSON digest mismatch"):
        load_transport_profile_bundle_json(path)

    with pytest.raises(ValueError, match="require source_digests"):
        make_calibrated_transport_profile_bundle(
            bundle_id="invalid",
            profile_provenance="MEASURED_CALIBRATION_RUN_42",
            source_digests=(),
            hardware_profile=hardware,
            control_profile=control,
        )


def test_profile_bundle_loader_rejects_non_boolean_eligibility_and_calibrated_synthetic_components(tmp_path):
    profile_set = make_synthetic_profile_sensitivity_set(
        profile_set_id="strict-json",
        profile_provenance="SYNTHETIC_STRICT_JSON",
        max_batch_tasks=4,
    )
    path = tmp_path / "strict.json"
    write_transport_profile_bundle_json(path, profile_set.as_profile_bundle())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hardware_profile"]["performance_eligible"] = "false"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        load_transport_profile_bundle_json(path)

    with pytest.raises(ValueError, match="component provenance cannot be synthetic"):
        make_calibrated_transport_profile_bundle(
            bundle_id="laundered",
            profile_provenance="MEASURED_WRAPPER",
            source_digests=("sha256:source",),
            hardware_profile=profile_set.hardware_profile,
            control_profile=profile_set.control_profile,
        )
