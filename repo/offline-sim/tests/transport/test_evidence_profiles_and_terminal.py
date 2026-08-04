from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim import (
    LinkClass,
    SimulationKernel,
    SubmitOutcome,
    make_control_plane_profile,
    make_hardware_profile,
    make_network_topology,
    make_transfer_batch,
)
from rs_sim.transport import ReceiptStateError, build_formal_transports

from .conftest import ControlLog, build_harness
from .test_control_plane import request


def _run(kernel: SimulationKernel) -> None:
    while kernel.has_events():
        kernel.run_next_timestamp()


def _link_stats(stats):
    return {
        link_class: dict(values)
        for link_class, values in stats["link_class_statistics"]
    }


def test_bundle_exports_stable_manifest_evidence_and_terminal_check():
    source = build_harness()
    kernel = SimulationKernel()
    control_sink = ControlLog()
    bundle = build_formal_transports(
        kernel=kernel,
        task_lookup=source.transport.task_lookup,
        permit_lookup=source.transport.permit_lookup,
        authority_validation=source.transport.authority_validation,
        resource_resolver=source.resolver,
        completion_sink=source.completion,
        resource_release_sink=source.release,
        control_delivery_sink=control_sink,
        hardware_profile=source.transport.hardware_profile,
    )
    assert bundle.terminal_state()["terminal"] is True

    bundle.control_plane.publish_row(
        request(src_rank=0, published_at_ns=5, payload_bytes=10)
    )
    outcome, receipt = bundle.data_plane.prepare_commit(source.batch("t0"), 0)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    assert bundle.terminal_state()["terminal"] is False
    bundle.data_plane.confirm_commit(receipt)
    _run(kernel)

    bundle.assert_terminal()
    evidence = bundle.evidence()
    manifest = evidence["manifest_fragment"]
    assert manifest["transport_transport"] is True
    assert manifest["transport_evidence_version"] == "TRANSPORT_BUNDLE_EVIDENCE"
    assert manifest["transport_terminal_check_supported"] is True
    assert manifest["transport_performance_eligible"] is False
    assert evidence["terminal_state"]["terminal"] is True
    assert evidence["data_plane"]["terminal_state"]["prepared_count"] == 0
    assert evidence["data_plane"]["terminal_state"]["active_transfer_count"] == 0
    assert evidence["data_plane"]["terminal_state"]["all_resources_free"] is True
    assert evidence["data_plane"]["statistics"]["physical_completed_bytes"] == 8
    assert evidence["control_plane"]["statistics"]["delivered_payload_bytes"] == 10


def test_control_terminal_state_counts_future_arrivals_not_only_active_delivery():
    source = build_harness()
    kernel = SimulationKernel()
    sink = ControlLog()
    bundle = build_formal_transports(
        kernel=kernel,
        task_lookup=source.transport.task_lookup,
        permit_lookup=source.transport.permit_lookup,
        authority_validation=source.transport.authority_validation,
        resource_resolver=source.resolver,
        completion_sink=source.completion,
        resource_release_sink=source.release,
        control_delivery_sink=sink,
        hardware_profile=source.transport.hardware_profile,
    )
    bundle.control_plane.publish_row(request(src_rank=0, published_at_ns=100))
    state = bundle.control_plane.terminal_state()
    assert state["terminal"] is False
    assert state["pending_arrival_count"] == 1
    _run(kernel)
    assert bundle.control_plane.terminal_state()["terminal"] is True


class _SnapshotCompletion:
    def __init__(self, transport):
        self.transport = transport
        self.busy_during_completion = []

    def on_transfer_started(self, event):
        del event

    def on_transfer_completed(self, event):
        del event
        snapshot = self.transport.snapshot()
        self.busy_during_completion.append(
            (snapshot.busy_src_ranks, snapshot.busy_dst_ranks, snapshot.busy_lane_ids)
        )


def test_completion_callback_observes_resources_before_release():
    harness = build_harness(task_specs=(("a", 0, 2, 8),))
    sink = _SnapshotCompletion(harness.transport)
    harness.transport.completion_sink = sink
    outcome, receipt = harness.transport.prepare_commit(harness.batch("a"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    _run(harness.kernel)
    assert sink.busy_during_completion == [((0,), (2,), ("inter-0",))]
    assert harness.transport.terminal_state()["all_resources_free"] is True
    assert harness.release.phases == [harness.phase]


def test_repeated_abort_double_confirm_and_confirm_after_abort():
    harness = build_harness()
    outcome, aborted = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and aborted is not None
    harness.transport.abort_commit(aborted)
    harness.transport.abort_commit(aborted)
    with pytest.raises(ReceiptStateError):
        harness.transport.confirm_commit(aborted)

    outcome, confirmed = harness.transport.prepare_commit(harness.batch("t1"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and confirmed is not None
    harness.transport.confirm_commit(confirmed)
    with pytest.raises(ReceiptStateError):
        harness.transport.confirm_commit(confirmed)
    assert harness.transport.statistics()["aborted_commit_count"] == 1
    assert harness.transport.statistics()["confirmed_commit_count"] == 1


def test_intra_and_inter_node_statistics_and_independent_launches():
    topology = make_network_topology(
        topology_id="mixed-statistics",
        rank_to_node=(0, 0, 1, 2),
        tx_nic_id_by_rank=("n0-tx", "n0-tx", "n1-tx", "n2-tx"),
        rx_nic_id_by_rank=("n0-rx", "n0-rx", "n1-rx", "n2-rx"),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra-0", "intra-1")),
            (LinkClass.INTER_NODE, ("inter-0", "inter-1")),
        ),
        nic_id_by_lane=(
            ("intra-0", "fabric-intra"),
            ("intra-1", "fabric-intra"),
            ("inter-0", "fabric-inter"),
            ("inter-1", "fabric-inter"),
        ),
    )
    harness = build_harness(
        topology=topology,
        task_specs=(("intra", 0, 1, 10), ("inter", 2, 3, 20)),
    )
    out_i, rec_i = harness.transport.prepare_commit(
        harness.batch("intra", link_class=LinkClass.INTRA_NODE), harness.transport.kernel.now_ns)
    assert out_i is SubmitOutcome.PREPARED and rec_i is not None
    harness.transport.confirm_commit(rec_i)
    out_e, rec_e = harness.transport.prepare_commit(
        harness.batch("inter", link_class=LinkClass.INTER_NODE), harness.transport.kernel.now_ns)
    assert out_e is SubmitOutcome.PREPARED and rec_e is not None
    harness.transport.confirm_commit(rec_e)
    _run(harness.kernel)

    link_stats = _link_stats(harness.transport.statistics())
    assert link_stats["INTRA_NODE"]["launch_count"] == 1
    assert link_stats["INTRA_NODE"]["completed_bytes"] == 10
    assert link_stats["INTER_NODE"]["launch_count"] == 1
    assert link_stats["INTER_NODE"]["completed_bytes"] == 20
    lane_rows = {
        lane: busy for lane, busy, _ in harness.transport.statistics()["lane_utilization_rational"]
    }
    assert lane_rows["intra-0"] > 0
    assert lane_rows["inter-0"] > 0
    assert harness.transport.terminal_state()["physical_completed_bytes"] == 30


def test_resource_busy_has_no_internal_queue_and_requires_explicit_recompile():
    harness = build_harness(task_specs=(("a", 0, 2, 8), ("b", 0, 3, 8)))
    outcome, receipt = harness.transport.prepare_commit(harness.batch("a"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    outcome, rejected = harness.transport.prepare_commit(harness.batch("b"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.RETRYABLE_RESOURCE_BUSY and rejected is None
    assert tuple(record.task_id for record in harness.transport.physical_records()) == ("a",)
    assert harness.transport.statistics()["resource_wait_retry_counts"] == (
        ("RANK_ENDPOINT", 1),
    )
    _run(harness.kernel)
    # transport did not retain or auto-launch the rejected batch.
    assert tuple(record.task_id for record in harness.transport.physical_records()) == ("a",)

    recompiled = make_transfer_batch(
        batch_id="batch:b:recompiled",
        phase_key=harness.phase,
        task_ids=("b",),
        authority_stamp=harness.stamp,
        link_class=LinkClass.INTER_NODE,
        topology_digest=harness.resolver.topology.topology_digest,
        compiled_at_ns=harness.kernel.now_ns,
    )
    outcome, receipt = harness.transport.prepare_commit(
        recompiled, harness.kernel.now_ns
    )
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    _run(harness.kernel)
    assert tuple(record.task_id for record in harness.transport.physical_records()) == (
        "a",
        "b",
    )


def test_profile_sensitivity_changes_time_without_changing_provenance_boundary():
    def profile(profile_id, fixed_ns, bandwidth):
        return make_hardware_profile(
            profile_id=profile_id,
            profile_provenance="SYNTHETIC_TEST_ONLY",
            performance_eligible=False,
            max_batch_tasks=4,
            launch_delay_ns_by_link_class=(
                (LinkClass.INTRA_NODE, 1),
                (LinkClass.INTER_NODE, 1),
            ),
            fixed_latency_ns_by_link_class=(
                (LinkClass.INTRA_NODE, fixed_ns),
                (LinkClass.INTER_NODE, fixed_ns),
            ),
            bandwidth_bytes_per_second_by_link_class=(
                (LinkClass.INTRA_NODE, bandwidth),
                (LinkClass.INTER_NODE, bandwidth),
            ),
        )

    fast = build_harness(profile=profile("fast-synthetic", 2, 2_000_000_000))
    slow = build_harness(profile=profile("slow-synthetic", 20, 500_000_000))
    for harness in (fast, slow):
        outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
        assert outcome is SubmitOutcome.PREPARED and receipt is not None
        harness.transport.confirm_commit(receipt)
    fast_record = fast.transport.physical_records()[0]
    slow_record = slow.transport.physical_records()[0]
    assert slow_record.complete_at_ns > fast_record.complete_at_ns
    assert fast.transport.manifest_fragment()["profile_sensitivity_input_supported"] is True
    assert fast.transport.manifest_fragment()["performance_eligible"] is False
    assert slow.transport.manifest_fragment()["profile_provenance"] == "SYNTHETIC_TEST_ONLY"


def test_ep4_four_task_batch_runs_one_launch_with_independent_completion():
    topology = make_network_topology(
        topology_id="ep4-four-task",
        rank_to_node=(0, 1, 2, 3),
        tx_nic_id_by_rank=("r0-tx", "r1-tx", "r2-tx", "r3-tx"),
        rx_nic_id_by_rank=("r0-rx", "r1-rx", "r2-rx", "r3-rx"),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra-0",)),
            (LinkClass.INTER_NODE, ("inter-0", "inter-1", "inter-2", "inter-3")),
        ),
        nic_id_by_lane=(
            ("intra-0", "fabric-intra"),
            ("inter-0", "fabric-inter"),
            ("inter-1", "fabric-inter"),
            ("inter-2", "fabric-inter"),
            ("inter-3", "fabric-inter"),
        ),
    )
    harness = build_harness(
        topology=topology,
        task_specs=(
            ("e0", 0, 1, 8),
            ("e1", 1, 2, 16),
            ("e2", 2, 3, 24),
            ("e3", 3, 0, 32),
        ),
    )
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("e0", "e1", "e2", "e3"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    _run(harness.kernel)
    assert len(harness.completion.started) == 4
    assert len(harness.completion.completed) == 4
    stats = _link_stats(harness.transport.statistics())["INTER_NODE"]
    assert stats["launch_count"] == 1
    assert stats["completed_bytes"] == 80
    harness.transport.assert_terminal()


def test_builder_constructs_directly_from_backend_transport_public_adapters():
    from rs_sim.runtime.adapters.public_ports import (
        CatalogueTaskLookup,
        PhaseAuthorityValidation,
        ReceiverPermitLookup,
        SharedTopologyTaskResolver,
    )

    source = build_harness(task_specs=(("public", 0, 2, 8),))

    class Catalogue:
        def get(self, task_id):
            return source.tasks[task_id]

    class Receiver:
        def receive_permit(self, task_id):
            return source.permits.get(task_id)

    class Authority:
        def stamp_is_current(self, phase_key, authority_stamp):
            return phase_key == source.phase and authority_stamp == source.stamp

    kernel = SimulationKernel()
    bundle = build_formal_transports(
        kernel=kernel,
        task_lookup=CatalogueTaskLookup(Catalogue()),
        permit_lookup=ReceiverPermitLookup(Receiver()),
        authority_validation=PhaseAuthorityValidation(Authority()),
        resource_resolver=SharedTopologyTaskResolver(source.resolver.topology),
        completion_sink=source.completion,
        resource_release_sink=source.release,
        control_delivery_sink=ControlLog(),
        hardware_profile=source.transport.hardware_profile,
    )
    outcome, receipt = bundle.data_plane.prepare_commit(source.batch("public"), bundle.data_plane.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    bundle.data_plane.confirm_commit(receipt)
    _run(kernel)
    bundle.assert_terminal()
    assert bundle.data_plane.completed_task_ids == ("public",)
