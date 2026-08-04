from __future__ import annotations

from dataclasses import replace

from rs_sim import (
    LinkClass,
    SubmitOutcome,
    TaskResourceFootprint,
    make_task_resource_footprint,
)

from rs_sim.transport import RejectionCode

from .conftest import build_harness, topology_mixed_nodes


def assert_no_side_effects(harness, batch, expected):
    before = harness.transport.snapshot()
    outcome, receipt = harness.transport.prepare_commit(batch, harness.transport.kernel.now_ns)
    assert outcome is expected
    assert receipt is None
    assert harness.transport.snapshot() == before
    assert harness.transport.prepared_count == 0
    assert harness.transport.physical_records() == ()
    assert harness.kernel.pending_event_count() == 0


def test_stale_authority_is_retryable_and_has_no_side_effects(harness):
    harness.transport.authority_validation.current = False
    assert_no_side_effects(
        harness, harness.batch("t0"), SubmitOutcome.RETRYABLE_STALE_AUTHORITY
    )
    assert harness.transport.last_rejection.code is RejectionCode.STALE_AUTHORITY


def test_topology_mismatch_is_fatal_and_has_no_side_effects(harness):
    assert_no_side_effects(
        harness,
        harness.batch("t0", topology_digest="wrong-topology"),
        SubmitOutcome.FATAL_CONTRACT_ERROR,
    )
    assert harness.transport.last_rejection.code is RejectionCode.TOPOLOGY_CONTRACT_MISMATCH


def test_expectation_digest_mismatch_is_fatal_and_has_no_side_effects(harness):
    harness.permits["t0"] = replace(
        harness.permits["t0"], expectation_digest="wrong-expectation"
    )
    assert_no_side_effects(
        harness, harness.batch("t0"), SubmitOutcome.FATAL_CONTRACT_ERROR
    )
    assert harness.transport.last_rejection.code is RejectionCode.PERMIT_CONTRACT_MISMATCH


def test_edge_chunk_offset_and_bytes_are_all_bound(harness):
    mutations = (
        {"chunk_index": 9},
        {"byte_offset": 4},
        {"task_bytes": 9},
        {"edge_key": harness.tasks["t1"].edge_key},
    )
    for values in mutations:
        h = build_harness()
        h.permits["t0"] = replace(h.permits["t0"], **values)
        assert_no_side_effects(h, h.batch("t0"), SubmitOutcome.FATAL_CONTRACT_ERROR)


def test_local_diagonal_submission_is_fatal():
    h = build_harness(task_specs=(("local", 0, 0, 8),))
    assert_no_side_effects(h, h.batch("local"), SubmitOutcome.FATAL_CONTRACT_ERROR)
    assert h.transport.last_rejection.code is RejectionCode.LOCAL_DIAGONAL_TASK


def test_internal_rank_conflict_is_fatal_not_retryable():
    h = build_harness(task_specs=(("a", 0, 2, 8), ("b", 0, 3, 8)))
    assert_no_side_effects(h, h.batch("a", "b"), SubmitOutcome.FATAL_CONTRACT_ERROR)
    assert h.transport.last_rejection.code is RejectionCode.INTERNAL_ENDPOINT_CONFLICT


def test_mixed_link_class_batch_is_fatal():
    topo = topology_mixed_nodes()
    h = build_harness(
        topology=topo,
        task_specs=(("intra", 0, 1, 8), ("inter", 2, 3, 8)),
    )
    assert_no_side_effects(
        h,
        h.batch("intra", "inter", link_class=LinkClass.INTER_NODE),
        SubmitOutcome.FATAL_CONTRACT_ERROR,
    )
    assert h.transport.last_rejection.code is RejectionCode.MIXED_LINK_CLASS


def test_existing_rank_resource_conflict_is_retryable_without_new_side_effects():
    h = build_harness(task_specs=(("a", 0, 2, 8), ("b", 0, 3, 8)))
    outcome, receipt = h.transport.prepare_commit(h.batch("a"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    before = h.transport.snapshot()
    outcome, second = h.transport.prepare_commit(h.batch("b"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.RETRYABLE_RESOURCE_BUSY and second is None
    assert h.transport.snapshot() == before
    assert h.transport.prepared_count == 1


def test_existing_node_nic_conflict_is_retryable():
    topo = topology_mixed_nodes()
    h = build_harness(
        topology=topo,
        task_specs=(("a", 0, 2, 8), ("b", 1, 3, 8)),
    )
    outcome, receipt = h.transport.prepare_commit(h.batch("a"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    before = h.transport.snapshot()
    outcome, second = h.transport.prepare_commit(h.batch("b"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.RETRYABLE_RESOURCE_BUSY and second is None
    assert h.transport.snapshot() == before


def test_existing_lane_conflict_is_retryable_even_with_free_endpoints_and_nics():
    base = build_harness()
    topo = base.resolver.topology
    h = build_harness(task_specs=(("a", 0, 2, 8), ("b", 1, 3, 8)))
    overrides = {}
    for task_id in ("a", "b"):
        fp = make_task_resource_footprint(
            task_id=task_id,
            src_rank=h.tasks[task_id].src_rank,
            dst_rank=h.tasks[task_id].dst_rank,
            topology=topo,
        )
        overrides[task_id] = replace(fp, eligible_lane_ids=("inter-0",))
    h = build_harness(
        topology=topo,
        task_specs=(("a", 0, 2, 8), ("b", 1, 3, 8)),
        resolver_overrides=overrides,
    )
    outcome, receipt = h.transport.prepare_commit(h.batch("a"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    before = h.transport.snapshot()
    outcome, second = h.transport.prepare_commit(h.batch("b"), h.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.RETRYABLE_RESOURCE_BUSY and second is None
    assert h.transport.snapshot() == before


def test_duplicate_task_after_completion_remains_fatal(harness):
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    while harness.kernel.has_events():
        harness.kernel.run_next_timestamp()
    outcome, duplicate = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.FATAL_CONTRACT_ERROR and duplicate is None
    assert harness.transport.last_rejection.code is RejectionCode.DUPLICATE_PHYSICAL_TASK


def test_future_commit_time_is_fatal_and_has_no_side_effects(harness):
    before = harness.transport.snapshot()
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("t0"), harness.transport.kernel.now_ns + 1
    )
    assert outcome is SubmitOutcome.FATAL_CONTRACT_ERROR
    assert receipt is None
    assert harness.transport.last_rejection.code is RejectionCode.INVALID_COMMIT_TIME
    assert harness.transport.snapshot() == before
