from __future__ import annotations

from rs.runtime.observation.instrumentation import BufferedEvidenceSink, build_runtime_instrumentation
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime


def _runtime(*, rank: int = 2) -> RouterSenseInjectionRuntime:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            observation_profile="perf_light",
            selected_layer_ids=("1",),
            schedule_layer_selector="selected",
        ),
        rank=int(rank),
        local_rank=int(rank),
        run_id="release-ledger",
        step_id="step",
        microbatch_id="mb0",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="localhost",
        ep_group_ranks=(0, 1, 2, 3),
        ep_group_root_global_rank=0,
    )
    runtime.runtime_instrumentation = build_runtime_instrumentation(
        instrumentation_mode="perf_light",
        evidence_sink=BufferedEvidenceSink(),
    )
    runtime._instrumentation_mode = "perf_light"
    runtime._commit_sha = "test-sha"
    runtime._git_clean = True
    return runtime


def test_release_ledger_requires_all_p0_payload_roles_before_p1_release() -> None:
    runtime = _runtime(rank=2)
    runtime.begin_forward(forward_epoch=3)
    assert runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1") == ()

    hidden_release = runtime.record_phase_payload_completion(
        layer_id="1",
        phase="P0",
        payload_role="hidden_states",
    )
    assert hidden_release == ()
    assert runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1") == ()

    routing_release = runtime.record_phase_payload_completion(
        layer_id="1",
        phase="P0",
        payload_role="routing_probs",
    )
    assert routing_release == (
        "release:p0_inbound_complete:0",
        "release:p0_inbound_complete:1",
        "release:p0_inbound_complete:2",
        "release:p0_inbound_complete:3",
    )
    assert runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1") == routing_release

    p1_release = runtime.record_phase_payload_completion(
        layer_id="1",
        phase="P1",
        payload_role="hidden_states",
    )
    assert p1_release == (
        "release:p1_inbound_complete:0",
        "release:p1_inbound_complete:1",
        "release:p1_inbound_complete:2",
        "release:p1_inbound_complete:3",
    )
    assert all(item in runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P2") for item in p1_release)


def test_release_ledger_clears_on_new_forward_and_result_bundle_finalizes() -> None:
    runtime = _runtime(rank=1)
    runtime.begin_forward(forward_epoch=1)
    runtime.record_phase_payload_completion(layer_id="1", phase="P0", payload_role="hidden_states")
    runtime.record_phase_payload_completion(layer_id="1", phase="P0", payload_role="routing_probs")
    runtime.record_execution_outcome(
        layer_id="1",
        phase="P0",
        payload_role="hidden_states",
        outcome={
            "success": True,
            "all_work_completed": True,
            "completed_task_ids": ("t0",),
        },
    )
    bundle = runtime._finalize_result_bundle()  # noqa: SLF001
    assert bundle is not None
    assert bundle.status == "success"
    assert bundle.summary["release_id_count"] == 4
    evidence_sink = runtime.runtime_instrumentation.evidence_sink
    assert isinstance(evidence_sink, BufferedEvidenceSink)
    assert evidence_sink.latest_result() is not None

    runtime.begin_forward(forward_epoch=2)
    assert runtime.satisfied_release_dependency_ids_for(layer_id="1", phase="P1") == ()
