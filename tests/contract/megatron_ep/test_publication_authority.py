from __future__ import annotations

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.online.megatron_ep.target_planning import TargetPlanKey, TargetPlanStore
from rs.runtime.online.megatron_ep.target_planning.contracts import PreparationToken


def test_publication_authority_rejects_older_expected_token_after_newer_registration() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey(run_id="run", forward_epoch=3, microbatch_id="mb", target_layer_id="1")
    newer = PreparationToken(
        service_session_id=2,
        forward_generation=3,
        target_key=key,
        task_version=9,
        publish_sequence=20,
    )
    older = PreparationToken(
        service_session_id=1,
        forward_generation=3,
        target_key=key,
        task_version=1,
        publish_sequence=1,
    )
    assert store.register_expected_publication(newer) is True
    assert store.register_expected_publication(older) is False


def test_request_digest_changes_with_planning_track_and_p2_semantics() -> None:
    base_kwargs = dict(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        p0_dispatch_rows=((0, 2), (1, 0)),
        p1_return_rows=((0, 1), (2, 0)),
        p2_hint_rows=((0, 3), (4, 0)),
        predictor_id="copy_current_dispatch",
        confidence=1.0,
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=4, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
        hint_type="copy_current_dispatch",
    )
    lookahead = build_window_planning_request(
        **base_kwargs,
        planning_track="runtime_lookahead",
        p2_semantics="advisory_hint",
    )
    execution = build_window_planning_request(
        **base_kwargs,
        planning_track="execution_window",
        p2_semantics="executable_actual",
    )
    assert lookahead.semantic_digest() != execution.semantic_digest()
