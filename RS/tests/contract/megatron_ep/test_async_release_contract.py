from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleaseEvent,
    AsyncReleaseState,
    AsyncReleaseWindowKey,
    apply_event,
    build_shadow_plan_from_matrices,
    decide_next_action,
    ready_task_ids,
    register_shadow_plan,
    validate_async_release_state,
    validate_shadow_plan,
)


def _window_key() -> AsyncReleaseWindowKey:
    return AsyncReleaseWindowKey(
        run_id_digest="run1234",
        layer_id="7",
        ep_group_hash="ep4",
        forward_epoch="0",
        microbatch_id="mb0",
    )


def test_async_release_shadow_plan_builds_expected_task_states() -> None:
    plan = build_shadow_plan_from_matrices(
        window_key=_window_key(),
        policy_name="routersense_async_shadow",
        created_at_event="evt0",
        applies_to_layer_id="8",
        p0_dispatch_matrix=((0, 32), (16, 0)),
        p1_return_matrix=((0, 8), (4, 0)),
        p2_next_dispatch_forecast_matrix=((0, 2), (6, 0)),
        forecast_digest="fd0",
    )
    task_map = {task.task_id: task for task in plan.tasks}
    assert task_map["P0:0->1"].release_state == "ready"
    assert task_map["P1:0->1"].release_state == "blocked"
    assert task_map["P1:0->1"].dependency == "wait_p0_complete"
    assert task_map["P2:0->1"].dependency == "forecast_only"
    assert task_map["P2:0->1"].source == "predicted"
    validation = validate_shadow_plan(plan)
    assert validation["valid"] is True


def test_async_release_ready_task_ids_only_return_ready_p0() -> None:
    state = AsyncReleaseState(window_key=_window_key())
    state = register_shadow_plan(
        state,
        build_shadow_plan_from_matrices(
            window_key=_window_key(),
            policy_name="routersense_async_shadow",
            created_at_event="evt0",
            applies_to_layer_id="8",
            p0_dispatch_matrix=((0, 32), (0, 0)),
            p1_return_matrix=((0, 8), (0, 0)),
            p2_next_dispatch_forecast_matrix=((0, 2), (0, 0)),
            forecast_digest="fd0",
        ),
    )
    assert ready_task_ids(state) == ("P0:0->1",)
    assert decide_next_action(state).action == "release_ready_tasks"


def test_async_release_validation_rejects_p1_complete_before_p0_complete() -> None:
    state = AsyncReleaseState(window_key=_window_key())
    plan = build_shadow_plan_from_matrices(
        window_key=_window_key(),
        policy_name="routersense_async_shadow",
        created_at_event="evt0",
        applies_to_layer_id="8",
        p0_dispatch_matrix=((0, 32), (0, 0)),
        p1_return_matrix=((0, 8), (0, 0)),
        p2_next_dispatch_forecast_matrix=((0, 0), (0, 0)),
        forecast_digest="fd0",
    )
    state = register_shadow_plan(state, plan)
    task = state.tasks_by_id["P1:0->1"]
    state.tasks_by_id["P1:0->1"] = task.__class__(**{**task.to_dict(), "release_state": "completed"})
    result = validate_async_release_state(state)
    assert result["valid"] is False
    assert any("P1 task completed before P0 completion" in error for error in result["errors"])


def test_async_release_validation_rejects_forecast_only_completion() -> None:
    state = AsyncReleaseState(window_key=_window_key())
    plan = build_shadow_plan_from_matrices(
        window_key=_window_key(),
        policy_name="routersense_async_shadow",
        created_at_event="evt0",
        applies_to_layer_id="8",
        p0_dispatch_matrix=((0, 0), (0, 0)),
        p1_return_matrix=((0, 0), (0, 0)),
        p2_next_dispatch_forecast_matrix=((0, 2), (0, 0)),
        forecast_digest="fd0",
    )
    state = register_shadow_plan(state, plan)
    task = state.tasks_by_id["P2:0->1"]
    state.tasks_by_id["P2:0->1"] = task.__class__(**{**task.to_dict(), "release_state": "completed"})
    result = validate_async_release_state(state)
    assert result["valid"] is False
    assert any("forecast_only task cannot complete" in error for error in result["errors"])


def test_async_release_fallback_blocks_further_release_decisions() -> None:
    state = AsyncReleaseState(window_key=_window_key())
    state = register_shadow_plan(
        state,
        build_shadow_plan_from_matrices(
            window_key=_window_key(),
            policy_name="routersense_async_shadow",
            created_at_event="evt0",
            applies_to_layer_id="8",
            p0_dispatch_matrix=((0, 32), (0, 0)),
            p1_return_matrix=((0, 0), (0, 0)),
            p2_next_dispatch_forecast_matrix=((0, 0), (0, 0)),
            forecast_digest="fd0",
        ),
    )
    state = apply_event(
        state,
        AsyncReleaseEvent(
            event_id="evt1",
            window_key=_window_key(),
            rank=0,
            layer_id="7",
            phase="P0",
            event_type="fallback_required",
        ),
    )
    decision = decide_next_action(state)
    assert decision.action == "fallback_phase_sync"
    result = validate_async_release_state(state, decision=decision)
    assert result["valid"] is True
