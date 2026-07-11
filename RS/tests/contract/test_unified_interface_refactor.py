from __future__ import annotations

from dataclasses import is_dataclass

import torch

from rs.runtime.offline.replay_unified import (
    CanonicalBucketizer,
    PlanningHint,
    ReplayWindow,
    build_execution_truth,
    build_multiphase_problem,
    build_planning_problem,
)
from rs.runtime.online.megatron_ep.compiler_facade import CompilationOptions, PlanCompilationRequest, UnifiedScheduleCompiler
from rs.runtime.online.megatron_ep.execution.executor_facade import ExecutionResult
from rs.runtime.online.megatron_ep.pending_window.policy_adapter import compile_prepared_window_phase_plan
from rs.scheduling.catalog import algorithm_specs
from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalSchedulePlan,
    LogicalTopology,
    LogicalWave,
    PreparedWindowPlan,
    ReleaseConstraint,
)
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.registry import resolve_policy
from rs.scheduling.unified_interface import PolicyOptions, build_policy, build_request_from_problem, build_request_from_replay_window
from rs.scheduling.validation import stable_hash
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _window() -> ReplayWindow:
    return ReplayWindow(
        fixture_id="fixture",
        window_id="1->2",
        layer_id=1,
        p0_truth_rows=((0, 3), (2, 0)),
        p1_truth_rows=((0, 2), (3, 0)),
        p2_truth_rows=((0, 4), (1, 0)),
        matrix_unit="rows",
        group_size=2,
        payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
        metadata={},
    )


def _problem():
    window = _window()
    hint = PlanningHint(
        hint_type="copy_current_dispatch",
        p2_hint_rows=window.p0_truth_rows,
        confidence=1.0,
        source_layer=1,
        target_layer=2,
    )
    planning_problem = build_planning_problem(replay_window=window, planning_hint=hint)
    execution_truth = build_execution_truth(window)
    return build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=execution_truth,
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
    )


def _manual_prepared_plan() -> PreparedWindowPlan:
    flow = FlowDemand(
        flow_id="p0_dispatch:1->0",
        phase="p0_dispatch",
        src_rank=1,
        dst_rank=0,
        byte_count=16,
        release_state="ready",
        is_executable=True,
    )
    return PreparedWindowPlan(
        window_key="window-priority",
        forecast_digest="forecast-priority",
        logical_plan=LogicalSchedulePlan(
            policy_name="routersense_multiphase_lookahead:p0_p1_p2",
            waves=(LogicalWave(wave_id=0, flows=(flow,), duration=16.0),),
            diagnostics={"unit": "manual"},
        ),
        created_at_layer_id="0",
        applies_from_layer_id="1",
        execution_capability_required="multiphase_pending_window",
    )


def _joint_prepared_plan() -> PreparedWindowPlan:
    matrix = ((0, 8), (4, 0))
    problem = build_multiphase_problem(
        planning_problem=build_planning_problem(
            replay_window=ReplayWindow(
                fixture_id="joint",
                window_id="0->1",
                layer_id=0,
                p0_truth_rows=matrix,
                p1_truth_rows=((0, 4), (8, 0)),
                p2_truth_rows=((0, 6), (2, 0)),
                matrix_unit="rows",
                group_size=2,
                payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
                metadata={},
            ),
            planning_hint=PlanningHint(
                hint_type="perfect_trace_hint",
                p2_hint_rows=((0, 6), (2, 0)),
                confidence=1.0,
                source_layer=0,
                target_layer=1,
            ),
        ),
        execution_truth=build_execution_truth(
            ReplayWindow(
                fixture_id="joint",
                window_id="0->1",
                layer_id=0,
                p0_truth_rows=matrix,
                p1_truth_rows=((0, 4), (8, 0)),
                p2_truth_rows=((0, 6), (2, 0)),
                matrix_unit="rows",
                group_size=2,
                payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
                metadata={},
            )
        ),
        scheduling_mode="runtime_lookahead",
        expert_compute_delay=0.0,
    )
    return RouterSenseMultiphaseLookaheadPolicy(
        information_mode="p0_p1_p2",
        p0_weight=1.0,
        p1_reservation_weight=1.0,
        p2_hint_weight=1.0,
    ).build_prepared_window_plan(problem=problem, created_at_layer_id="0", applies_from_layer_id="1")


def test_unified_scheduling_request_has_single_hint_and_no_p2_truth() -> None:
    window = _window()
    request = build_request_from_replay_window(
        replay_window=window,
        p2_hint_rows=((0, 0), (0, 0)),
        hint_type="zero_hint",
        confidence=0.0,
        bucket_rows=2,
        policy_options=PolicyOptions(),
    )
    assert request.p2_hint_rows == ((0, 0), (0, 0))
    assert request.p0_truth_rows == window.p0_truth_rows
    assert request.p1_truth_rows == window.p1_truth_rows
    assert not hasattr(request, "p2_truth_rows")
    assert all(task.task_id for task in request.tasks)


def test_build_request_from_problem_uses_rows_and_canonical_tasks() -> None:
    problem = _problem()
    request = build_request_from_problem(
        request_id="offline:test",
        problem=problem,
        bucket_rows=2,
        policy_options=PolicyOptions(),
        hint_type="copy_current_dispatch",
        confidence=1.0,
        layer_id=1,
    )
    replay_tasks = CanonicalBucketizer(bucket_rows=2).bucketize(_window())
    assert request.p0_truth_rows == _window().p0_truth_rows
    assert request.p1_truth_rows == _window().p1_truth_rows
    assert request.p2_hint_rows == _window().p2_truth_rows
    assert CanonicalBucketizer.digest(request.tasks) == CanonicalBucketizer.digest(replay_tasks)


def test_unified_policy_builder_matches_legacy_for_all_canonical_policies() -> None:
    problem = _problem()
    request = build_request_from_problem(
        request_id="equivalence:all",
        problem=problem,
        bucket_rows=2,
        policy_options=PolicyOptions(),
        hint_type="copy_current_dispatch",
        confidence=1.0,
        layer_id=1,
    )
    for spec in algorithm_specs():
        if spec.canonical_id in {"barrier_criticality_posthoc_best", "oracle_local_cp_sat"}:
            continue
        policy = build_policy(spec.canonical_id, request.policy_options)
        unified_plan = policy.plan(request)
        legacy_plan = resolve_policy(policy_name=spec.builder_key, bucket_rows=2).build_logical_plan(problem)
        assert stable_hash([wave.to_dict() for wave in unified_plan.waves]) == stable_hash(
            [wave.to_dict() for wave in legacy_plan.waves]
        ), spec.canonical_id
        assert unified_plan.diagnostics.get("makespan") == legacy_plan.diagnostics.get("makespan"), spec.canonical_id


def test_reference_only_posthoc_best_is_explicitly_rejected() -> None:
    try:
        build_policy("barrier_criticality_posthoc_best", PolicyOptions())
    except ValueError as exc:
        assert "reference-only posthoc selector" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected barrier_criticality_posthoc_best to be rejected explicitly")


def test_reference_only_local_oracle_is_explicitly_rejected() -> None:
    try:
        build_policy("oracle_local_cp_sat", PolicyOptions())
    except ValueError as exc:
        assert "reference-only reporting alias" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected oracle_local_cp_sat to be rejected explicitly")


def test_compiler_bridge_requires_prepared_plan_or_abstract_plan() -> None:
    compiler = UnifiedScheduleCompiler()
    request = PlanCompilationRequest(
        logical_plan=build_policy("fifo_bucket", PolicyOptions()).plan(
            build_request_from_problem(
                request_id="compiler:test",
                problem=_problem(),
                bucket_rows=2,
                policy_options=PolicyOptions(),
                hint_type="copy_current_dispatch",
                confidence=1.0,
                layer_id=1,
            )
        ),
        local_context=None,  # type: ignore[arg-type]
        global_contexts=(),  # type: ignore[arg-type]
        canonical_tasks=(),
        phase="P0",
        tensor_role="hidden_states",
        rank_context={},
        compilation_options=CompilationOptions(bucket_rows=2),
    )
    try:
        compiler.compile(request)
    except ValueError as exc:
        assert "prepared_plan or abstract_phase_execution_plan bridge" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected compiler bridge failure")


def test_execution_result_schema_counts_are_explicit() -> None:
    result = ExecutionResult(
        output_tensor=torch.zeros((2, 4), dtype=torch.float32),
        backend_id="async_release",
        execution_plan_digest="digest",
        send_op_count=1,
        recv_op_count=2,
        local_copy_task_count=3,
        local_copy_row_count=7,
        enqueue_us=1.0,
        wait_us=2.0,
        total_us=3.0,
        fallback_used=False,
        timeout=False,
        raw_summary={},
        execution_entries=(),
    )
    payload = result.to_dict()
    assert is_dataclass(result)
    assert payload["local_copy_task_count"] == 3
    assert payload["local_copy_row_count"] == 7
    assert payload["output_tensor"]["shape"] == (2, 4)


def test_unified_compiler_matches_legacy_phase_local_physical_plan() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 1, 1), (1, 0, 1), (1, 1, 0)), p2_hint_mode="none")
    prepared = _manual_prepared_plan()
    legacy_plan = compile_prepared_window_phase_plan(
        prepared_plan=prepared,
        local_context=contexts[0],
        global_contexts=contexts,
        bucket_rows=0,
        p2_hint_weight=1.0,
    )
    result = UnifiedScheduleCompiler().compile(
        PlanCompilationRequest(
            logical_plan=prepared.logical_plan,
            local_context=contexts[0],
            global_contexts=contexts,
            canonical_tasks=(),
            phase="P0",
            tensor_role="hidden_states",
            rank_context={"global_rank": 0, "local_rank": 0},
            compilation_options=CompilationOptions(bucket_rows=0, p2_hint_weight=1.0),
            prepared_plan=prepared,
            legacy_phase_policy_name="routersense_p0p1p2_hint",
        )
    )
    assert stable_hash([wave.to_dict() for wave in legacy_plan.waves]) == stable_hash(
        [wave.to_dict() for wave in result.execution_plan.waves]
    )
    assert legacy_plan.plan_hash == result.execution_plan.plan_hash
    assert result.audit.metrics["legacy_secondary_policy_invocation_count"] == 1


def test_unified_compiler_matches_legacy_joint_physical_plan_bridge() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 8), (4, 0)), p2_hint_mode="none")
    prepared = _joint_prepared_plan()
    legacy_plan = compile_prepared_window_phase_plan(
        prepared_plan=prepared,
        local_context=contexts[0],
        global_contexts=contexts,
        bucket_rows=0,
        p2_hint_weight=1.0,
    )
    result = UnifiedScheduleCompiler().compile(
        PlanCompilationRequest(
            logical_plan=prepared.logical_plan,
            local_context=contexts[0],
            global_contexts=contexts,
            canonical_tasks=(),
            phase="P0",
            tensor_role="hidden_states",
            rank_context={"global_rank": 0, "local_rank": 0},
            compilation_options=CompilationOptions(bucket_rows=0, p2_hint_weight=1.0),
            prepared_plan=prepared,
            legacy_phase_policy_name="routersense_p0p1p2_hint",
        )
    )
    assert stable_hash([wave.to_dict() for wave in legacy_plan.waves]) == stable_hash(
        [wave.to_dict() for wave in result.execution_plan.waves]
    )
    assert legacy_plan.plan_hash == result.execution_plan.plan_hash
