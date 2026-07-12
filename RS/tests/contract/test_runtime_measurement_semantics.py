from __future__ import annotations

from dataclasses import replace

from experiments.online.support.runtime_presets import resolve_strategy_runtime
from rs.runtime.online.megatron_ep.async_release import runtime_projection as runtime_projection_mod
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.scheduling.bucketizer import (
    BUCKET_MODE_DYNAMIC_CURRENT,
    BUCKET_MODE_FIXED_ROWS,
    CanonicalBucketizer,
    bucket_mode_for_rows,
    summarize_bucket_tasks,
)
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


class _WindowLike:
    def __init__(
        self,
        *,
        p0_truth_rows: tuple[tuple[int, ...], ...],
        p1_truth_rows: tuple[tuple[int, ...], ...] | None = None,
        p2_truth_rows: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.p0_truth_rows = p0_truth_rows
        self.p1_truth_rows = p1_truth_rows or tuple(tuple(0 for _ in row) for row in p0_truth_rows)
        self.p2_truth_rows = p2_truth_rows or tuple(tuple(0 for _ in row) for row in p0_truth_rows)


def _runtime(*, safe_projection_mode: str) -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="none",
            bucket_mode="dynamic_current",
            bucket_rows=0,
            safe_projection_mode=safe_projection_mode,
            p2_hint_weight=0.0,
            observation_profile="execution",
            invariant_mode="evaluation_strict",
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def test_dynamic_bucket_current_uses_per_edge_row_count() -> None:
    window = _WindowLike(
        p0_truth_rows=((0, 1500, 300), (64, 0, 0), (0, 0, 0)),
    )
    tasks = CanonicalBucketizer(bucket_rows=0).bucketize(window)
    summary = summarize_bucket_tasks(tasks)
    assert bucket_mode_for_rows(0) == BUCKET_MODE_DYNAMIC_CURRENT
    per_edge = {
        f"{row['phase']}:{row['src_group_rank']}->{row['dst_group_rank']}": row["bucket_rows"]
        for row in summary["per_edge"]
    }
    assert per_edge["P0:0->1"] == [1500]
    assert per_edge["P0:0->2"] == [300]
    assert per_edge["P0:1->0"] == [64]


def test_fixed_rows_1024_splits_large_edge() -> None:
    window = _WindowLike(
        p0_truth_rows=((0, 2500), (0, 0)),
    )
    tasks = CanonicalBucketizer(bucket_rows=1024).bucketize(window)
    summary = summarize_bucket_tasks(tasks)
    assert bucket_mode_for_rows(1024) == BUCKET_MODE_FIXED_ROWS
    assert summary["per_edge"][0]["bucket_rows"] == [1024, 1024, 452]
    assert summary["per_edge"][0]["row_offsets"] == [0, 1024, 2048]


def test_runtime_presets_distinguish_raw_and_safe_async_joint_paths() -> None:
    predicted_raw = resolve_strategy_runtime(
        strategy_name="routersense_joint_predicted_raw_async",
        runtime_line="async_release",
    )
    predicted_safe = resolve_strategy_runtime(
        strategy_name="routersense_joint_predicted_safe_async",
        runtime_line="async_release",
    )
    legacy_predicted = resolve_strategy_runtime(
        strategy_name="routersense_joint_predicted_async_p2p",
        runtime_line="async_release",
    )
    legacy_safe = resolve_strategy_runtime(
        strategy_name="routersense_safe_joint_async",
        runtime_line="async_release",
    )
    assert predicted_raw["safe_projection_mode"] == "disabled"
    assert predicted_safe["safe_projection_mode"] == "host_select"
    assert legacy_predicted["safe_projection_mode"] == "disabled"
    assert legacy_safe["safe_projection_mode"] == "host_select"


def test_bucket_mode_mismatch_hard_fails() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="none",
            bucket_mode="fixed_rows",
            bucket_rows=0,
            safe_projection_mode="disabled",
            invariant_mode="evaluation_strict",
        ),
        rank=0,
        local_rank=0,
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )
    try:
        runtime._assert_bucket_mode_consistency()  # noqa: SLF001
    except RuntimeError as exc:
        assert "bucket mode mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected bucket mode mismatch to hard fail")


def test_runtime_raw_and_safe_joint_store_different_selected_plan_digests(monkeypatch) -> None:
    matrix = ((0, 5), (3, 0))
    contexts = make_contexts_from_matrix(phase="P0", matrix=matrix, p2_hint_mode="none")

    def _select_paired(*, raw_u_plan, paired_b_plan):
        return {
            "ideal_raw_u_estimated_makespan": 10.0,
            "host_projected_raw_u_estimated_makespan": 10.0,
            "ideal_paired_b_estimated_makespan": 8.0,
            "host_projected_paired_b_estimated_makespan": 8.0,
            "host_projected_safe_selection": str(paired_b_plan.policy_name),
        }

    monkeypatch.setattr(runtime_projection_mod, "host_project_safe_selection", _select_paired)

    raw_runtime = _runtime(safe_projection_mode="disabled")
    raw_observation = raw_runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    raw_runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=contexts[0],
        observation_p0=raw_observation,
        actual_p0_full_row_matrix=matrix,
    )

    safe_runtime = _runtime(safe_projection_mode="host_select")
    safe_observation = safe_runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
    safe_runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
        layer_name="model.layers.0.mlp",
        phase_ctx=contexts[0],
        observation_p0=safe_observation,
        actual_p0_full_row_matrix=matrix,
    )

    raw_plan = raw_runtime._runtime_state.read("global_joint_window_plan")  # noqa: SLF001
    safe_plan = safe_runtime._runtime_state.read("global_joint_window_plan")  # noqa: SLF001
    assert raw_plan["safe_projection_mode"] == "disabled"
    assert safe_plan["safe_projection_mode"] == "host_select"
    assert raw_plan["safe_selected_policy"] != safe_plan["safe_selected_policy"]
    assert raw_runtime._runtime_state.read("stored_p1_logical_plan_digest") != safe_runtime._runtime_state.read("stored_p1_logical_plan_digest")  # noqa: SLF001
