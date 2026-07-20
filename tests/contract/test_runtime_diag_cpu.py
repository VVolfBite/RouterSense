from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.observation.runtime_diag import (
    aggregate_control_collectives,
    critical_rank,
    expected_preflight_collective_count,
    hook_attribution,
    measurement_perturbation_audit,
    preflight_contract,
    selected_window_alias,
    task_size_buckets,
)
from rs.scheduling.phase_execution import PhaseExecutionPlan, PlanWave
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def test_preflight_mode_contract_compact_and_full() -> None:
    assert expected_preflight_collective_count("compact") == 2
    assert expected_preflight_collective_count("full") == 9
    assert expected_preflight_collective_count("local_only") == 0
    compact = preflight_contract(
        requested_mode="compact",
        effective_mode="compact",
        executor_mode="compact",
        actual_collective_count=2,
    )
    assert compact["preflight_mode_match"] is True
    assert compact["preflight_collective_count_exact"] is True
    full_mismatch = preflight_contract(
        requested_mode="compact",
        effective_mode="compact",
        executor_mode="full",
        actual_collective_count=9,
    )
    assert full_mismatch["preflight_mode_match"] is False


def test_lifecycle_rejects_plan_effective_preflight_mismatch() -> None:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="prepared_priority",
            planner_id="current:p012:joint:event:rscf",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="none",
            bucket_mode="dynamic_current",
            bucket_rows=0,
            safe_projection_mode="disabled",
            p2_hint_weight=0.0,
            observation_profile="timeline_light",
            invariant_mode="evaluation_strict",
            preflight_mode="compact",
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
    context = make_contexts_from_matrix(phase="P0", matrix=((0, 1), (1, 0)))[0]
    plan = PhaseExecutionPlan(
        plan_key=dict(context.plan_key),
        phase="P0",
        policy_name="test",
        policy_version="v1",
        control_mode="sync_before_phase",
        execution_mode="joint_window_async_p2p",
        transport_mutation=True,
        is_shadow_only=False,
        future_hint_mode="none",
        root_rank=0,
        observation_digest="obs",
        plan_hash="plan",
        waves=(PlanWave(wave_id=0, phase="P0", bucket_tasks=()),),
        metrics={"preflight_mode": "full"},
    )
    with pytest.raises(RuntimeError, match="preflight mode mismatch"):
        runtime._activate_transport(layer_name="model.layers.0.mlp", phase="P0", context=context, plan=plan)  # noqa: SLF001


def test_hook_attribution_rejects_negative_unattributed() -> None:
    payload = hook_attribution(
        hook_total_us=100.0,
        components={"observation_us": 10.0, "plan_us": 20.0, "submit_us": 30.0},
    )
    assert payload["hook_unattributed_us"] == 40.0
    assert payload["hook_unattributed_status"] == "derived"
    with pytest.raises(ValueError, match="negative"):
        hook_attribution(hook_total_us=10.0, components={"plan_us": 20.0})


def test_control_collective_aggregation_schema() -> None:
    summary = aggregate_control_collectives(
        [
            {"category": "preflight_collective", "call_count": 2, "payload_bytes": 128, "submit_us": 3, "wait_us": 1, "total_us": 4},
            {"category": "plan_agreement", "call_count": 1, "payload_bytes": 64, "submit_us": 5, "wait_us": 2, "total_us": 7},
            {"category": "preflight_collective", "call_count": 2, "payload_bytes": 128, "submit_us": 3, "wait_us": 1, "total_us": 4},
        ]
    )
    assert summary["preflight_collective"]["call_count"] == 4
    assert summary["preflight_collective"]["payload_bytes"] == 256
    assert summary["plan_agreement"]["total_us"] == 7.0


def test_rank_critical_component_and_rank3_excess() -> None:
    result = critical_rank(
        [
            {"rank": 0, "hook_unattributed_us": 10},
            {"rank": 1, "hook_unattributed_us": 20},
            {"rank": 2, "hook_unattributed_us": 30},
            {"rank": 3, "hook_unattributed_us": 50},
        ],
        "hook_unattributed_us",
    )
    assert result["critical_rank"] == 3
    assert result["rank3_excess_us"] == 30.0


def test_task_size_buckets_and_selected_window_alias() -> None:
    buckets = task_size_buckets(
        [
            {"byte_count": 0},
            {"byte_count": 1024},
            {"byte_count": 8192},
            {"byte_count": 32768},
            {"byte_count": 131072},
            {"byte_count": 1048576},
        ]
    )
    assert buckets["total_task_count"] == 6
    assert buckets["buckets"]["0 bytes"]["task_count"] == 1
    assert buckets["buckets"][">256 KiB"]["total_bytes"] == 1048576
    alias = selected_window_alias(first_ns=1000, last_ns=31000)
    assert alias["selected_window_span_us"] == 30.0
    assert alias["communication_span_us_deprecated"] is True


def test_measurement_perturbation_audit_distinguishes_induced_sync() -> None:
    audit = measurement_perturbation_audit(
        [
            {"kind": "work.wait", "source": "execution_required"},
            {"kind": "torch.cuda.synchronize", "source": "measurement_induced"},
            {"kind": "tensor.cpu", "source": "measurement_induced"},
            {"kind": "file_write", "source": "measurement_induced"},
        ]
    )
    assert audit["measurement_sync_count"] == 2
    assert audit["measurement_induced_sync_count"] == 1
    assert audit["measurement_host_copy_count"] == 1
    assert audit["measurement_file_write_count_in_timed_path"] == 1
