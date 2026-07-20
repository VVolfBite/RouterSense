from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from rs.experiments_support.gpu_a2_strategy_compare import aggregate_hotpath_rank_counts
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime


@dataclass(frozen=True)
class _DummyContext:
    expert_placement_hash: str = "expert"


def _runtime(*, execution_mode: str = "joint_window_async_p2p", scheduler_mode: str = "disabled") -> RouterSenseInjectionRuntime:
    runtime = RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="" if scheduler_mode in {"native_order", "native_passthrough_identity"} else "prepared_priority",
            planner_id="" if scheduler_mode in {"native_order", "native_passthrough_identity"} else "current:p012:local:event:rscf",
            scheduler_mode=scheduler_mode,
            execution_mode=execution_mode,
            control_mode="sync_before_phase",
            bucket_mode="dynamic_current",
            bucket_rows=0,
            observation_profile="attribution_light",
            invariant_mode="evaluation_strict",
            schedule_layer_selector="selected",
            selected_layer_ids=("0",),
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
    runtime.configure_hook_scope(available_layer_names=("module.decoder.layers.0.mlp",))
    runtime.begin_forward(forward_epoch=0)
    return runtime


def test_async_on_dispatch_and_on_combine_finalize_without_shadow(monkeypatch) -> None:
    runtime = _runtime()
    observation_calls = {"count": 0}
    agreement_calls = {"count": 0}

    def _fake_observation(**_kwargs):
        observation_calls["count"] += 1
        return {"unexpected": True}

    def _fake_agreement(**_kwargs):
        agreement_calls["count"] += 1
        return None

    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle_parts.hooks_combine.build_runtime_observation", _fake_observation)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle_parts.hooks_combine.run_policy_agreement", _fake_agreement)

    runtime.on_dispatch(
        layer_name="module.decoder.layers.0.mlp",
        dispatcher=SimpleNamespace(),
        hidden_states=None,
    )
    runtime.on_combine(
        layer_name="module.decoder.layers.0.mlp",
        dispatcher=SimpleNamespace(),
        hidden_states=None,
    )

    summary = runtime.export_prepared_plan_summary()
    assert observation_calls["count"] == 0
    assert agreement_calls["count"] == 0
    assert summary["shadow_dispatch_execution_count"] == 0
    assert summary["shadow_combine_execution_count"] == 0
    assert summary["shadow_policy_agreement_count"] == 0
    assert summary["shadow_plan_build_count"] == 0
    assert summary["shadow_control_collective_count"] == 0
    assert summary["observation_finalize_dispatch_count"] == 1
    assert summary["observation_finalize_combine_count"] == 1
    assert runtime.perf_counters["hook_on_dispatch_total"]["count"] == 1.0
    assert runtime.perf_counters["hook_on_combine_total"]["count"] == 1.0


def test_legacy_shadow_mode_remains_available(monkeypatch) -> None:
    runtime = _runtime(execution_mode="native_passthrough", scheduler_mode="native_order")
    observation_calls = {"count": 0}
    agreement_calls = {"count": 0}

    def _fake_observation(**kwargs):
        observation_calls["count"] += 1
        return SimpleNamespace(expert_placement_hash="expert") if kwargs["phase"] == "P0" else SimpleNamespace()

    def _fake_policy():
        return "policy"

    def _fake_context(_layer_name: str):
        return _DummyContext()

    def _fake_agreement(**_kwargs):
        agreement_calls["count"] += 1
        plan = SimpleNamespace(
            plan_hash="plan",
            policy_name="native_order",
            execution_mode="shadow_only",
            waves=(1, 2),
            ready_waves=(1,),
            blocked_future_waves=(2,),
        )
        agreement = SimpleNamespace(to_dict=lambda: {"ok": True})
        return plan, agreement

    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle_parts.hooks_combine.build_runtime_observation", _fake_observation)
    monkeypatch.setattr(runtime, "_phase_policy", _fake_policy)
    monkeypatch.setattr(runtime, "_context", _fake_context)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.lifecycle_parts.hooks_combine.run_policy_agreement", _fake_agreement)

    runtime.on_dispatch(
        layer_name="module.decoder.layers.0.mlp",
        dispatcher=SimpleNamespace(),
        hidden_states=None,
    )
    runtime.on_combine(
        layer_name="module.decoder.layers.0.mlp",
        dispatcher=SimpleNamespace(),
        hidden_states=None,
    )

    summary = runtime.export_prepared_plan_summary()
    assert observation_calls["count"] == 2
    assert agreement_calls["count"] == 1
    assert summary["shadow_dispatch_execution_count"] == 1
    assert summary["shadow_combine_execution_count"] == 1
    assert summary["shadow_policy_agreement_count"] == 1
    assert summary["shadow_plan_build_count"] == 1
    assert summary["shadow_control_collective_count"] == 1


def test_hotpath_aggregate_rejects_shadow_counter_for_async() -> None:
    ranks = [
        {
            "rank": rank,
            "selected_p0_hook_count": 4,
            "selected_p1_hook_count": 4,
            "prediction_source_p0_hook_count": 0,
            "none_heavy_hook_count": 0,
            "real_p0_execution_count": 4,
            "real_p1_execution_count": 4,
            "shadow_dispatch_execution_count": 1 if rank == 0 else 0,
            "shadow_combine_execution_count": 0,
            "observation_finalize_dispatch_count": 4,
            "observation_finalize_combine_count": 4,
            "shadow_policy_agreement_count": 0,
            "shadow_plan_build_count": 0,
            "shadow_control_collective_count": 0,
            "joint_build_count": 0,
            "local_build_count": 0,
            "predict_count": 0,
        }
        for rank in range(4)
    ]
    aggregate = aggregate_hotpath_rank_counts(
        ranks,
        expected_world_size=4,
        warmup_iters=1,
        measure_iters=1,
        selected_layer_ids=["0", "1"],
        prediction_source_layer_ids=[],
        strategy="routersense_current_p012_local_event_rscf_async",
    )
    assert aggregate["hotpath_eligible"] is False
    assert any("shadow_dispatch_execution_count:rank=0:expected=0:actual=1" in item for item in aggregate["eligibility_reasons"])
