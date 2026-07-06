from __future__ import annotations

from experiments.online.support.comparison_metrics import (
    add_baseline_deltas,
    build_comparison_report,
    communication_makespan_from_timeline,
)
from rs.runtime.online.megatron_ep.control import plan_agreement as plan_agreement_mod
from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def test_communication_makespan_from_timeline() -> None:
    timeline = [
        {"event": "before_wave", "ts_us": 100},
        {"event": "after_wave", "ts_us": 180},
        {"event": "before_wave", "ts_us": 210},
        {"event": "after_wave", "ts_us": 260},
    ]
    assert communication_makespan_from_timeline(timeline) == 160.0


def test_net_benefit_formula() -> None:
    strategy = {
        "communication_makespan_us": {"mean": 800.0},
        "scheduling_overhead_us": {"mean": 50.0},
        "total_forward_us": {"mean": 1200.0},
    }
    baseline = {
        "communication_makespan_us": {"mean": 1000.0},
        "total_forward_us": {"mean": 1300.0},
    }
    out = add_baseline_deltas(strategy, baseline)
    assert out["net_comm_savings_us"]["mean"] == 200.0
    assert out["net_benefit_us"]["mean"] == 150.0
    assert out["benefit_ratio"]["mean"] == 4.0


def test_comparison_report_structure() -> None:
    report = build_comparison_report(
        run_id="run",
        baseline="disabled",
        strategies=[
            {
                "name": "disabled",
                "description": "baseline",
                "repetitions": 1,
                "metrics": {"communication_makespan_us": {"mean": 100.0}, "total_wave_count": {"mean": 10.0}, "total_forward_us": {"mean": 200.0}},
            },
            {
                "name": "routersense_p0p1p2_hint",
                "description": "candidate",
                "repetitions": 1,
                "metrics": {"communication_makespan_us": {"mean": 80.0}, "total_wave_count": {"mean": 8.0}, "scheduling_overhead_us": {"mean": 5.0}, "total_forward_us": {"mean": 180.0}},
            },
        ],
    )
    assert report["run_id"] == "run"
    assert report["baseline"] == "disabled"
    assert report["strategies"][1]["metrics"]["net_benefit_us"]["mean"] == 15.0
    assert report["pairwise_vs_baseline"]["routersense_p0p1p2_hint"]["comm_makespan_delta_pct"] == -20.0
    assert "routersense_p0p1p2_hint_vs_birkhoff_phase_local" not in report["pairwise_head_to_head"]


def test_plan_agreement_timing_in_metrics(monkeypatch) -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 2), (3, 0)))
    local_context = contexts[0]
    policy = resolve_phase_policy(policy_name="phase_barrier_fifo", bucket_rows=0)

    monkeypatch.setattr(plan_agreement_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(plan_agreement_mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(plan_agreement_mod.dist, "get_process_group_ranks", lambda group=None: [0, 1])

    class _Group:
        WORLD = object()

    monkeypatch.setattr(plan_agreement_mod.dist, "group", _Group)

    def all_gather_object(output, value, group=None):
        if isinstance(value, str):
            output[0] = value
            output[1] = value
            return
        output[0] = contexts[0]
        output[1] = contexts[1]

    def broadcast_object_list(buffer, src=0, group=None):
        return None

    monkeypatch.setattr(plan_agreement_mod.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast_object_list", broadcast_object_list)

    plan = plan_agreement_mod.run_phase_plan_agreement(local_context=local_context, policy=policy, group=None)
    for key in (
        "all_gather_time_us",
        "build_plan_time_us",
        "broadcast_time_us",
        "verify_time_us",
        "total_agreement_time_us",
    ):
        assert key in plan.metrics
        assert float(plan.metrics[key]) >= 0.0
