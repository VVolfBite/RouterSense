from __future__ import annotations

import pytest

from rs.evaluation import (
    MetricBaselineIdentity,
    MetricProvenance,
    OfflineWindowMetrics,
    OptimizationMetrics,
    PerformanceMetricRecord,
    StrategyIdentity,
    improvement_pct,
)


def _record() -> PerformanceMetricRecord:
    return PerformanceMetricRecord(
        strategy=StrategyIdentity(
            planner_id="future:p012:joint:global:rscf",
            timing="future",
            horizon="p012",
            scope="joint",
            engine="global",
            core="rscf",
            predictor="fate",
            prediction_fidelity="faithful_fate",
        ),
        baseline=MetricBaselineIdentity(
            planner_id="current:p01:local:global:rscf",
            comparison_key="same_engine_current_p01_local",
        ),
        provenance=MetricProvenance(
            metric_domain="offline_logical",
            time_unit="logical_time",
            trace_digest="trace-1",
            sample_set_digest="sample-set-1",
            measurement_status="complete",
            source="full_offline_dimension_matrix",
            ep_size=4,
            sample_count=32,
        ),
        metrics=OfflineWindowMetrics(
            communication_makespan=95.0,
            tail_latency_p95=60.0,
            tail_latency_p99=62.0,
            tail_latency_max=63.0,
            first_token_time=20.0,
            planning_ms=1.0,
            bind_ms=0.2,
            target_entry_overhead_ms=0.2,
            total_control_ms=1.2,
            wave_count=10,
            p1_remote_token_count=32,
            current_layer_completion=62.0,
        ),
        optimization=OptimizationMetrics(
            communication_optimization_pct=5.0,
            tail_optimization_pct=4.0,
            first_token_optimization_pct=30.0,
        ),
    )


def test_performance_metric_roundtrip() -> None:
    payload = _record().to_dict()
    assert PerformanceMetricRecord.from_dict(payload).to_dict() == payload
    assert improvement_pct(100.0, 95.0) == 5.0


def test_metric_domain_and_tail_order_are_enforced() -> None:
    payload = _record().to_dict()
    payload["provenance"]["metric_domain"] = "mixed"
    with pytest.raises(ValueError, match="metric_domain"):
        PerformanceMetricRecord.from_dict(payload)
    payload = _record().to_dict()
    payload["metrics"]["tail_latency_p95"] = 70.0
    with pytest.raises(ValueError, match="p95"):
        PerformanceMetricRecord.from_dict(payload)
