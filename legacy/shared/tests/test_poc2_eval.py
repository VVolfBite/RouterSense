from __future__ import annotations

from routesense_poc2.eval import build_sweep_aggregate


def test_build_sweep_aggregate_emits_judgment():
    aggregate = build_sweep_aggregate(
        [
            {
                "model_key": "olmoe",
                "service_backend": "synthetic-matmul",
                "routing_mode": "real_router_logits",
                "service_mode": "synthetic::synthetic-matmul",
                "service_is_synthetic": True,
                "output_dir": "x",
                "summary": {
                    "strategies": {
                        "fifo": {
                            "mean_total_batch_completion_proxy": 1.0,
                            "mean_barrier_join_proxy": 1.0,
                            "mean_per_rank_idle_proxy_sum": 1.0,
                            "mean_critical_bucket_completion_proxy": 1.0,
                            "mean_release_order_delta_vs_fifo": 0.0,
                        },
                        "state-only": {
                            "mean_total_batch_completion_proxy": 0.8,
                            "mean_barrier_join_proxy": 0.8,
                            "mean_per_rank_idle_proxy_sum": 0.8,
                            "mean_critical_bucket_completion_proxy": 0.8,
                            "mean_release_order_delta_vs_fifo": 1.0,
                        },
                        "dependency-only": {
                            "mean_total_batch_completion_proxy": 0.7,
                            "mean_barrier_join_proxy": 0.7,
                            "mean_per_rank_idle_proxy_sum": 0.7,
                            "mean_critical_bucket_completion_proxy": 0.7,
                            "mean_release_order_delta_vs_fifo": 2.0,
                        },
                        "full": {
                            "mean_total_batch_completion_proxy": 0.6,
                            "mean_barrier_join_proxy": 0.6,
                            "mean_per_rank_idle_proxy_sum": 0.6,
                            "mean_critical_bucket_completion_proxy": 0.6,
                            "mean_release_order_delta_vs_fifo": 3.0,
                        },
                    }
                },
            }
        ]
    )
    assert aggregate["mainline_signal_strength"] in {"weak", "moderate", "strong"}
    assert aggregate["best_backend_for_signal"] == "synthetic-matmul"
    assert aggregate["runs"][0]["backend_realism_level"] == "synthetic"
