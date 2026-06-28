from __future__ import annotations

from routesense_poc2.runner import RunnerConfig, run_single_node_experiment


def test_mock_single_node_runner_writes_all_strategies(tmp_path):
    config = RunnerConfig(
        output_dir=str(tmp_path),
        microbatch_size=2,
        microbatch_count=2,
        mock_routing=True,
        service_backend="mock-sleep",
        strategies=["fifo", "state-only", "dependency-only", "full"],
    )
    payload = run_single_node_experiment(config)
    assert (tmp_path / "summary.json").exists()
    assert payload["summary"]["strategies"]["fifo"]["microbatch_count"] == 2
    assert set(payload["summary"]["strategies"]) == {"fifo", "state-only", "dependency-only", "full"}
    assert payload["service_backend"] == "mock-sleep"
    assert payload["service_is_synthetic"] is True
