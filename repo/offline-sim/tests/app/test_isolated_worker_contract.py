from rs_sim.app.experiment import _isolated_worker_config


def test_isolated_worker_receives_oracle_budget() -> None:
    config = {
        "name": "test",
        "simulation": {"release_mode": "PHASE_BARRIER"},
        "oracle": {
            "time_limit_ms_per_window": 250,
            "relative_gap": 0.0,
            "require_all_certified": False,
        },
        "save_raw_events": False,
        "save_task_timeline": False,
        "save_plans": False,
    }
    worker = _isolated_worker_config(config)
    assert worker["oracle"] == config["oracle"]
