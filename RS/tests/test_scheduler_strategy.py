from __future__ import annotations

from routesense.runtime.distributed_ep.core.scheduler import Scheduler
from routesense.scheduler import SchedulingContext, get_strategy, list_strategies


def test_strategy_registry_contains_expected_algorithms() -> None:
    names = list_strategies()
    assert "greedy" in names
    assert "birkhoff" in names
    assert "oracle" in names
    assert "best_of" in names
    assert len(names) >= 20


def test_strategy_instance_solves_simple_case() -> None:
    strategy = get_strategy("birkhoff")
    ctx = SchedulingContext(
        dispatch_matrix=[[0, 1], [1, 0]],
        combine_matrix=[[0, 1], [1, 0]],
        next_dispatch_matrix=[[0, 1], [1, 0]],
        num_gpus=2,
    )
    result = strategy.solve(ctx)
    assert result.strategy == "birkhoff"
    assert result.makespan >= 0.0


def test_runtime_scheduler_can_switch_strategies() -> None:
    scheduler = Scheduler()
    assert scheduler.strategy.name == "greedy"
    scheduler.set_strategy("oracle")
    assert scheduler.strategy.name == "oracle"
