from __future__ import annotations

from typing import Any, Callable

from . import fast, greedy, oracle
from .strategy import (
    SchedulingContext,
    SchedulingResult,
    SchedulingStrategy,
    register_strategy,
)


def _ctx_args(ctx: SchedulingContext) -> tuple[list[list[int]], list[list[int]], list[list[int]], int]:
    return (
        ctx.dispatch_matrix,
        ctx.combine_matrix,
        ctx.next_dispatch_matrix,
        ctx.num_gpus,
    )


def _ctx_kwargs(ctx: SchedulingContext) -> dict[str, Any]:
    return {
        "model": ctx.model,
        "expert_compute_delay": ctx.expert_compute_delay,
    }


def _from_mapping(name: str, payload: dict[str, Any]) -> SchedulingResult:
    return SchedulingResult(
        makespan=float(payload.get("makespan", 0.0)),
        schedule=list(payload.get("schedule", [])),
        solve_time_ms=float(payload.get("solve_time_ms", 0.0)),
        strategy=str(payload.get("strategy", name)),
    )


def _from_scalar(name: str, makespan: float) -> SchedulingResult:
    return SchedulingResult(
        makespan=float(makespan),
        schedule=[],
        solve_time_ms=0.0,
        strategy=name,
    )


class _FunctionSchedulingStrategy(SchedulingStrategy):
    strategy_name: str
    schedule_fn: Callable[..., Any]
    result_mode: str = "mapping"

    @property
    def name(self) -> str:
        return self.strategy_name

    def solve(self, ctx: SchedulingContext) -> SchedulingResult:
        payload = self.schedule_fn(*_ctx_args(ctx), **_ctx_kwargs(ctx))
        if self.result_mode == "scalar":
            return _from_scalar(self.name, float(payload))
        return _from_mapping(self.name, payload)


def _register_function_strategy(
    class_name: str,
    strategy_name: str,
    schedule_fn: Callable[..., Any],
    *,
    result_mode: str = "mapping",
) -> None:
    strategy_cls = type(
        class_name,
        (_FunctionSchedulingStrategy,),
        {
            "strategy_name": strategy_name,
            "schedule_fn": staticmethod(schedule_fn),
            "result_mode": result_mode,
        },
    )
    register_strategy(strategy_cls)


_register_function_strategy(
    "GreedyStrategy",
    "greedy",
    greedy.greedy_schedule_pairwise,
    result_mode="scalar",
)
_register_function_strategy("BirkhoffStrategy", "birkhoff", fast.fast_schedule_birkhoff)
_register_function_strategy(
    "SimulatedAnnealingStrategy",
    "simulated_annealing",
    fast.fast_schedule_simulated_annealing,
)
_register_function_strategy("GRASPStrategy", "grasp", fast.fast_schedule_grasp)
_register_function_strategy("IBBRStrategy", "ibbr", fast.fast_schedule_ibbr)
_register_function_strategy("TabuSearchStrategy", "tabu_search", fast.fast_schedule_tabu_search)
_register_function_strategy("LNSStrategy", "lns", fast.fast_schedule_lns)
_register_function_strategy("LagrangianStrategy", "lagrangian", fast.fast_schedule_lagrangian)
_register_function_strategy(
    "IteratedGreedyStrategy",
    "iterated_greedy",
    fast.fast_schedule_iterated_greedy,
)
_register_function_strategy(
    "BarrierAwareBirkhoffStrategy",
    "barrier_aware_birkhoff",
    fast.fast_schedule_barrier_aware_birkhoff,
)
_register_function_strategy(
    "RandomizedMultistartBirkhoffStrategy",
    "randomized_multistart_birkhoff",
    fast.fast_schedule_randomized_multistart_birkhoff,
)
_register_function_strategy("TwoStageStrategy", "two_stage", fast.fast_schedule_two_stage)
_register_function_strategy(
    "CriticalPathCompressionStrategy",
    "critical_path_compression",
    fast.fast_schedule_critical_path_compression,
)
_register_function_strategy(
    "CompletionBalancedStrategy",
    "completion_balanced",
    fast.fast_schedule_completion_balanced,
)
_register_function_strategy("CPLPTStrategy", "cp_lpt", fast.fast_schedule_cp_lpt)
_register_function_strategy("CPLocalSwapStrategy", "cp_local_swap", fast.fast_schedule_cp_local_swap)
_register_function_strategy(
    "LookaheadLPTStrategy",
    "lookahead_lpt",
    fast.fast_schedule_lookahead_lpt,
)
_register_function_strategy(
    "PhaseAwareGreedyStrategy",
    "phase_aware_greedy",
    fast.fast_schedule_phase_aware_greedy,
)
_register_function_strategy("BestOfStrategy", "best_of", fast.fast_schedule_pairwise)
_register_function_strategy("OracleStrategy", "oracle", oracle.pairwise_oracle)

