from __future__ import annotations

import inspect
from typing import Any, Callable

from . import fast, greedy, oracle
from .multiphase_global import (
    fast_schedule_u_barrier_criticality_global_matching,
    fast_schedule_u_barrier_price_adaptive_matching,
    fast_schedule_u_gated_greedy_maximal,
    fast_schedule_u_gated_maxweight_matching,
)
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
    strategy_prediction_aware: bool = False
    strategy_description: str = ""

    @property
    def name(self) -> str:
        return self.strategy_name

    @property
    def prediction_aware(self) -> bool:
        return self.strategy_prediction_aware

    @property
    def description(self) -> str:
        return self.strategy_description

    def solve(self, ctx: SchedulingContext) -> SchedulingResult:
        kwargs = dict(_ctx_kwargs(ctx))
        signature = inspect.signature(self.schedule_fn)
        if "mode" in signature.parameters:
            kwargs["mode"] = ctx.mode
        if "prediction_confidence" in signature.parameters:
            kwargs["prediction_confidence"] = ctx.prediction_confidence
        payload = self.schedule_fn(*_ctx_args(ctx), **kwargs)
        if self.result_mode == "scalar":
            return _from_scalar(self.name, float(payload))
        return _from_mapping(self.name, payload)


def _register_function_strategy(
    class_name: str,
    strategy_name: str,
    schedule_fn: Callable[..., Any],
    *,
    result_mode: str = "mapping",
    prediction_aware: bool = False,
    description: str = "",
) -> None:
    strategy_cls = type(
        class_name,
        (_FunctionSchedulingStrategy,),
        {
            "strategy_name": strategy_name,
            "schedule_fn": staticmethod(schedule_fn),
            "result_mode": result_mode,
            "strategy_prediction_aware": prediction_aware,
            "strategy_description": description,
        },
    )
    register_strategy(strategy_cls)


_register_function_strategy(
    "GreedyStrategy",
    "greedy",
    greedy.greedy_schedule_pairwise,
    result_mode="scalar",
    description="Greedy pairwise baseline.",
)
_register_function_strategy(
    "BirkhoffStrategy",
    "birkhoff",
    fast.fast_schedule_birkhoff,
    prediction_aware=False,
    description="Per-phase Birkhoff decomposition; next phase does not influence earlier phases.",
)
_register_function_strategy(
    "DBirkhoffStrategy",
    "D_birkhoff",
    fast.fast_schedule_birkhoff,
    prediction_aware=False,
    description="D baseline: per-phase Birkhoff decomposition.",
)
_register_function_strategy(
    "BirkhoffExhaustiveStrategy",
    "birkhoff_exhaustive",
    fast.fast_schedule_birkhoff_exhaustive,
    prediction_aware=False,
    description="Validation-only exhaustive Birkhoff round permutation search.",
)
_register_function_strategy(
    "SimulatedAnnealingStrategy",
    "simulated_annealing",
    fast.fast_schedule_simulated_annealing,
    prediction_aware=False,
    description="Local search over phase-local order swaps.",
)
_register_function_strategy("GRASPStrategy", "grasp", fast.fast_schedule_grasp, prediction_aware=False, description="Greedy randomized adaptive search.")
_register_function_strategy("IBBRStrategy", "ibbr", fast.fast_schedule_ibbr, prediction_aware=False, description="Iterated Birkhoff barrier repair.")
_register_function_strategy("TabuSearchStrategy", "tabu_search", fast.fast_schedule_tabu_search, prediction_aware=False, description="Tabu search on same-phase swaps.")
_register_function_strategy("LNSStrategy", "lns", fast.fast_schedule_lns, prediction_aware=False, description="Large-neighborhood repair around bottleneck GPUs.")
_register_function_strategy(
    "LNSCPRepairStrategy",
    "lns_cp_repair",
    fast.fast_schedule_lns_cp_repair,
    prediction_aware=False,
    description="CP-repair around single-phase bottlenecks.",
)
_register_function_strategy(
    "LagrangianStrategy",
    "lagrangian",
    fast.fast_schedule_lagrangian,
    prediction_aware=True,
    description="Cross-phase Lagrangian ordering using all three phase matrices.",
)
_register_function_strategy(
    "EjectionChainTabuStrategy",
    "ejection_chain_tabu",
    fast.fast_schedule_ejection_chain_tabu,
    prediction_aware=False,
    description="Cross-phase tabu search with shallow ejection chains.",
)
_register_function_strategy(
    "IteratedGreedyStrategy",
    "iterated_greedy",
    fast.fast_schedule_iterated_greedy,
    prediction_aware=False,
    description="Destroy-repair local search over existing phase orders.",
)
_register_function_strategy(
    "BarrierAwareBirkhoffStrategy",
    "barrier_aware_birkhoff",
    fast.fast_schedule_barrier_aware_birkhoff,
    prediction_aware=False,
    description="Searches Birkhoff variants but does not use phase-2 values to guide earlier phases.",
)
_register_function_strategy(
    "DBarrierAwareBirkhoffStrategy",
    "D_barrier_aware_birkhoff",
    fast.fast_schedule_barrier_aware_birkhoff,
    prediction_aware=False,
    description="D baseline: Birkhoff variant search within phase-local space.",
)
_register_function_strategy(
    "RandomizedMultistartBirkhoffStrategy",
    "randomized_multistart_birkhoff",
    fast.fast_schedule_randomized_multistart_birkhoff,
    prediction_aware=False,
    description="Randomized Birkhoff variant search.",
)
_register_function_strategy("TwoStageStrategy", "two_stage", fast.fast_schedule_two_stage, prediction_aware=False, description="Two-stage Birkhoff-based ordering.")
_register_function_strategy("DTwoStageStrategy", "D_two_stage", fast.fast_schedule_two_stage, prediction_aware=False, description="D baseline: two-stage Birkhoff-based ordering.")
_register_function_strategy(
    "DecomposedStrategy",
    "decomposed",
    fast.fast_schedule_decomposed,
    prediction_aware=False,
    description="Grouped subproblem decomposition.",
)
_register_function_strategy(
    "QuantizedDecomposedStrategy",
    "quantized_decomposed",
    fast.fast_schedule_quantized_decomposed,
    prediction_aware=False,
    description="Quantized grouped decomposition with deferred reinsert.",
)
_register_function_strategy(
    "CriticalPathCompressionStrategy",
    "critical_path_compression",
    fast.fast_schedule_critical_path_compression,
    prediction_aware=False,
    description="Critical-path swap heuristic.",
)
_register_function_strategy(
    "CompletionBalancedStrategy",
    "completion_balanced",
    fast.fast_schedule_completion_balanced,
    prediction_aware=False,
    description="Slack-balanced per-phase ordering.",
)
_register_function_strategy(
    "CPLPTStrategy",
    "cp_lpt",
    fast.fast_schedule_cp_lpt,
    prediction_aware=True,
    description="Critical-path LPT using later_work from next phase.",
)
_register_function_strategy(
    "DCPLPTStrategy",
    "D_cp_lpt",
    fast.fast_schedule_cp_lpt,
    prediction_aware=True,
    description="D baseline: critical-path LPT using next-phase later_work.",
)
_register_function_strategy("CPLocalSwapStrategy", "cp_local_swap", fast.fast_schedule_cp_local_swap, prediction_aware=False, description="Local swap refinement over CP-LPT seed.")
_register_function_strategy(
    "LookaheadLPTStrategy",
    "lookahead_lpt",
    fast.fast_schedule_lookahead_lpt,
    prediction_aware=True,
    description="Lookahead LPT using next-dispatch marginal loads.",
)
_register_function_strategy(
    "PhaseAwareGreedyStrategy",
    "phase_aware_greedy",
    fast.fast_schedule_phase_aware_greedy,
    prediction_aware=False,
    description="Greedy with downstream phase-size proxy.",
)
_register_function_strategy("BestOfStrategy", "best_of", fast.fast_schedule_pairwise, description="Best-of candidate fast scheduler pool.")
_register_function_strategy("OracleStrategy", "oracle", oracle.pairwise_oracle, prediction_aware=True, description="CP-SAT upper bound over all provided phases.")
_register_function_strategy(
    "DIBBRStrategy",
    "D_ibbr",
    fast.fast_schedule_ibbr,
    prediction_aware=False,
    description="D baseline: iterated Birkhoff barrier repair.",
)
_register_function_strategy(
    "DLagrangianStrategy",
    "D_lagrangian",
    fast.fast_schedule_lagrangian,
    prediction_aware=True,
    description="D baseline: existing lagrangian-style cross-phase ordering.",
)
_register_function_strategy(
    "UBarrierCriticalityGlobalMatchingStrategy",
    "U_barrier_criticality_global_matching",
    fast_schedule_u_barrier_criticality_global_matching,
    prediction_aware=True,
    description="U scheduler: global ready-set max-weight matching with barrier criticality.",
)
_register_function_strategy(
    "UBarrierPriceAdaptiveMatchingStrategy",
    "U_barrier_price_adaptive_matching",
    fast_schedule_u_barrier_price_adaptive_matching,
    prediction_aware=True,
    description="U scheduler: global ready-set matching with bounded adaptive barrier prices.",
)
_register_function_strategy(
    "UGatedMaxWeightMatchingStrategy",
    "U_gated_maxweight_matching",
    fast_schedule_u_gated_maxweight_matching,
    prediction_aware=True,
    description="U scheduler: exact global max-weight matching over currently released edges.",
)
_register_function_strategy(
    "UGatedGreedyMaximalStrategy",
    "U_gated_greedy_maximal",
    fast_schedule_u_gated_greedy_maximal,
    prediction_aware=True,
    description="U baseline: greedy maximal matching over the same ready-set scores.",
)
