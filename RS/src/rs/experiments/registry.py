from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts.result import EligibilityResult, ResultBundle, RunIdentity
from rs.experiments.specs import PlanningCase, RunKind, RunPlan


class Runner(Protocol):
    def run(self, plan: RunPlan) -> ResultBundle:
        ...


def _base_result(plan: RunPlan, *, pipeline: str) -> ResultBundle:
    return ResultBundle(
        run_identity=RunIdentity(
            run_id=f"{plan.suite_id}:{plan.case_id}",
            pipeline=pipeline,
            claim_scope="formal",
            trace_origin="planned",
            future_information_mode=plan.planning_case.prediction_mode,
        ),
        status="invalid",
        eligibility=EligibilityResult(
            correctness_eligible=False,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            reasons=("runner_not_wired",),
        ),
        summary={"all_work_completed": False},
        details={"run_kind": plan.run_kind.value, "planner_id": plan.planning_case.planner_id},
    )


def _diagnostic_success_result(plan: RunPlan) -> ResultBundle:
    return ResultBundle(
        run_identity=RunIdentity(
            run_id=f"{plan.suite_id}:{plan.case_id}",
            pipeline="online",
            claim_scope="formal",
            trace_origin="planned",
            future_information_mode=plan.planning_case.prediction_mode,
        ),
        status="success",
        eligibility=EligibilityResult(
            correctness_eligible=True,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            reasons=("diagnostic_mode",),
        ),
        summary={
            "all_work_completed": True,
            "runner_kind": "diagnostic",
        },
        details={
            "run_kind": plan.run_kind.value,
            "planner_id": plan.planning_case.planner_id,
            "planner_family": plan.planning_case.planner_family,
            "execution_backend": plan.planning_case.execution_backend,
            "instrumentation_mode": plan.planning_case.instrumentation_mode,
        },
    )


@dataclass
class OfflineEvaluationRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="offline")


@dataclass
class GlooFunctionalRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online")


@dataclass
class GPUCorrectnessRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online")


@dataclass
class GPUPerformanceRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online")


@dataclass
class MultinodeRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online")


@dataclass
class TraceCollectionRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online")


@dataclass
class DiagnosticRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _diagnostic_success_result(plan)


class RunnerRegistry:
    def __init__(self) -> None:
        self._runners: dict[RunKind, Runner] = {
            RunKind.OFFLINE_EVALUATION: OfflineEvaluationRunner(),
            RunKind.GLOO_FUNCTIONAL: GlooFunctionalRunner(),
            RunKind.GPU_CORRECTNESS: GPUCorrectnessRunner(),
            RunKind.GPU_PERFORMANCE: GPUPerformanceRunner(),
            RunKind.MULTINODE_CORRECTNESS: MultinodeRunner(),
            RunKind.MULTINODE_PERFORMANCE: MultinodeRunner(),
            RunKind.TRACE_COLLECTION: TraceCollectionRunner(),
            RunKind.DIAGNOSTIC: DiagnosticRunner(),
        }

    def resolve(self, run_kind: RunKind) -> Runner:
        return self._runners[run_kind]

    def list_run_kinds(self) -> tuple[str, ...]:
        return tuple(item.value for item in self._runners)
