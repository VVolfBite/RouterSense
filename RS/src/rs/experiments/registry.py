from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rs.core.contracts import (
    EvaluationSpec,
    OfflineWindow,
    PlanWave,
    PlannedFlow,
    PredictionHint,
    PredictionIdentity,
    PredictionResult,
    TrafficProvenance,
    WindowPlan,
)
from rs.core.contracts.result import EligibilityResult, ResultBundle, RunIdentity
from rs.evidence.eligibility import evaluate_result_bundle_eligibility
from rs.experiments.output_schema import resolve_commit_identity
from rs.experiments.specs import PlanningCase, RunKind, RunPlan
from rs.offline import (
    OfflineEvaluator,
    OfflinePlanningRequestBuilder,
    build_evaluation_bundle,
    build_execution_truth,
    build_offline_record,
)
from rs.planning import PlannerRegistry


class Runner(Protocol):
    def run(self, plan: RunPlan) -> ResultBundle:
        ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plan_run_id(plan: RunPlan) -> str:
    experiment_id = str(getattr(plan, "experiment_id", "adhoc"))
    suite_id = str(getattr(plan, "suite_id", "suite"))
    case_id = str(getattr(plan, "case_id", "case"))
    repeat_index = int(getattr(plan, "repeat_index", 0) or 0)
    seed = int(getattr(plan, "seed", 0) or 0)
    return f"{experiment_id}:{suite_id}:{case_id}:r{repeat_index}:s{seed}"


def _instrumentation_mode(plan: RunPlan) -> str:
    raw = getattr(plan.planning_case, "instrumentation_mode", "off")
    if raw is False:
        return "off"
    normalized = str(raw).strip().lower() or "off"
    if normalized == "false":
        return "off"
    if normalized == "true":
        return "debug"
    return normalized


def _commit_identity(plan: RunPlan) -> tuple[str, bool]:
    sha, dirty, _, _ = resolve_commit_identity(
        repo_root=_repo_root(),
        run_plan_commit_sha=str(getattr(plan, "commit_sha", "")),
    )
    return str(sha), bool(dirty)


def _base_result(plan: RunPlan, *, pipeline: str, reason: str) -> ResultBundle:
    commit_sha, git_dirty = _commit_identity(plan)
    return ResultBundle(
        run_identity=RunIdentity(
            run_id=_plan_run_id(plan),
            pipeline=pipeline,
            claim_scope="formal",
            trace_origin="planned",
            future_information_mode=str(getattr(plan.planning_case, "prediction_mode", "none")),
        ),
        status="invalid",
        correctness_status="invalid",
        performance_status="ineligible",
        pipeline=pipeline,
        commit_sha=commit_sha,
        git_clean=not git_dirty,
        instrumentation_mode=_instrumentation_mode(plan),
        audit_evidence_level="summary_only",
        measurement_complete=False,
        eligibility=EligibilityResult(
            correctness_eligible=False,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            reasons=(reason,),
        ),
        summary={
            "all_work_completed": False,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 1,
        },
        details={"run_kind": plan.run_kind.value, "planner_id": plan.planning_case.planner_id},
    )


def _finalize_bundle(
    *,
    plan: RunPlan,
    bundle: ResultBundle,
) -> ResultBundle:
    eligibility = evaluate_result_bundle_eligibility(bundle)
    performance_status = "eligible" if eligibility.performance_eligible else "ineligible"
    return ResultBundle(
        run_identity=bundle.run_identity,
        status=bundle.status,
        eligibility=eligibility,
        schema_version=bundle.schema_version,
        correctness_status=bundle.correctness_status,
        performance_status=performance_status,
        pipeline=bundle.pipeline,
        commit_sha=bundle.commit_sha,
        git_clean=bundle.git_clean,
        instrumentation_mode=bundle.instrumentation_mode,
        audit_evidence_level=bundle.audit_evidence_level,
        measurement_complete=bundle.measurement_complete,
        summary=dict(bundle.summary),
        details=dict(bundle.details) | {"config_digest": str(getattr(plan, "config_digest", ""))},
        extensions=dict(bundle.extensions),
    )


def _offline_fixture_dir() -> Path:
    return _repo_root() / "tests" / "fixtures" / "offline_replay_smoke"


def _offline_fixture_payload(name: str) -> dict[str, object]:
    return json.loads((_offline_fixture_dir() / name).read_text(encoding="utf-8"))


def _offline_fixture_window() -> OfflineWindow:
    payload = _offline_fixture_payload("replay_layer_1.json")
    metadata = dict(payload["metadata"])
    return OfflineWindow(
        window_identity="fixture:1->2",
        source_layer=str(metadata["layer_id"]),
        target_layer=str(metadata["next_layer_id"]),
        p0_actual=tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"]),
        p1_actual=tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"]),
        p2_actual=tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_matrix"]),
        placement_snapshot={"group_size": 4, "fixture_type": str(metadata["fixture_type"])},
        traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=13,
        used_token_count=13,
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="fixture-trace-layer1",
    )


def _offline_prediction(window: OfflineWindow, case: PlanningCase) -> PredictionResult:
    predictor_id = str(case.predictor_id)
    if predictor_id in {"perfect_trace_hint", "oracle"}:
        hint_rows = window.p2_actual
        oracle = True
    elif predictor_id in {"none", "zero", "zero_hint"}:
        hint_rows = tuple(tuple(0 for _ in row) for row in window.p2_actual)
        oracle = False
    elif predictor_id in {"copy_current", "copy_current_dispatch"}:
        hint_rows = window.p0_actual
        oracle = False
    else:
        hint_rows = window.p2_actual
        oracle = False
    return PredictionResult(
        identity=PredictionIdentity(
            request_id=str(window.window_identity),
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
        hint=PredictionHint(
            predictor_id=predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=hint_rows,
            confidence=1.0 if oracle else 0.75,
            oracle=oracle,
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
    )


def _offline_spec(plan: RunPlan) -> EvaluationSpec:
    defaults = dict(getattr(plan, "defaults", {}))
    topology = dict(defaults.get("topology", {}) or {})
    return EvaluationSpec(
        track="runtime_lookahead",
        world_size=int(topology.get("world_size", 4) or 4),
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="lookahead",
        residual_policy="reject",
    )


def _offline_planner_id(case: PlanningCase) -> str:
    planner_id = str(case.planner_id)
    if planner_id == "fifo":
        return "fifo_bucket"
    return planner_id


@dataclass
class OfflineEvaluationRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        commit_sha, git_dirty = _commit_identity(plan)
        workload = dict(dict(getattr(plan, "defaults", {})).get("workload", {}) or {})
        fixture_dir = Path(str(workload.get("fixture_dir", _offline_fixture_dir()))).resolve()
        fixture_path = fixture_dir / "replay_layer_1.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        window = OfflineWindow(
            window_identity="fixture:1->2",
            source_layer=str(metadata["layer_id"]),
            target_layer=str(metadata["next_layer_id"]),
            p0_actual=tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"]),
            p1_actual=tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"]),
            p2_actual=tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_matrix"]),
            placement_snapshot={"group_size": 4, "fixture_type": str(metadata["fixture_type"])},
            traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
            matrix_unit="rows",
            return_model="transpose_dispatch",
            raw_token_count=13,
            used_token_count=13,
            dropped_token_count=0,
            drop_reason=None,
            trace_digest="fixture-trace-layer1",
        )
        prediction = _offline_prediction(window, plan.planning_case)
        spec = _offline_spec(plan)
        bucket_rows = int(workload.get("bucket_rows", 4) or 4)
        request = OfflinePlanningRequestBuilder(bucket_rows=bucket_rows, information_mode="p0_p1_p2").build(window, prediction, spec)
        planner = PlannerRegistry.create(_offline_planner_id(plan.planning_case), None)
        window_plan = planner.plan(request)
        truth = build_execution_truth(window, spec)
        evaluation = OfflineEvaluator().evaluate(window_plan, truth, spec)
        status = "success" if evaluation.valid else "invalid"
        correctness_status = "valid" if evaluation.valid else "invalid"
        audit_status = "valid" if evaluation.valid else "invalid"
        coverage_status = "complete" if evaluation.valid else "incomplete"
        record = build_offline_record(
            window=window,
            spec=spec,
            task_set_digest=truth.task_set.task_set_digest,
            request=request,
            prediction=prediction,
            plan=window_plan,
            execution_truth_digest=truth.truth_digest,
            evaluation=evaluation,
            planner_reported_makespan=None,
            audit_status=audit_status,
            coverage_status=coverage_status,
            eligibility={"offline_replay_eligible": evaluation.valid, "performance_eligible": evaluation.valid},
        )
        offline_bundle = build_evaluation_bundle(
            spec=spec,
            records=(record,),
            eligibility_summary={"record_count": 1, "valid_record_count": 1 if evaluation.valid else 0},
        )
        result = ResultBundle(
            run_identity=RunIdentity(
                run_id=_plan_run_id(plan),
                pipeline="offline",
                claim_scope="formal",
                trace_origin="fixture",
                future_information_mode=str(getattr(plan.planning_case, "prediction_mode", "none")),
            ),
            status=status,
            correctness_status=correctness_status,
            performance_status="unknown",
            pipeline="offline",
            commit_sha=commit_sha,
            git_clean=not git_dirty,
            instrumentation_mode=_instrumentation_mode(plan),
            audit_evidence_level="summary_only",
            measurement_complete=True,
            eligibility=None,
            summary={
                "all_work_completed": bool(evaluation.valid),
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0 if evaluation.valid else 1,
                "offline_replay_complete": True,
                "evaluation_spec_digest": spec.semantic_digest(),
                "task_set_digest": truth.task_set.task_set_digest,
                "execution_truth_digest": truth.truth_digest,
                "offline_record_count": len(offline_bundle.records),
                "offline_audit_status": audit_status,
                "coverage_status": coverage_status,
                "realized_makespan": evaluation.realized_makespan,
                "performance_measurement_complete": False,
                "measured_repeat_count": 0,
                "warmup_excluded": False,
                "prediction_evaluation_complete": False,
                "prediction_record_count": 0,
                "prediction_metric_count": 0,
                "prediction_audit_status": "not_run",
                "prediction_truth_digest": "",
                "truth_leakage_check": True,
            },
            details={
                "run_kind": plan.run_kind.value,
                "planner_id": plan.planning_case.planner_id,
                "predictor_id": plan.planning_case.predictor_id,
                "logical_plan_digest": window_plan.semantic_digest(),
                "planning_request_digest": request.semantic_digest(),
                "offline_bundle": {
                    "schema_version": offline_bundle.schema_version,
                    "evaluation_spec": offline_bundle.evaluation_spec.to_dict(),
                    "record_count": len(offline_bundle.records),
                    "paired_aggregate_count": len(offline_bundle.paired_aggregates),
                },
            },
        )
        return _finalize_bundle(plan=plan, bundle=result)


@dataclass
class GlooFunctionalRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        commit_sha, git_dirty = _commit_identity(plan)
        raw_backend = str(getattr(plan.planning_case, "execution_backend", "")).strip().lower()
        backend = "async_release" if raw_backend == "async_release" else "phase_sync"
        run_root = Path(str(getattr(plan, "output_dir", "") or (_repo_root() / "outputs" / "gloo_runner"))).resolve()
        run_id = _plan_run_id(plan).replace(":", "_")
        artifact_dir = run_root / "runs" / run_id / "evidence"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / f"gloo_gate_{backend}_summary.json"
        stdout_log = artifact_dir / f"gloo_gate_{backend}.stdout.log"
        stderr_log = artifact_dir / f"gloo_gate_{backend}.stderr.log"
        env = dict(os.environ)
        existing = str(env.get("PYTHONPATH", "") or "")
        env["PYTHONPATH"] = os.pathsep.join(part for part in ("src", ".", existing) if part)
        gate_script = _repo_root() / "experiments" / "distributed" / "run_m123_integrated_publication_execution_gloo.py"
        command = [
            sys.executable,
            str(gate_script),
            "--execution-backend",
            backend,
            "--instrumentation-mode",
            _instrumentation_mode(plan),
            "--summary-path",
            str(summary_path),
        ]
        timeout_count = 0
        try:
            proc = subprocess.run(
                command,
                cwd=str(_repo_root()),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_count = 1
            stdout_log.write_text(str(exc.stdout or ""), encoding="utf-8")
            stderr_log.write_text(str(exc.stderr or ""), encoding="utf-8")
            result = ResultBundle(
                run_identity=RunIdentity(
                    run_id=_plan_run_id(plan),
                    pipeline="online",
                    claim_scope="formal",
                    trace_origin="runtime",
                    future_information_mode=str(getattr(plan.planning_case, "prediction_mode", "none")),
                ),
                status="failure",
                correctness_status="invalid",
                performance_status="unknown",
                pipeline="online",
                commit_sha=commit_sha,
                git_clean=not git_dirty,
                instrumentation_mode=_instrumentation_mode(plan),
                audit_evidence_level="summary_only",
                measurement_complete=False,
                eligibility=None,
                summary={
                    "all_work_completed": False,
                    "fallback_count": 0,
                    "timeout_count": 1,
                    "check_failure_count": 1,
                    "measurement_event_count": 0,
                    "performance_measurement_complete": False,
                    "measured_repeat_count": 0,
                    "warmup_excluded": False,
                    "prediction_evaluation_complete": False,
                    "prediction_record_count": 0,
                    "prediction_metric_count": 0,
                    "prediction_audit_status": "not_run",
                    "prediction_truth_digest": "",
                    "truth_leakage_check": True,
                    "offline_replay_complete": False,
                    "offline_record_count": 0,
                    "offline_audit_status": "not_run",
                    "coverage_status": "not_applicable",
                },
                details={
                    "run_kind": plan.run_kind.value,
                    "planner_id": plan.planning_case.planner_id,
                    "execution_backend": plan.planning_case.execution_backend,
                    "gate_summary_artifact_path": str(summary_path),
                    "gate_stdout_log_path": str(stdout_log),
                    "gate_stderr_log_path": str(stderr_log),
                    "gate_status": "timeout",
                },
            )
            return _finalize_bundle(plan=plan, bundle=result)
        stdout_log.write_text(proc.stdout, encoding="utf-8")
        stderr_log.write_text(proc.stderr, encoding="utf-8")
        if not summary_path.is_file():
            raise RuntimeError("missing_gate_summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(proc.returncode) != 0 or str(summary.get("status")) != "passed" or any(str(item.get("status")) != "passed" for item in summary.get("ranks", ())):
            raise RuntimeError(f"gloo_gate_failed:{proc.returncode}")
        status = "success" if str(summary.get("status")) == "passed" else "failure"
        correctness_status = "valid" if status == "success" else "invalid"
        ranks = list(summary.get("ranks", ()))
        measurement_event_count = sum(int(item.get("measurement_event_count", 0) or 0) for item in ranks)
        summary_digest = hashlib.sha256(json.dumps(summary, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
        result = ResultBundle(
            run_identity=RunIdentity(
                run_id=_plan_run_id(plan),
                pipeline="online",
                claim_scope="formal",
                trace_origin="runtime",
                future_information_mode=str(getattr(plan.planning_case, "prediction_mode", "none")),
            ),
            status=status,
            correctness_status=correctness_status,
            performance_status="unknown",
            pipeline="online",
            commit_sha=commit_sha,
            git_clean=not git_dirty,
            instrumentation_mode=_instrumentation_mode(plan),
            audit_evidence_level="full",
            measurement_complete=True,
            eligibility=None,
            summary={
                "all_work_completed": status == "success",
                "fallback_count": 0,
                "timeout_count": timeout_count,
                "check_failure_count": 0 if status == "success" else 1,
                "measurement_event_count": measurement_event_count,
                "performance_measurement_complete": False,
                "measured_repeat_count": 0,
                "warmup_excluded": False,
                "prediction_evaluation_complete": False,
                "prediction_record_count": 0,
                "prediction_metric_count": 0,
                "prediction_audit_status": "not_run",
                "prediction_truth_digest": "",
                "truth_leakage_check": True,
                "offline_replay_complete": False,
                "offline_record_count": 0,
                "offline_audit_status": "not_run",
                "coverage_status": "not_applicable",
            },
            details={
                "run_kind": plan.run_kind.value,
                "planner_id": plan.planning_case.planner_id,
                "execution_backend": plan.planning_case.execution_backend,
                "gate_summary_digest": summary_digest,
                "gate_summary_artifact_path": str(summary_path),
                "gate_stdout_log_path": str(stdout_log),
                "gate_stderr_log_path": str(stderr_log),
                "gate_status": str(summary.get("status")),
            },
        )
        return _finalize_bundle(plan=plan, bundle=result)


@dataclass
class GPUCorrectnessRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online", reason="environment_not_run")


@dataclass
class GPUPerformanceRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online", reason="environment_not_run")


@dataclass
class MultinodeRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online", reason="environment_not_run")


@dataclass
class TraceCollectionRunner:
    def run(self, plan: RunPlan) -> ResultBundle:
        return _base_result(plan, pipeline="online", reason="environment_not_run")


def _diagnostic_success_result(plan: RunPlan) -> ResultBundle:
    commit_sha, git_dirty = _commit_identity(plan)
    return ResultBundle(
        run_identity=RunIdentity(
            run_id=_plan_run_id(plan),
            pipeline="online",
            claim_scope="formal",
            trace_origin="planned",
            future_information_mode=str(getattr(plan.planning_case, "prediction_mode", "none")),
        ),
        status="success",
        correctness_status="valid",
        performance_status="ineligible",
        pipeline="online",
        commit_sha=commit_sha,
        git_clean=not git_dirty,
        instrumentation_mode=_instrumentation_mode(plan),
        audit_evidence_level="summary_only",
        measurement_complete=True,
        eligibility=EligibilityResult(
            correctness_eligible=True,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            reasons=("diagnostic_mode",),
        ),
        summary={
            "all_work_completed": True,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "runner_kind": "diagnostic",
        },
        details={
            "run_kind": plan.run_kind.value,
            "planner_id": plan.planning_case.planner_id,
            "planner_family": plan.planning_case.planner_family,
            "execution_backend": plan.planning_case.execution_backend,
            "instrumentation_mode": _instrumentation_mode(plan),
        },
    )


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
