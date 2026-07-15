from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from rs.core.contracts import EvaluationSpec, OfflineEvaluationRecord
from rs.experiments.config_loader import ExperimentConfigLoader
from rs.experiments.registry import RunnerRegistry
from rs.experiments.reporting_cli import write_plot_artifacts, write_report_artifacts
from rs.experiments.specs import RunPlan
from rs.offline import build_evaluation_bundle


def _plan_run_id(plan: RunPlan) -> str:
    experiment_id = str(getattr(plan, "experiment_id", "adhoc"))
    suite_id = str(getattr(plan, "suite_id", "suite"))
    case_id = str(getattr(plan, "case_id", "case"))
    repeat_index = int(getattr(plan, "repeat_index", 0) or 0)
    seed = int(getattr(plan, "seed", 0) or 0)
    return f"{experiment_id}:{suite_id}:{case_id}:r{repeat_index}:s{seed}"


def _seed_values(defaults: dict[str, Any]) -> tuple[int, ...]:
    if "seeds" in defaults and isinstance(defaults["seeds"], list):
        return tuple(int(item) for item in defaults["seeds"])
    if "seed" in defaults:
        return (int(defaults["seed"]),)
    return (0,)


def _build_run_plans(config_path: str | Path, *, output_dir: str = "") -> tuple[RunPlan, ...]:
    loaded = ExperimentConfigLoader().load(config_path=config_path)
    defaults = dict(loaded.spec.defaults)
    evaluation = dict(defaults.get("evaluation", {}) or {})
    workload = dict(defaults.get("workload", {}) or {})
    repeats = max(1, int(evaluation.get("repeats", 1) or 1))
    warmup = max(0, int(evaluation.get("warmup", 0) or 0))
    max_windows = max(0, int(workload.get("max_windows", 0) or 0))
    cases = {case.case_id: case for case in loaded.spec.planning_cases}
    plans: list[RunPlan] = []
    for suite in loaded.spec.suites:
        for case_id in suite.case_ids:
            case = cases[case_id]
            for repeat_index in range(repeats):
                for seed in _seed_values(defaults):
                    plans.append(
                        RunPlan(
                            experiment_id=loaded.spec.experiment_id,
                            suite_id=suite.suite_id,
                            case_id=case.case_id,
                            run_kind=case.run_kind,
                            config_digest=loaded.config_digest,
                            planning_case=case,
                            commit_sha="",
                            defaults=dict(defaults),
                            config_path=str(Path(config_path).resolve()),
                            output_dir=str(output_dir),
                            repeat_index=repeat_index,
                            seed=int(seed),
                            warmup=warmup,
                            max_windows=max_windows,
                        )
                    )
    return tuple(plans)


def _write_run_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    loaded: Any,
    plan: RunPlan,
    result_payload: dict[str, Any],
) -> str:
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": result_payload["run_identity"]["run_id"],
                "config_path": str(config_path.resolve()),
                "config_digest": loaded.config_digest,
                "commit_sha": result_payload["commit_sha"],
                "commit_sha_source": str(result_payload.get("details", {}).get("commit_sha_source", "")),
                "suite_id": plan.suite_id,
                "case_id": plan.case_id,
                "repeat_index": int(plan.repeat_index),
                "seed": int(plan.seed),
                "warmup": int(plan.warmup),
                "max_windows": int(plan.max_windows),
                "status": result_payload["status"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.yaml").write_text(loaded.resolved_config_yaml, encoding="utf-8")
    (run_dir / "migration_report.json").write_text(
        json.dumps(loaded.migration_report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": result_payload["status"],
                "run_id": result_payload["run_identity"]["run_id"],
                "correctness_status": result_payload["correctness_status"],
                "performance_status": result_payload["performance_status"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result_bundle_path = run_dir / "result_bundle.json"
    result_bundle_path.write_text(json.dumps(result_payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return str(result_bundle_path.resolve())


def _offline_record_from_payload(payload: dict[str, Any]) -> OfflineEvaluationRecord | None:
    details = dict(payload.get("details", {}))
    record = details.get("offline_record")
    if not isinstance(record, dict):
        return None
    return OfflineEvaluationRecord(
        window_identity=str(record["window_identity"]),
        evaluation_spec_digest=str(record["evaluation_spec_digest"]),
        task_set_digest=str(record["task_set_digest"]),
        planning_request_digest=str(record["planning_request_digest"]),
        prediction_digest=str(record["prediction_digest"]),
        logical_plan_digest=str(record["logical_plan_digest"]),
        execution_truth_digest=str(record["execution_truth_digest"]),
        planner_id=str(record["planner_id"]),
        planner_family=str(record["planner_family"]),
        predictor_id=str(record["predictor_id"]),
        track=str(record["track"]),
        realized_makespan=record["realized_makespan"],
        planner_reported_makespan=record["planner_reported_makespan"],
        audit_status=str(record["audit_status"]),
        coverage_status=str(record["coverage_status"]),
        fallback_status=str(record["fallback_status"]),
        oracle_status=str(record["oracle_status"]),
        eligibility=dict(record.get("eligibility", {})),
        metrics=dict(record.get("metrics", {})),
    )


def _eligible_offline_record(record: OfflineEvaluationRecord) -> bool:
    eligibility = dict(record.eligibility)
    return (
        str(record.audit_status) == "valid"
        and str(record.coverage_status) == "complete"
        and str(record.fallback_status) == "none"
        and record.realized_makespan is not None
        and bool(eligibility.get("offline_replay_eligible", False))
    )


def _paired_percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    index = int((len(ordered) - 1) * float(quantile))
    return float(ordered[index])


def _paired_case_aggregate(
    records: list[OfflineEvaluationRecord],
    *,
    comparison_id: str,
    baseline_planner_id: str,
    baseline_predictor_id: str,
    candidate_planner_id: str,
    candidate_predictor_id: str,
    track: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str], list[OfflineEvaluationRecord]] = {}
    for record in records:
        if str(record.track) != str(track):
            continue
        metrics = dict(record.metrics)
        key = (
            str(metrics.get("trace_digest", record.window_identity)),
            str(record.window_identity),
            str(record.evaluation_spec_digest),
            str(record.task_set_digest),
            str(record.execution_truth_digest),
            str(record.track),
            str(metrics.get("repeat", "")),
            str(metrics.get("seed", "")),
        )
        grouped.setdefault(key, []).append(record)
    gains: list[float] = []
    exclusion_reasons: dict[str, int] = {}
    for bucket in grouped.values():
        baseline = next(
            (
                item
                for item in bucket
                if str(item.planner_id) == str(baseline_planner_id)
                and str(item.predictor_id) == str(baseline_predictor_id)
            ),
            None,
        )
        candidate = next(
            (
                item
                for item in bucket
                if str(item.planner_id) == str(candidate_planner_id)
                and str(item.predictor_id) == str(candidate_predictor_id)
            ),
            None,
        )
        if baseline is None or candidate is None:
            exclusion_reasons["missing_pair"] = int(exclusion_reasons.get("missing_pair", 0)) + 1
            continue
        if not _eligible_offline_record(baseline):
            exclusion_reasons["baseline_ineligible"] = int(exclusion_reasons.get("baseline_ineligible", 0)) + 1
            continue
        if not _eligible_offline_record(candidate):
            exclusion_reasons["candidate_ineligible"] = int(exclusion_reasons.get("candidate_ineligible", 0)) + 1
            continue
        gains.append(float(baseline.realized_makespan) - float(candidate.realized_makespan))
    wins = sum(1 for item in gains if float(item) > 0.0)
    ties = sum(1 for item in gains if float(item) == 0.0)
    losses = sum(1 for item in gains if float(item) < 0.0)
    return {
        "comparison_id": str(comparison_id),
        "track": str(track),
        "baseline_planner_id": str(baseline_planner_id),
        "baseline_predictor_id": str(baseline_predictor_id),
        "candidate_planner_id": str(candidate_planner_id),
        "candidate_predictor_id": str(candidate_predictor_id),
        "sample_count": int(len(gains)),
        "excluded_count": int(sum(int(value) for value in exclusion_reasons.values())),
        "exclusion_reasons": dict(exclusion_reasons),
        "mean": (sum(gains) / len(gains)) if gains else None,
        "median": median(gains) if gains else None,
        "p25": _paired_percentile(gains, 0.25),
        "p50": _paired_percentile(gains, 0.50),
        "p75": _paired_percentile(gains, 0.75),
        "p90": _paired_percentile(gains, 0.90),
        "positive_window_rate": (wins / len(gains)) if gains else None,
        "negative_window_rate": (losses / len(gains)) if gains else None,
        "tie_rate": (ties / len(gains)) if gains else None,
        "win_rate": (wins / len(gains)) if gains else None,
        "loss_rate": (losses / len(gains)) if gains else None,
        "best": max(gains) if gains else None,
        "worst": min(gains) if gains else None,
    }


def _write_offline_suite_aggregates(*, suite_dir: Path, suite_id: str, suite_results: list[dict[str, Any]]) -> None:
    records = [item for item in (_offline_record_from_payload(payload) for payload in suite_results) if item is not None]
    if not records:
        return
    first_offline_payload = next(
        payload for payload in suite_results if isinstance(dict(payload.get("details", {})).get("offline_bundle"), dict)
    )
    eval_spec_payload = dict(first_offline_payload.get("details", {}).get("offline_bundle", {}).get("evaluation_spec", {}))
    spec = EvaluationSpec(**eval_spec_payload)
    track = str(spec.track)
    paired = [
        _paired_case_aggregate(
            records,
            comparison_id="u_vs_b_zero",
            baseline_planner_id="birkhoff_bucket_phase_local",
            baseline_predictor_id="zero_hint",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="zero_hint",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="u_vs_b_copy",
            baseline_planner_id="birkhoff_bucket_phase_local",
            baseline_predictor_id="copy_current_dispatch",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="copy_current_dispatch",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="u_vs_b_perfect",
            baseline_planner_id="birkhoff_bucket_phase_local",
            baseline_predictor_id="perfect_trace_hint",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="perfect_trace_hint",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="b_vs_fifo_zero",
            baseline_planner_id="fifo_bucket",
            baseline_predictor_id="zero_hint",
            candidate_planner_id="birkhoff_bucket_phase_local",
            candidate_predictor_id="zero_hint",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="b_vs_greedy_zero",
            baseline_planner_id="greedy_bucket",
            baseline_predictor_id="zero_hint",
            candidate_planner_id="birkhoff_bucket_phase_local",
            candidate_predictor_id="zero_hint",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="safe_u_vs_b_copy",
            baseline_planner_id="birkhoff_bucket_phase_local",
            baseline_predictor_id="copy_current_dispatch",
            candidate_planner_id="barrier_criticality_runtime_safe",
            candidate_predictor_id="copy_current_dispatch",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="copy_vs_zero_u",
            baseline_planner_id="barrier_criticality_joint",
            baseline_predictor_id="zero_hint",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="copy_current_dispatch",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="perfect_vs_zero_u",
            baseline_planner_id="barrier_criticality_joint",
            baseline_predictor_id="zero_hint",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="perfect_trace_hint",
            track=track,
        ),
        _paired_case_aggregate(
            records,
            comparison_id="copy_vs_perfect_u",
            baseline_planner_id="barrier_criticality_joint",
            baseline_predictor_id="perfect_trace_hint",
            candidate_planner_id="barrier_criticality_joint",
            candidate_predictor_id="copy_current_dispatch",
            track=track,
        ),
    ]
    bundle = build_evaluation_bundle(
        spec=spec,
        records=tuple(records),
        paired_aggregates=tuple(paired),
        eligibility_summary={
            "suite_id": str(suite_id),
            "record_count": len(records),
            "paired_aggregate_count": len(paired),
        },
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": bundle.schema_version,
        "evaluation_spec": bundle.evaluation_spec.to_dict(),
        "record_count": len(bundle.records),
        "paired_aggregate_count": len(bundle.paired_aggregates),
        "paired_aggregates": list(bundle.paired_aggregates),
        "eligibility_summary": dict(bundle.eligibility_summary),
    }
    (suite_dir / "offline_evaluation_bundle.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _run_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    loaded = ExperimentConfigLoader().load(config_path=config_path)
    registry = RunnerRegistry()
    run_root = Path(args.output_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    plans = tuple(
        plan
        for plan in _build_run_plans(config_path, output_dir=str(run_root))
        if not args.suite_id or plan.suite_id == args.suite_id
    )
    for plan in plans:
        run_id = _plan_run_id(plan).replace(":", "_")
        run_dir = run_root / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(f"run output already exists: {run_dir}")
    results = []
    for plan in plans:
        result = registry.resolve(plan.run_kind).run(plan)
        result_payload = result.to_dict()
        run_id = str(result.run_identity.run_id).replace(":", "_")
        run_dir = run_root / "runs" / run_id
        result_bundle_path = _write_run_artifacts(
            run_dir=run_dir,
            config_path=config_path,
            loaded=loaded,
            plan=plan,
            result_payload=result_payload,
        )
        results.append(
            {
                "status": result.status,
                "run_id": result.run_identity.run_id,
                "result_bundle_path": result_bundle_path,
                "eligibility": None if result.eligibility is None else result.eligibility.to_dict(),
            }
        )
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        run_id = _plan_run_id(plan).replace(":", "_")
        payload = json.loads((run_root / "runs" / run_id / "result_bundle.json").read_text(encoding="utf-8"))
        by_suite.setdefault(str(plan.suite_id), []).append(payload)
    for suite_id, suite_results in by_suite.items():
        _write_offline_suite_aggregates(
            suite_dir=run_root / "suites" / str(suite_id),
            suite_id=str(suite_id),
            suite_results=suite_results,
        )
    print(json.dumps({"status": "success", "runs": results}, ensure_ascii=True, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rs.experiments.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-config")
    inspect_parser.add_argument("--config", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)

    list_suites_parser = subparsers.add_parser("list-suites")
    list_suites_parser.add_argument("--config", required=True)

    list_cases_parser = subparsers.add_parser("list-cases")
    list_cases_parser.add_argument("--config", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--suite-id")
    run_parser.add_argument("--output-dir", required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--input-dir", required=True)
    report_parser.add_argument("--output-dir", required=True)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--input-dir", required=True)
    plot_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.command in {"inspect-config", "validate", "list-suites", "list-cases", "plan"}:
        loaded = ExperimentConfigLoader().load(config_path=args.config)
        if args.command == "inspect-config":
            print(json.dumps(loaded.spec.to_dict(), ensure_ascii=True, sort_keys=True))
            return 0
        if args.command == "validate":
            print(json.dumps({"status": "ok", "config_digest": loaded.config_digest}, ensure_ascii=True, sort_keys=True))
            return 0
        if args.command == "list-suites":
            print(json.dumps([suite.to_dict() for suite in loaded.spec.suites], ensure_ascii=True, sort_keys=True))
            return 0
        if args.command == "list-cases":
            print(json.dumps([case.to_dict() for case in loaded.spec.planning_cases], ensure_ascii=True, sort_keys=True))
            return 0
        if args.command == "plan":
            print(json.dumps([plan.to_dict() for plan in _build_run_plans(args.config)], ensure_ascii=True, sort_keys=True))
            return 0
    if args.command == "run":
        return _run_command(args)
    if args.command == "report":
        artifacts = write_report_artifacts(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
        print(json.dumps({"status": "success", "artifacts": artifacts}, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "plot":
        artifacts = write_plot_artifacts(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
        print(json.dumps({"status": "success", "artifacts": artifacts}, ensure_ascii=True, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
