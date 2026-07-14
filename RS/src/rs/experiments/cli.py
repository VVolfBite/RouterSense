from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rs.experiments.config_loader import ExperimentConfigLoader
from rs.experiments.registry import RunnerRegistry
from rs.experiments.reporting_cli import write_plot_artifacts, write_report_artifacts
from rs.experiments.specs import RunPlan


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
