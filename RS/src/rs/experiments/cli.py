from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from rs.experiments.config_loader import ExperimentConfigLoader
from rs.experiments.registry import RunnerRegistry
from rs.experiments.specs import RunPlan


def _plan_run_id(plan: RunPlan) -> str:
    experiment_id = str(getattr(plan, "experiment_id", "adhoc"))
    suite_id = str(getattr(plan, "suite_id", "suite"))
    case_id = str(getattr(plan, "case_id", "case"))
    return f"{experiment_id}:{suite_id}:{case_id}"


def _build_run_plans(config_path: str | Path, *, output_dir: str = "") -> tuple[RunPlan, ...]:
    loaded = ExperimentConfigLoader().load(config_path=config_path)
    cases = {case.case_id: case for case in loaded.spec.planning_cases}
    plans: list[RunPlan] = []
    for suite in loaded.spec.suites:
        for case_id in suite.case_ids:
            case = cases[case_id]
            plans.append(
                RunPlan(
                    experiment_id=loaded.spec.experiment_id,
                    suite_id=suite.suite_id,
                    case_id=case.case_id,
                    run_kind=case.run_kind,
                    config_digest=loaded.config_digest,
                    planning_case=case,
                    commit_sha="",
                    defaults=dict(loaded.spec.defaults),
                    config_path=str(Path(config_path).resolve()),
                    output_dir=str(output_dir),
                )
            )
    return tuple(plans)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rs.experiments.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("inspect-config", "plan", "run", "validate", "list-suites", "list-cases"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        if command == "run":
            sub.add_argument("--suite-id")
            sub.add_argument("--output-dir")

    args = parser.parse_args(argv)
    loader = ExperimentConfigLoader()
    loaded = loader.load(config_path=args.config)

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
        registry = RunnerRegistry()
        run_root = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
        if run_root is not None:
            run_root.mkdir(parents=True, exist_ok=True)
        planned_run_dirs: list[Path] = []
        for plan in _build_run_plans(args.config, output_dir=str(run_root or "")):
            if args.suite_id and plan.suite_id != args.suite_id:
                continue
            if run_root is None:
                continue
            run_id = _plan_run_id(plan).replace(":", "_")
            run_dir = run_root / "runs" / run_id
            if run_dir.exists():
                raise FileExistsError(f"run output already exists: {run_dir}")
            planned_run_dirs.append(run_dir)
        results = []
        for plan in _build_run_plans(args.config, output_dir=str(run_root or "")):
            if args.suite_id and plan.suite_id != args.suite_id:
                continue
            result = registry.resolve(plan.run_kind).run(plan)
            result_payload = result.to_dict()
            result_bundle_path = ""
            if run_root is not None:
                run_id = str(result.run_identity.run_id).replace(":", "_")
                run_dir = run_root / "runs" / run_id
                (run_dir / "logs").mkdir(parents=True, exist_ok=True)
                (run_dir / "evidence").mkdir(parents=True, exist_ok=True)
                (run_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "run_id": result.run_identity.run_id,
                            "config_path": str(Path(args.config).resolve()),
                            "config_digest": loaded.config_digest,
                            "commit_sha": result.commit_sha,
                        },
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (run_dir / "resolved_config.yaml").write_text(loaded.resolved_config_yaml, encoding="utf-8")
                (run_dir / "status.json").write_text(
                    json.dumps({"status": result.status, "run_id": result.run_identity.run_id}, ensure_ascii=True, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                result_bundle_path = str((run_dir / "result_bundle.json").resolve())
                (run_dir / "result_bundle.json").write_text(
                    json.dumps(result_payload, ensure_ascii=True, indent=2, sort_keys=True),
                    encoding="utf-8",
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
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
