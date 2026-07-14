from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs.experiments.config_loader import ExperimentConfigLoader
from rs.experiments.registry import RunnerRegistry
from rs.experiments.specs import RunPlan


def _build_run_plans(config_path: str | Path) -> tuple[RunPlan, ...]:
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
        results = []
        for plan in _build_run_plans(args.config):
            if args.suite_id and plan.suite_id != args.suite_id:
                continue
            results.append(registry.resolve(plan.run_kind).run(plan).to_dict())
        print(json.dumps(results, ensure_ascii=True, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
