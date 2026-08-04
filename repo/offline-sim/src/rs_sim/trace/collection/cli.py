"""CLI for independent trace collection and one-click simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import example_config, load_pipeline_config
from .fixture_builder import build_fixtures_from_capture
from .pipeline import bundle_pipeline_artifacts, doctor, finalize_and_simulate, launch_collection, simulate_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rs-sim-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="write an editable Megatron EP4 capture config")
    init.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("doctor", help="validate config, model path, Torch and Megatron imports")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--require-megatron", action="store_true")

    collect = sub.add_parser("collect", help="launch a model command with automatic capture injection")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("model_command", nargs=argparse.REMAINDER)

    finalize = sub.add_parser("finalize", help="merge rank artifacts and build validated fixtures")
    finalize.add_argument("--config", type=Path, required=True)

    bundle = sub.add_parser("bundle", help="package raw trace, fixtures, results and SHA-256 manifest")
    bundle.add_argument("--config", type=Path, required=True)

    simulate = sub.add_parser("simulate", help="run formal Current-P12 simulation on captured fixtures")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--fixture", action="append", type=Path, default=[])

    run = sub.add_parser("run", help="one command: collect → finalize → validate → simulate")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--skip-collect", action="store_true")
    run.add_argument("model_command", nargs=argparse.REMAINDER)
    return parser


def _command_tail(values: list[str]) -> list[str]:
    return values[1:] if values and values[0] == "--" else values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-config":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(example_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "config": str(args.output)}, ensure_ascii=False, indent=2))
        return 0
    config = load_pipeline_config(args.config)
    if args.command == "doctor":
        print(json.dumps(doctor(config, require_megatron=args.require_megatron), ensure_ascii=False, indent=2))
        return 0
    if args.command == "collect":
        print(json.dumps(launch_collection(config, command_override=_command_tail(args.model_command) or None), ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize":
        paths = build_fixtures_from_capture(config)
        print(json.dumps({"status": "PASS", "fixtures": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "bundle":
        bundle_path, sha_path = bundle_pipeline_artifacts(config)
        print(json.dumps({"status": "PASS", "bundle": str(bundle_path), "sha256_file": str(sha_path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "simulate":
        fixtures = tuple(args.fixture) or tuple(sorted((Path(config["output_dir"]) / "fixtures").glob("*.json")))
        if not fixtures:
            raise SystemExit("no fixtures found; run finalize first")
        results = [simulate_fixture(config, path) for path in fixtures]
        print(json.dumps({"status": "PASS", "results": [str(path) for path in results]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        if not args.skip_collect:
            launch_collection(config, command_override=_command_tail(args.model_command) or None)
        summary = finalize_and_simulate(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
