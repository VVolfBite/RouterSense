from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .config_io import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rs-sim",
        description="RouterSense two-command Current-P12 trace and experiment system",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="collect a trace locally or execute a multi-node collection phase")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--phase", choices=("all", "prepare", "worker", "finalize"), default="all")
    collect.add_argument("--node-rank", type=int)
    collect.add_argument("--node-artifact", action="append", type=Path, default=[])
    run = sub.add_parser("run", help="run the configured experiment matrix on one or more traces")
    run.add_argument("--config", type=Path, required=True)
    sweep = sub.add_parser(
        "sweep",
        help="recursively run a trace repository and durably append one wide CSV",
    )
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--trace-root", type=Path, action="append", required=True)
    sweep.add_argument("--output-csv", type=Path, required=True)
    sweep.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    sweep.add_argument(
        "--rerun-failures", action=argparse.BooleanOptionalAction, default=True
    )
    sweep.add_argument("--max-fixtures", type=int)
    sweep.add_argument(
        "--trace-kind", choices=("measured", "projected", "all"), default="all"
    )
    sweep.add_argument(
        "--workers", type=int, default=1,
        help="number of isolated treatment workers; CSV writes remain serialized and durable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "collect":
        from .collect import run_collect
        if args.phase == "all":
            result = run_collect(config)
        else:
            from .multinode_collect import finalize_multinode, prepare_multinode, run_multinode_worker
            if args.phase == "prepare":
                result = prepare_multinode(config)
            elif args.phase == "worker":
                if args.node_rank is None:
                    raise ValueError("--node-rank is required for --phase worker")
                result = run_multinode_worker(config, node_rank=args.node_rank)
            elif args.phase == "finalize":
                result = finalize_multinode(config, node_artifacts=args.node_artifact)
            else:
                raise AssertionError(args.phase)
    elif args.command == "run":
        from .experiment import run_experiment
        result = run_experiment(config)
    elif args.command == "sweep":
        from .formal_sweep import run_repository_sweep
        result = run_repository_sweep(
            config,
            trace_roots=args.trace_root,
            output_csv=args.output_csv,
            resume=bool(args.resume),
            rerun_failures=bool(args.rerun_failures),
            max_fixtures=args.max_fixtures,
            workers=int(args.workers),
            trace_kind=str(args.trace_kind),
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


def console_main(argv: list[str] | None = None) -> None:
    """Process-level entry point with deterministic termination.

    The simulator and imported numerical runtimes may leave third-party
    non-daemon threads alive after all authoritative artifacts are committed.
    A command-line invocation is a process boundary, so flush user-visible
    output and let the operating system reclaim those resources instead of
    hanging after a successful run.  Unit tests continue to call ``main``.
    """

    code = 1
    try:
        code = int(main(argv))
    except SystemExit as exc:
        raw = exc.code
        code = int(raw) if isinstance(raw, int) else 1
        if raw not in (None, 0):
            traceback.print_exc()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


if __name__ == "__main__":
    console_main()
