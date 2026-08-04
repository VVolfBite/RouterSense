from __future__ import annotations

"""One-command durable formal sweep launcher.

This entry point intentionally requires only a trace repository and an output
CSV.  Presets define the treatment matrix; every completed treatment is
committed to the CSV with flush+fsync and can be resumed after interruption.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .formal_sweep import run_repository_sweep


def _treatment(
    name: str,
    algorithm: str,
    *,
    information: str,
    release_mode: str,
    role: str,
    joint_diagnostic: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "algorithm": algorithm,
        "information": information,
        "overlap": "OVERLAP",
        "experiment_role": role,
        "release_mode": release_mode,
    }
    if joint_diagnostic:
        row["allow_matched_joint_diagnostic"] = True
    return row


LOCAL_BASELINES = (
    _treatment("FIFO-Local", "local(global_(fifo()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="PAPER_BASELINE"),
    _treatment("Greedy-Local", "local(global_(greedy()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="PAPER_BASELINE"),
    _treatment("Birkhoff-Local", "local(global_(birkhoff()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="PAPER_BASELINE"),
    _treatment("iSLIP-Local", "local(global_(islip()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="EXPLICIT_DIAGNOSTIC_ABLATION"),
    _treatment("Residual-MWM-Local", "local(global_(residual_mwm()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="PAPER_BASELINE"),
    _treatment("FAST-Local", "local(global_(fast()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="PAPER_BASELINE"),
    _treatment("Aurora-Local", "local(global_(aurora()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="EXPLICIT_DIAGNOSTIC_ABLATION"),
    _treatment("RSCF-Local", "local(global_(rscf()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="JOINT_ABLATION"),
)

JOINT_GENERICITY = tuple(
    _treatment(
        f"{label}-Joint",
        f"joint(global_({core}()))",
        information="FATE_P2",
        release_mode="RANK_LOCAL",
        role="EXPLICIT_DIAGNOSTIC_ABLATION",
        joint_diagnostic=True,
    )
    for label, core in (
        ("FIFO", "fifo"),
        ("Greedy", "greedy"),
        ("Birkhoff", "birkhoff"),
        ("iSLIP", "islip"),
        ("Residual-MWM", "residual_mwm"),
        ("FAST", "fast"),
        ("Aurora", "aurora"),
    )
)

RSCF_VARIANTS = (
    _treatment("RSCF-Joint-Zero", "joint(global_(rscf()))", information="ZERO_P2", release_mode="RANK_LOCAL", role="PREDICTION_ABLATION"),
    _treatment("RSCF-Joint-FATE", "joint(global_(rscf()))", information="FATE_P2", release_mode="RANK_LOCAL", role="PAPER_TREATMENT"),
    _treatment("RSCF-Joint-Perfect", "joint(global_(rscf()))", information="PERFECT_P2", release_mode="RANK_LOCAL", role="PREDICTION_UPPER_BOUND"),
)

ORACLE_REFERENCES = (
    _treatment("Oracle-Local", "local(global_(oracle()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="CLAIRVOYANT_REFERENCE"),
    _treatment("Oracle-Joint", "joint(global_(oracle()))", information="PERFECT_P2", release_mode="RANK_LOCAL", role="CLAIRVOYANT_REFERENCE"),
)

SMOKE = (
    LOCAL_BASELINES[0],
    LOCAL_BASELINES[2],
    RSCF_VARIANTS[1],
)

MAIN = (
    LOCAL_BASELINES[0],  # FIFO
    LOCAL_BASELINES[1],  # Greedy
    LOCAL_BASELINES[2],  # Birkhoff
    LOCAL_BASELINES[3],  # iSLIP
    LOCAL_BASELINES[4],  # Residual-MWM / GMWD-style
    LOCAL_BASELINES[5],  # FAST
    LOCAL_BASELINES[6],  # Aurora
    LOCAL_BASELINES[7],  # matched RSCF-Local
    *RSCF_VARIANTS,
)

ORACLE = (
    LOCAL_BASELINES[0],  # FIFO comparator
    LOCAL_BASELINES[2],  # Birkhoff comparator
    LOCAL_BASELINES[7],  # RSCF-Local comparator
    RSCF_VARIANTS[2],   # RSCF-Joint-Perfect comparator
    *ORACLE_REFERENCES,
)

PAPER = (
    *MAIN,
    *ORACLE_REFERENCES,
)

ALL = (
    _treatment("Null-Local", "local(global_(null()))", information="ZERO_P2", release_mode="PHASE_BARRIER", role="EXPLICIT_DIAGNOSTIC_ABLATION"),
    *LOCAL_BASELINES,
    *JOINT_GENERICITY,
    *RSCF_VARIANTS,
    *ORACLE_REFERENCES,
)

PRESETS = {"smoke": SMOKE, "main": MAIN, "oracle": ORACLE, "paper": PAPER, "all": ALL}


def build_formal_config(
    *,
    trace_roots: list[Path],
    output_csv: Path,
    preset: str,
    max_task_bytes: int,
    runtime_profile: Path | None,
    oracle_time_limit_ms: int,
    per_run_timeout_seconds: int,
) -> dict[str, Any]:
    treatments = PRESETS[preset]
    return {
        "version": 1,
        "name": f"routersense_formal_{preset}",
        "traces": [str(path) for path in trace_roots],
        "simulation": {
            "release_mode": "RANK_LOCAL",
            "p0_p1_compute_end_barrier": True,
            "staging": "1.0X",
            "max_task_bytes": int(max_task_bytes),
            "max_window_prefix_tasks": 4096,
            "alignment_bytes": 256,
            "max_timestamps": 2_000_000,
            **({"runtime_profile": str(runtime_profile)} if runtime_profile is not None else {}),
        },
        "experiments": {"treatments": [dict(row) for row in treatments]},
        "repetitions": {"warmup": 0, "measure": 1, "seed": 1234},
        "execution": {
            "mode": "SUBPROCESS_ISOLATED",
            "per_run_timeout_seconds": int(per_run_timeout_seconds),
            "kill_grace_seconds": 5,
            "fail_fast": False,
        },
        "oracle": {
            "time_limit_ms_per_window": int(oracle_time_limit_ms),
            "relative_gap": 0.05,
            "require_all_certified": False,
        },
        "comparison": {
            "claim_mode": "DIAGNOSTIC",
            "baseline": "FIFO-Local",
            "primary_metric": "compute_excluded_communication_makespan_ns_sum",
            "target_improvement_percent": 0.0,
            "tie_tolerance_percent": 0.05,
            "minimum_paired_samples": 1,
        },
        "output": {
            "directory": str(output_csv.parent),
            "complete_csv_filename": output_csv.name,
            "overwrite": False,
            "save_raw_events": False,
            "save_task_timeline": False,
            "save_plans": False,
            "raw_only": True,
        },
        "__config_path": str(output_csv.parent / "generated_formal_config.json"),
        "__config_dir": str(output_csv.parent),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rs-sim-formal",
        description="run a crash-resilient RouterSense trace repository sweep",
    )
    parser.add_argument("--trace-root", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="main")
    parser.add_argument(
        "--trace-kind",
        choices=("measured", "projected", "all"),
        default="measured",
        help=(
            "select measured traces, projected EP16/EP32 traces, or both; "
            "the paper-facing default is measured"
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--task-bytes", type=int, default=262_144)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--oracle-ms", type=int, default=1_000)
    parser.add_argument("--per-run-timeout-seconds", type=int, default=1_200)
    parser.add_argument("--max-fixtures", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-failures", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trace_roots = [path.expanduser().resolve() for path in args.trace_root]
    output_csv = args.output_csv.expanduser().resolve()
    runtime_profile = None if args.runtime_profile is None else args.runtime_profile.expanduser().resolve()
    if runtime_profile is None:
        print(
            "WARNING: no --runtime-profile supplied; using the built-in "
            "SYNTHETIC_TEST_ONLY profile and DIAGNOSTIC claim mode.",
            file=sys.stderr,
        )
    config = build_formal_config(
        trace_roots=trace_roots,
        output_csv=output_csv,
        preset=str(args.preset),
        max_task_bytes=int(args.task_bytes),
        runtime_profile=runtime_profile,
        oracle_time_limit_ms=int(args.oracle_ms),
        per_run_timeout_seconds=int(args.per_run_timeout_seconds),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated = output_csv.with_suffix(output_csv.suffix + ".config.json")
    generated.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = run_repository_sweep(
        config,
        trace_roots=trace_roots,
        output_csv=output_csv,
        resume=bool(args.resume),
        rerun_failures=bool(args.rerun_failures),
        max_fixtures=args.max_fixtures,
        workers=int(args.workers),
        trace_kind=str(args.trace_kind),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "INVALID_ORACLE_REFERENCE"} else 1


def console_main(argv: list[str] | None = None) -> None:
    code = 1
    try:
        code = int(main(argv))
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
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
