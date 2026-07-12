from __future__ import annotations

import json
import sys
import traceback
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "outputs/distributed/run_stage3_guard_fault_injection_gate"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.runtime.guards import InvariantContext, InvariantFailure
from rs.runtime.guards.distributed import distributed_invariant_gate, invariant_failure_to_dict
from rs.runtime.guards.errors import RouterSenseInvariantError


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _local_failure(case_name: str, rank: int) -> InvariantFailure | None:
    if case_name == "rank1_preflight_failure":
        if rank == 1:
            return InvariantFailure(
                error_code="RS-TRANSPORT-001",
                stage="transport",
                message="forced preflight failure for distributed guard test",
                rank=rank,
                layer_id=7,
                phase="P0",
            )
        return None
    if case_name == "rank0_planning_digest_failure":
        if rank == 0:
            return InvariantFailure(
                error_code="RS-PLANNING-001",
                stage="planning",
                message="forced planning digest mismatch for distributed guard test",
                rank=rank,
                layer_id=9,
                phase="P1",
            )
        return None
    raise ValueError(f"unsupported case {case_name!r}")


def _worker(rank: int, init_file: str, case_name: str) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
        )
        local_failure = _local_failure(case_name, rank)
        distributed_invariant_gate(
            local_failure=local_failure,
            process_group=dist.group.WORLD,
            context=InvariantContext(
                stage=case_name,
                error_code=(local_failure.error_code if local_failure is not None else "RS-TRANSPORT-001"),
                rank=rank,
            ),
        )
        _write_json(
            RUN_DIR / case_name / f"rank{rank}.json",
            {"rank": rank, "case_name": case_name, "status": "unexpected_success"},
        )
    except RouterSenseInvariantError as exc:
        _write_json(
            RUN_DIR / case_name / f"rank{rank}.json",
            {
                "rank": rank,
                "case_name": case_name,
                "status": "guard_failed",
                "failure": invariant_failure_to_dict(exc.failure),
            },
        )
    except Exception as exc:  # pragma: no cover - explicit failure artifact path
        _write_json(
            RUN_DIR / case_name / f"rank{rank}.json",
            {
                "rank": rank,
                "case_name": case_name,
                "status": "unexpected_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_case(case_name: str) -> dict:
    case_dir = RUN_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"rs_guard_{case_name}_") as tmp:
        init_file = str(Path(tmp) / "dist_init")
        ctx = get_context("fork")
        procs = []
        for rank in range(2):
            proc = ctx.Process(target=_worker, args=(rank, init_file, case_name))
            proc.start()
            procs.append(proc)
        exit_codes = []
        for proc in procs:
            proc.join(timeout=20.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
                exit_codes.append(-999)
            else:
                exit_codes.append(int(proc.exitcode or 0))
    rank_payloads = [json.loads((case_dir / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    primary_codes = {payload["failure"]["error_code"] for payload in rank_payloads if payload.get("status") == "guard_failed"}
    passed = (
        all(code == 0 for code in exit_codes)
        and all(payload.get("status") == "guard_failed" for payload in rank_payloads)
        and len(primary_codes) == 1
    )
    return {
        "case_name": case_name,
        "passed": passed,
        "exit_codes": exit_codes,
        "primary_error_codes": sorted(primary_codes),
        "ranks": rank_payloads,
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        _run_case("rank1_preflight_failure"),
        _run_case("rank0_planning_digest_failure"),
    ]
    summary = {
        "passed": all(bool(item["passed"]) for item in cases),
        "case_count": len(cases),
        "cases": cases,
    }
    _write_json(RUN_DIR / "summary.json", summary)
    if not bool(summary["passed"]):
        raise SystemExit("guard fault injection gate failed")


if __name__ == "__main__":
    main()
