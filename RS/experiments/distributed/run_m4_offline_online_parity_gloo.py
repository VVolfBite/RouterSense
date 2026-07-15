from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from rs.core.contracts import (
    ActualPhaseContext,
    EvaluationSpec,
    OfflineWindow,
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    PredictionIdentity,
    PredictionResult,
    TrafficProvenance,
)
from rs.offline import OfflinePlanningRequestBuilder, build_execution_truth
from rs.planning import PlannerRegistry
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher, _window_plan_from_payload
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer


OUT_DIR = Path("outputs/closure/m4_offline_online_parity")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], check=False, capture_output=True, text=True)
        return
    try:
        os.killpg(int(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(int(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _run_gate_subprocess(*, execution_backend: str, timeout_seconds: float = 180.0) -> dict[str, object]:
    run_dir = OUT_DIR / f"{execution_backend}_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    env = dict(os.environ)
    existing = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in ("src", ".", existing) if part)
    command = [
        sys.executable,
        str(_repo_root() / "experiments" / "distributed" / "run_m123_integrated_publication_execution_gloo.py"),
        "--execution-backend",
        str(execution_backend),
        "--instrumentation-mode",
        "perf_light",
        "--summary-path",
        str(summary_path),
        "--quiet",
    ]
    started = time.monotonic()
    popen_kwargs: dict[str, object] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        stdout, stderr = proc.communicate(timeout=15)
    stdout_log.write_text(stdout or "", encoding="utf-8")
    stderr_log.write_text(stderr or "", encoding="utf-8")
    if timed_out:
        raise TimeoutError(f"parity gate timed out for backend={execution_backend}")
    if proc.returncode != 0:
        raise RuntimeError(f"parity gate failed for backend={execution_backend} rc={proc.returncode}")
    if not summary_path.is_file():
        raise RuntimeError(f"missing gate summary for backend={execution_backend}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("status")) != "passed":
        raise RuntimeError(f"gate summary failed for backend={execution_backend}")
    summary["summary_path"] = str(summary_path)
    summary["stdout_log"] = str(stdout_log)
    summary["stderr_log"] = str(stderr_log)
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    return summary


def _offline_window(summary: dict[str, object]) -> OfflineWindow:
    p0 = tuple(tuple(int(v) for v in row) for row in summary["p0_matrix"])
    p1 = tuple(tuple(int(v) for v in row) for row in summary["p1_matrix"])
    return OfflineWindow(
        window_identity="m123-gate:0->1",
        source_layer="0",
        target_layer="1",
        p0_actual=p0,
        p1_actual=p1,
        p2_actual=p0,
        placement_snapshot={"group_size": len(p0), "fixture_type": "m123_integrated_gate"},
        traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=sum(sum(row) for row in p0),
        used_token_count=sum(sum(row) for row in p0),
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="m123-integrated-gate",
    )


def _prediction(window: OfflineWindow) -> PredictionResult:
    return PredictionResult(
        identity=PredictionIdentity(
            request_id=str(window.window_identity),
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
        hint=PredictionHint(
            predictor_id="copy_current_dispatch",
            hint_type="traffic_matrix",
            target_dispatch_rows=window.p0_actual,
            confidence=1.0,
            oracle=False,
            source_layer_id=str(window.source_layer),
            target_layer_id=str(window.target_layer),
        ),
    )


def _spec(world_size: int) -> EvaluationSpec:
    return EvaluationSpec(
        track="runtime_lookahead",
        world_size=int(world_size),
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


def _actual_phase_context(payload: dict[str, object]) -> ActualPhaseContext:
    return ActualPhaseContext(
        layer_id=str(payload["layer_id"]),
        phase=str(payload["phase"]),
        world_size=int(payload["world_size"]),
        rank_space=str(payload["rank_space"]),
        layout_digest=str(payload["layout_digest"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _planning_request_from_payload(payload: dict[str, object]) -> PlanningRequest:
    request = PlanningRequest(
        identity=PlanningIdentity(**dict(payload["identity"])),
        traffic=PlanningTraffic(
            p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in payload["traffic"]["p0_dispatch_rows"]),
            p1_return_rows=tuple(tuple(int(v) for v in row) for row in payload["traffic"]["p1_return_rows"]),
        ),
        prediction_hint=PredictionHint(
            predictor_id=str(payload["prediction_hint"]["predictor_id"]),
            hint_type=str(payload["prediction_hint"]["hint_type"]),
            target_dispatch_rows=tuple(tuple(int(v) for v in row) for row in payload["prediction_hint"]["target_dispatch_rows"]),
            confidence=payload["prediction_hint"]["confidence"],
            oracle=bool(payload["prediction_hint"]["oracle"]),
            source_layer_id=payload["prediction_hint"]["source_layer_id"],
            target_layer_id=payload["prediction_hint"]["target_layer_id"],
        ),
        topology=PlanningTopology(**dict(payload["topology"])),
        constraints=PlanningConstraints(**dict(payload["constraints"])),
        weights=PlanningWeights(**dict(payload["weights"])),
        information_mode=str(payload["information_mode"]),
    )
    request.validate()
    return request


def _transfer_counter(items: list[dict[str, object]]) -> Counter[str]:
    return Counter(
        json.dumps(
            {
                "phase": str(item["phase"]),
                "payload_role": str(item["payload_role"]),
                "src_group_rank": int(item["src_group_rank"]),
                "dst_group_rank": int(item["dst_group_rank"]),
                "row_count": int(item["row_count"]),
            },
            sort_keys=True,
        )
        for item in items
    )


def _aggregate_transfer_counter(counter: Counter[str]) -> Counter[str]:
    grouped: dict[tuple[str, str, int, int], int] = {}
    for key, multiplicity in counter.items():
        payload = json.loads(key)
        group_key = (str(payload["phase"]), str(payload["payload_role"]), int(payload["src_group_rank"]), int(payload["dst_group_rank"]))
        grouped[group_key] = int(grouped.get(group_key, 0)) + int(payload["row_count"]) * int(multiplicity)
    return Counter(
        json.dumps(
            {
                "phase": phase,
                "payload_role": payload_role,
                "src_group_rank": src_group_rank,
                "dst_group_rank": dst_group_rank,
                "row_count": row_count,
            },
            sort_keys=True,
        )
        for (phase, payload_role, src_group_rank, dst_group_rank), row_count in grouped.items()
    )


def _expected_transfer_counter_from_truth(truth) -> Counter[str]:
    rows: list[dict[str, object]] = []
    for task in truth.task_set.tasks:
        if str(task.phase) == "p0_dispatch":
            rows.append({"phase": "P0", "payload_role": "hidden_states", "src_group_rank": int(task.src_rank), "dst_group_rank": int(task.dst_rank), "row_count": int(task.row_count)})
            rows.append({"phase": "P0", "payload_role": "routing_probs", "src_group_rank": int(task.src_rank), "dst_group_rank": int(task.dst_rank), "row_count": int(task.row_count)})
        elif str(task.phase) == "p1_return":
            rows.append({"phase": "P1", "payload_role": "hidden_states", "src_group_rank": int(task.src_rank), "dst_group_rank": int(task.dst_rank), "row_count": int(task.row_count)})
    return _transfer_counter(rows)


def _evaluate_backend(*, execution_backend: str) -> dict[str, object]:
    runtime_summary = _run_gate_subprocess(execution_backend=execution_backend)
    window = _offline_window(runtime_summary)
    rank0 = dict(runtime_summary["ranks"][0])
    runtime_request = _planning_request_from_payload(dict(rank0["publication_candidate_planning_request"]))
    prediction = _prediction(window)
    spec = _spec(len(window.p0_actual))
    offline_request = OfflinePlanningRequestBuilder(
        bucket_rows=int(runtime_request.constraints.bucket_rows),
        max_waves=int(runtime_request.constraints.max_waves),
        information_mode=str(runtime_request.information_mode),
    ).build(window, prediction, spec)
    runtime_window_plan = _window_plan_from_payload(dict(rank0["window_plan"]))
    planner = PlannerRegistry.create(str(runtime_window_plan.planner_id), None)
    offline_plan = planner.plan(offline_request)
    truth = build_execution_truth(window, spec)
    input_parity_ok = str(offline_request.semantic_digest()) == str(rank0["planning_request_digest"])
    plan_parity_ok = str(offline_plan.semantic_digest()) == str(rank0["window_plan_digest"])
    publisher = CanonicalPlanPublisher(
        rank_map=RankMap(group_ranks=tuple(int(v) for v in dict(rank0["rank_map"])["group_ranks"]), root_rank=int(dict(rank0["rank_map"])["root_global_rank"]))
    )
    published_plan = publisher.build(publication_slot=dict(rank0["publication_slot"]), window_plan=runtime_window_plan)
    materialized_runtime: dict[str, str] = {}
    materialized_offline: dict[str, str] = {}
    materializer = CommonPlanMaterializer()
    for item in runtime_summary["ranks"]:
        rank = int(item["rank"])
        p0_context = _actual_phase_context(dict(item["p0_actual_phase_context"]))
        p1_context = _actual_phase_context(dict(item["p1_actual_phase_context"]))
        p0_plan = materializer.materialize(published_plan, p0_context)
        p1_plan = materializer.materialize(published_plan, p1_context)
        materialized_offline[f"rank{rank}:P0"] = str(p0_plan.materialized_plan_digest)
        materialized_offline[f"rank{rank}:P1"] = str(p1_plan.materialized_plan_digest)
        materialized_runtime[f"rank{rank}:P0"] = str(item["p0_materialized_plan_digest"])
        materialized_runtime[f"rank{rank}:P1"] = str(item["p1_materialized_plan_digest"])
    materialization_ok = materialized_offline == materialized_runtime
    expected_transfers = _expected_transfer_counter_from_truth(truth)
    actual_rows: list[dict[str, object]] = []
    for item in runtime_summary["ranks"]:
        actual_rows.extend(list(item["p0_completed_transfer_keys"]))
        actual_rows.extend(list(item["p1_completed_transfer_keys"]))
    actual_transfers = _aggregate_transfer_counter(_transfer_counter(actual_rows))
    execution_ok = actual_transfers == expected_transfers
    return {
        "status": "passed" if all((input_parity_ok, plan_parity_ok, materialization_ok, execution_ok)) else "failed",
        "execution_backend": str(execution_backend),
        "gate_summary_path": str(runtime_summary.get("summary_path", "")),
        "gate_stdout_log": str(runtime_summary.get("stdout_log", "")),
        "gate_stderr_log": str(runtime_summary.get("stderr_log", "")),
        "input_parity": {
            "status": "PASS" if input_parity_ok else "FAIL",
            "offline_planning_request_digest": str(offline_request.semantic_digest()),
            "runtime_planning_request_digest": str(rank0["planning_request_digest"]),
            "identity_parity": "PASS" if runtime_request.identity_digest() == offline_request.identity_digest() else "DIFFERENT_IDENTITY_OK",
        },
        "plan_parity": {
            "status": "PASS" if plan_parity_ok else "FAIL",
            "offline_window_plan_digest": str(offline_plan.semantic_digest()),
            "runtime_window_plan_digest": str(rank0["window_plan_digest"]),
        },
        "materialization_parity": {
            "status": "PASS" if materialization_ok else "FAIL",
            "offline": materialized_offline,
            "runtime": materialized_runtime,
        },
        "execution_semantics_parity": {
            "status": "PASS" if execution_ok else "FAIL",
            "expected_transfers": dict(expected_transfers),
            "runtime_completed_transfers": dict(actual_transfers),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_sync = _evaluate_backend(execution_backend="phase_sync")
    async_release = _evaluate_backend(execution_backend="async_release")
    summary = {
        "status": "passed" if phase_sync["status"] == "passed" and async_release["status"] == "passed" else "failed",
        "phase_sync": phase_sync,
        "async_release": async_release,
    }
    (OUT_DIR / "m4_offline_online_parity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
