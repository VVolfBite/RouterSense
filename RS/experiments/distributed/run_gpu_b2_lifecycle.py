#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_yaml,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_selected_layers(selected_layers: str) -> set[str] | None:
    raw = str(selected_layers).strip()
    if raw in {"", "all"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _candidate_artifact_dir(run_root: Path, run_name: str) -> Path:
    return run_root / run_name


def _collect_candidate_evidence(candidate_dir: Path, *, world_size: int, selected_layers: set[str] | None) -> dict:
    rank_summaries = [read_json(candidate_dir / f"rank{rank}_summary.json") for rank in range(world_size)]
    rank_timelines = [_read_jsonl(candidate_dir / f"rank{rank}_control_timeline.jsonl") for rank in range(world_size)]
    prediction_audits = [_read_jsonl(candidate_dir / f"rank{rank}_prediction_audit.jsonl") for rank in range(world_size)]

    plan_store_events: list[dict] = []
    p1_consumed_events: list[dict] = []
    obs_by_key: dict[tuple[int, str, int], dict] = {}
    for rank, timeline_rows in enumerate(rank_timelines):
        for row in timeline_rows:
            layer_id = str(row.get("layer_id", ""))
            if selected_layers is not None and layer_id not in selected_layers:
                continue
            if row.get("event") == "runtime_joint_window_plan_stored":
                plan_store_events.append(dict(row))
            elif row.get("event") == "prepared_p1_plan_consumed":
                p1_consumed_events.append(dict(row))
            elif row.get("event") == "p0_pre_transport_observation_ready":
                obs_by_key[(int(row.get("forward_epoch", 0) or 0), layer_id, rank)] = dict(row)

    plan_store_events.sort(key=lambda item: (int(item.get("forward_epoch", 0) or 0), int(item.get("layer_id", 0) or 0), int(item.get("rank", 0) or 0), int(item.get("event_seq", 0) or 0)))
    p1_consumed_events.sort(key=lambda item: (int(item.get("forward_epoch", 0) or 0), int(item.get("layer_id", 0) or 0), int(item.get("rank", 0) or 0), int(item.get("event_seq", 0) or 0)))

    matrix_records: list[dict] = []
    dispatcher_consistency = True
    nonzero_matrix = False
    p1_exact_transpose = True
    epochs_layers: set[tuple[int, str]] = set()
    for epoch, layer_id, _rank in sorted(obs_by_key):
        epochs_layers.add((epoch, layer_id))
    for epoch, layer_id in sorted(epochs_layers):
        rows: list[list[int]] = []
        for rank in range(world_size):
            row = obs_by_key.get((epoch, layer_id, rank))
            if row is None:
                dispatcher_consistency = False
                continue
            local_p0_row = [int(value) for value in row.get("local_p0_row", [])]
            input_splits = [int(value) for value in row.get("input_splits", [])]
            if local_p0_row != input_splits:
                dispatcher_consistency = False
            rows.append(local_p0_row)
        if len(rows) != world_size:
            continue
        matrix_total = int(sum(sum(item) for item in rows))
        inferred_p1 = [[int(rows[src][dst]) for src in range(world_size)] for dst in range(world_size)]
        nonzero_matrix = nonzero_matrix or matrix_total > 0
        matrix_records.append(
            {
                "forward_epoch": int(epoch),
                "layer_id": str(layer_id),
                "actual_p0_full_row_matrix": rows,
                "actual_p0_total_rows": matrix_total,
                "inferred_p1_row_matrix": inferred_p1,
                "actual_p0_matrix_unit": "rows",
            }
        )

    stored_by_key = {
        (int(item.get("forward_epoch", 0) or 0), str(item.get("layer_id", ""))): str(item.get("stored_p1_plan_digest", "") or "")
        for item in plan_store_events
        if int(item.get("rank", 0) or 0) == 0
    }
    consumed_by_key = {
        (int(item.get("forward_epoch", 0) or 0), str(item.get("layer_id", ""))): str(item.get("consumed_p1_plan_digest", "") or "")
        for item in p1_consumed_events
        if int(item.get("rank", 0) or 0) == 0
    }
    digest_pairs: list[dict] = []
    for key, stored_digest in sorted(stored_by_key.items()):
        consumed_digest = consumed_by_key.get(key, "")
        digest_pairs.append(
            {
                "forward_epoch": int(key[0]),
                "layer_id": str(key[1]),
                "stored_p1_plan_digest": stored_digest,
                "consumed_p1_plan_digest": consumed_digest,
                "match": bool(stored_digest) and stored_digest == consumed_digest,
            }
        )
    all_digest_match = bool(digest_pairs) and all(item["match"] for item in digest_pairs)

    zero_hint_joint_plan_built = any(
        str(item.get("predictor_name", "")) == "zero_hint" and bool(item.get("stored_p1_plan_digest", ""))
        for item in plan_store_events
    )
    prediction_consumed = any(bool(item.get("consumed_during_p0_joint_planning", False)) for item in plan_store_events)

    audit_records: list[dict] = []
    for rank, rows in enumerate(prediction_audits):
        for row in rows:
            layer_id = str(row.get("layer_id", ""))
            if selected_layers is not None and layer_id not in selected_layers:
                continue
            audit_records.append({"rank": rank, **row})

    rank0_summary = rank_summaries[0]
    repeat_records = [row for row in (rank0_summary.get("repeat_records") or []) if not bool(row.get("warmup", False))]

    return {
        "rank_summaries": rank_summaries,
        "matrix_records": matrix_records,
        "plan_store_events": plan_store_events,
        "p1_consumed_events": p1_consumed_events,
        "prediction_audits": audit_records,
        "dispatcher_consistency": dispatcher_consistency,
        "actual_p0_total_rows_nonzero": nonzero_matrix,
        "p1_exact_transpose": p1_exact_transpose and all(bool(item.get("p1_is_exact_transpose", False)) for item in plan_store_events if int(item.get("rank", 0) or 0) == 0),
        "digest_pairs": digest_pairs,
        "stored_equals_consumed": all_digest_match,
        "zero_hint_joint_plan_built": zero_hint_joint_plan_built,
        "prediction_consumed_during_p0_joint_planning": prediction_consumed,
        "forward_epochs_tested": len(repeat_records),
        "rank0_summary": rank0_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU B2 lifecycle validation body or an explicit no-4GPU fallback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", default="routersense_joint_predicted_async_p2p")
    parser.add_argument("--profile", default="execution", choices=("debug", "execution", "perf"))
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fallback(output_dir: Path, *, world_size: int, config: str, strategy: str, profile: str, selected_layers: str, dry_run: bool) -> dict:
    gate_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
    ]
    proc = run_subprocess(gate_cmd)
    payload = {
        "runner": "run_gpu_b2_lifecycle",
        "config": str(config),
        "strategy": str(strategy),
        "profile": str(profile),
        "selected_layers": str(selected_layers),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "result_eligible_for_performance_comparison": False,
        "fallback_used": True,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": gate_cmd,
        "fallback_returncode": int(proc.returncode),
        "fallback_stdout_path": str(output_dir / "fallback_stdout.log"),
        "fallback_stderr_path": str(output_dir / "fallback_stderr.log"),
    }
    (output_dir / "fallback_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return payload


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_yaml(Path(args.config))
    payload = {
        "runner": "run_gpu_b2_lifecycle",
        "config": str(args.config),
        "strategy": str(args.strategy),
        "profile": str(args.profile),
        "selected_layers": str(args.selected_layers),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
        "checks": {
            "prediction_extra_collective_count_expected": 0,
            "p1_planning_collective_count_expected": 0,
            "async_executor_required": True,
            "stored_plan_digest_required": True,
        },
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(args.world_size):
        payload = _fallback(
            output_dir,
            world_size=int(args.world_size),
            config=str(args.config),
            strategy=str(args.strategy),
            profile=str(args.profile),
            selected_layers=str(args.selected_layers),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    candidate_run_dir = output_dir / "candidate"
    reference_run_dir = output_dir / "reference"
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name=str(args.strategy),
        run_name="b2_candidate",
        output_root=candidate_run_dir,
        profile=str(args.profile),
        selected_layers=str(args.selected_layers),
        save_logits=False,
    )
    candidate_config_path = generated_dir / "candidate.yaml"
    dump_yaml(candidate_config_path, candidate_config)
    candidate_cmd = torchrun_policy_command(
        config_path=candidate_config_path,
        run_id="b2_candidate",
        output_dir=candidate_run_dir,
        world_size=int(args.world_size),
        native=False,
    )
    candidate_proc = run_subprocess(candidate_cmd, extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": "2"})
    (output_dir / "candidate_stdout.log").write_text(candidate_proc.stdout, encoding="utf-8")
    (output_dir / "candidate_stderr.log").write_text(candidate_proc.stderr, encoding="utf-8")
    if candidate_proc.returncode != 0:
        payload.update(
            {
                "status": "candidate_failed",
                "candidate_command": candidate_cmd,
                "candidate_returncode": int(candidate_proc.returncode),
            }
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return int(candidate_proc.returncode)

    reference_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="native",
        run_name="b2_reference",
        output_root=reference_run_dir,
        profile=str(args.profile),
        selected_layers=str(args.selected_layers),
        save_logits=False,
    )
    reference_config_path = generated_dir / "reference.yaml"
    dump_yaml(reference_config_path, reference_config)
    reference_cmd = torchrun_policy_command(
        config_path=reference_config_path,
        run_id="b2_reference",
        output_dir=reference_run_dir,
        world_size=int(args.world_size),
        native=True,
    )
    reference_proc = run_subprocess(reference_cmd, extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": "2"})
    (output_dir / "reference_stdout.log").write_text(reference_proc.stdout, encoding="utf-8")
    (output_dir / "reference_stderr.log").write_text(reference_proc.stderr, encoding="utf-8")
    if reference_proc.returncode != 0:
        payload.update(
            {
                "status": "reference_failed",
                "reference_command": reference_cmd,
                "reference_returncode": int(reference_proc.returncode),
            }
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return int(reference_proc.returncode)

    candidate_summary_payload = read_json(candidate_run_dir / "b2_candidate" / "summary.json")
    selected_layer_set = _parse_selected_layers(str(args.selected_layers))
    candidate_evidence = _collect_candidate_evidence(
        _candidate_artifact_dir(candidate_run_dir, "b2_candidate"),
        world_size=int(args.world_size),
        selected_layers=selected_layer_set,
    )
    rank0_summary = dict(candidate_evidence["rank0_summary"])
    phase_sync_fallback_count = int(rank0_summary.get("phase_sync_fallback_count", 0) or 0)
    plan_store_events = [item for item in candidate_evidence["plan_store_events"] if int(item.get("rank", 0) or 0) == 0]
    latest_plan_event = plan_store_events[-1] if plan_store_events else {}
    matrix_records = list(candidate_evidence["matrix_records"])
    representative_matrix = matrix_records[0] if matrix_records else {}
    digest_pairs = list(candidate_evidence["digest_pairs"])
    representative_digest = digest_pairs[0] if digest_pairs else {}
    payload.update(
        {
            "stored_p1_plan_digest": representative_digest.get("stored_p1_plan_digest", ""),
            "consumed_p1_plan_digest": representative_digest.get("consumed_p1_plan_digest", ""),
            "planning_traffic_source": latest_plan_event.get("planning_traffic_source", ""),
            "pre_transport_observation_valid": latest_plan_event.get("pre_transport_observation_valid", False),
            "captured_before_transport": latest_plan_event.get("captured_before_transport", False),
            "dispatcher_send_splits": representative_matrix.get("actual_p0_full_row_matrix", [[]])[0] if representative_matrix.get("actual_p0_full_row_matrix") else [],
            "dispatcher_recv_splits": [],
            "local_p0_row": representative_matrix.get("actual_p0_full_row_matrix", [[]])[0] if representative_matrix.get("actual_p0_full_row_matrix") else [],
            "actual_p0_full_row_matrix": representative_matrix.get("actual_p0_full_row_matrix", []),
            "actual_p0_total_rows": representative_matrix.get("actual_p0_total_rows", 0),
            "actual_p0_matrix_unit": representative_matrix.get("actual_p0_matrix_unit", "rows"),
            "inferred_p1_row_matrix": representative_matrix.get("inferred_p1_row_matrix", []),
            "p1_is_exact_transpose": bool(candidate_evidence["p1_exact_transpose"]),
            "prediction_extra_collective_count": rank0_summary.get("prediction_extra_collective_count", 0),
            "p1_planning_collective_count": rank0_summary.get("p1_planning_collective_count", 0),
            "async_executor_invocation_count": rank0_summary.get("async_executor_invocation_count", 0),
            "batch_isend_irecv_call_count": rank0_summary.get("batch_isend_irecv_call_count", 0),
            "real_send_op_count": rank0_summary.get("real_send_op_count", 0),
            "real_recv_op_count": rank0_summary.get("real_recv_op_count", 0),
            "local_copy_task_count": rank0_summary.get("local_copy_task_count", 0),
            "before_async_p2p_phase_count": rank0_summary.get("before_async_p2p_phase_count", 0),
            "after_async_p2p_phase_count": rank0_summary.get("after_async_p2p_phase_count", 0),
            "phase_sync_fallback_count": phase_sync_fallback_count,
        }
    )
    payload.update(
        {
            "status": "executed",
            "reference_command": reference_cmd,
            "candidate_command": candidate_cmd,
            "actual_p0_global_matrix_available": bool(payload["actual_p0_full_row_matrix"]),
            "prediction_source_layer": latest_plan_event.get("source_layer_id", ""),
            "prediction_target_layer": latest_plan_event.get("target_layer_id", ""),
            "prediction_confidence": latest_plan_event.get("prediction_confidence", 0.0),
            "prediction_created_stage": "after_p0_observation",
            "prediction_first_consumed_stage": "during_p0_joint_planning" if candidate_evidence["prediction_consumed_during_p0_joint_planning"] else "",
            "consumer_layer": latest_plan_event.get("layer_id", ""),
            "consumer_phase": "P1" if candidate_evidence["prediction_consumed_during_p0_joint_planning"] else "",
            "prediction_audit": candidate_evidence["prediction_audits"][0] if candidate_evidence["prediction_audits"] else {},
            "raw_u_makespan": latest_plan_event.get("ideal_raw_u_makespan", rank0_summary.get("ideal_raw_u_makespan")),
            "paired_b_makespan": latest_plan_event.get("ideal_paired_b_makespan", rank0_summary.get("ideal_paired_b_makespan")),
            "host_projected_raw_u_makespan": latest_plan_event.get("host_projected_raw_u_makespan", rank0_summary.get("host_projected_raw_u_makespan")),
            "host_projected_paired_b_makespan": latest_plan_event.get("host_projected_paired_b_makespan", rank0_summary.get("host_projected_paired_b_makespan")),
            "safe_selected_policy": latest_plan_event.get("safe_selected_policy", ""),
            "p0_summary_gather_count": rank0_summary.get("p0_traffic_matrix_gather_count", 0),
            "fallback_count": phase_sync_fallback_count,
            "forward_epochs_tested": int(candidate_evidence["forward_epochs_tested"]),
            "digest_pairs": digest_pairs,
            "matrix_records": matrix_records,
            "plan_store_event_count": len(plan_store_events),
            "prediction_consumed_during_p0_joint_planning": bool(candidate_evidence["prediction_consumed_during_p0_joint_planning"]),
            "dispatcher_consistency": bool(candidate_evidence["dispatcher_consistency"]),
            "zero_hint_joint_plan_built": bool(candidate_evidence["zero_hint_joint_plan_built"]),
        }
    )
    checks = {
        "actual_p0_total_rows_nonzero": bool(candidate_evidence["actual_p0_total_rows_nonzero"]),
        "p1_exact_transpose": bool(payload["p1_is_exact_transpose"]),
        "stored_equals_consumed": bool(candidate_evidence["stored_equals_consumed"]),
        "p0_traffic_matrix_gather_once": int(payload["p0_summary_gather_count"]) == 1,
        "prediction_extra_collective_zero": int(payload["prediction_extra_collective_count"]) == 0,
        "p1_planning_collective_zero": int(payload["p1_planning_collective_count"]) == 0,
        "async_executor_invoked": int(payload["async_executor_invocation_count"]) > 0,
        "p2p_called": int(payload["batch_isend_irecv_call_count"]) > 0,
        "real_send_recv_present": int(payload["real_send_op_count"]) > 0 and int(payload["real_recv_op_count"]) > 0,
        "no_fallback": int(payload["phase_sync_fallback_count"]) == 0,
        "two_forward_epochs": int(payload["forward_epochs_tested"]) >= 2,
        "dispatcher_matches_local_rows": bool(candidate_evidence["dispatcher_consistency"]),
        "candidate_status_ready": str(candidate_summary_payload.get("status", "")) == "ready",
    }
    payload["checks"] = checks
    payload["status"] = "passed" if all(bool(value) for value in checks.values()) else "failed"
    write_json(output_dir / "b2_runner_summary.json", payload)
    print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
