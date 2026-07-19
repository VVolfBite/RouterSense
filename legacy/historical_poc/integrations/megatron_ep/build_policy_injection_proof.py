#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.routersense.phase import (
    FutureDemandHint,
    IncomingSlot,
    OutgoingSegment,
    PackedTensorDescriptor,
    PayloadSlice,
    PhaseReadyContext,
    TransportBundle,
)
from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy
from integrations.megatron_ep.routersense.policy.validation import stable_hash
from integrations.megatron_ep.routersense.trace_writer import write_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _decode_context(row: dict[str, Any]) -> PhaseReadyContext:
    return PhaseReadyContext(
        plan_key=dict(row["plan_key"]),
        phase=str(row["phase"]),
        control_mode=str(row["control_mode"]),
        forward_epoch=int(row["forward_epoch"]),
        layer_id=str(row["layer_id"]),
        layer_name=str(row["layer_name"]),
        global_rank=int(row["global_rank"]),
        local_rank=int(row["local_rank"]),
        ep_group_ranks=tuple(int(v) for v in row["ep_group_ranks"]),
        ep_group_root_rank=int(row["ep_group_root_rank"]),
        topology=dict(row["topology"]),
        dispatcher_class=str(row["dispatcher_class"]),
        dispatcher_fingerprint=dict(row["dispatcher_fingerprint"]),
        expert_placement_hash=str(row["expert_placement_hash"]),
        input_splits=tuple(int(v) for v in row["input_splits"]),
        output_splits=tuple(int(v) for v in row["output_splits"]),
        send_splits=tuple(int(v) for v in row["send_splits"]),
        recv_splits=tuple(int(v) for v in row["recv_splits"]),
        per_peer_rows=tuple(int(v) for v in row["per_peer_rows"]),
        per_peer_bytes=tuple(int(v) for v in row["per_peer_bytes"]),
        packed_send_layout_id=str(row["packed_send_layout_id"]),
        canonical_receive_layout_id=str(row["canonical_receive_layout_id"]),
        outgoing_segments=tuple(OutgoingSegment(**segment) for segment in row["outgoing_segments"]),
        incoming_slots=tuple(IncomingSlot(**slot) for slot in row["incoming_slots"]),
        transport_bundles=tuple(
            TransportBundle(
                bundle_id=str(bundle["bundle_id"]),
                phase=str(bundle["phase"]),
                atomic_submit=bool(bundle["atomic_submit"]),
                outgoing_segment=OutgoingSegment(**bundle["outgoing_segment"]),
                payloads=tuple(PackedTensorDescriptor(**payload) for payload in bundle["payloads"]),
                payload_slices=tuple(PayloadSlice(**payload) for payload in bundle["payload_slices"]),
            )
            for bundle in row["transport_bundles"]
        ),
        release_state=str(row["release_state"]),
        demand_known_at=str(row["demand_known_at"]),
        payload_exists=bool(row["payload_exists"]),
        p2_hint=FutureDemandHint(**row.get("p2_hint", {})),
    )


def _expected_execution_order_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    execution_ordinal = 0
    for wave in plan.get("waves", []):
        phase = str(wave["phase"])
        tensor_roles = ["hidden_states", "routing_probs"] if phase == "P0" else ["hidden_states"]
        for tensor_role in tensor_roles:
            for task in wave.get("bucket_tasks", []):
                execution_ordinal += 1
                rows.append(
                    {
                        "layer_id": str(plan["plan_key"]["layer_id"]),
                        "phase": phase,
                        "wave_id": int(wave["wave_id"]),
                        "bucket_id": str(task["task_id"]),
                        "src_rank": int(task["src_rank"]),
                        "dst_rank": int(task["dst_rank"]),
                        "sender_offset_rows": int(task["sender_offset_rows"]),
                        "receiver_offset_rows": int(task["receiver_offset_rows"]),
                        "row_count": int(task["row_count"]),
                        "tensor_role": tensor_role,
                        "execution_ordinal": execution_ordinal,
                    }
                )
    return rows


def _order_signature(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            str(row["layer_id"]),
            str(row["phase"]),
            int(row["wave_id"]),
            str(row["bucket_id"]),
            int(row["src_rank"]),
            int(row["dst_rank"]),
            str(row["tensor_role"]),
        )
        for row in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--baseline-policy", default="bucketed_fifo")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    candidate_run_dir = Path(args.candidate_run_dir)
    output_path = Path(args.output) if args.output else candidate_run_dir / "policy_injection_proof.json"

    summary = _load_json(candidate_run_dir / "summary.json")
    bucket_rows = int(summary["details"]["bucket_rows"])
    candidate_policy_name = str(summary["details"].get("policy_name", summary["details"].get("scheduler_mode", "disabled")))

    phase_context_rows: list[dict[str, Any]] = []
    for path in sorted(candidate_run_dir.glob("rank*_phase_contexts.jsonl")):
        phase_context_rows.extend(_load_jsonl(path))
    contexts = [_decode_context(row) for row in phase_context_rows]

    candidate_plans = _load_jsonl(candidate_run_dir / "rank0_scheduled_phase_plans.jsonl")
    actual_execution = [
        row
        for row in _load_jsonl(candidate_run_dir / "rank0_transport_execution.jsonl")
        if row.get("record_type") != "result_summary"
    ]

    baseline_execution_rows: list[dict[str, Any]] = []
    candidate_execution_rows: list[dict[str, Any]] = []
    baseline_plan_hashes: list[str] = []
    candidate_plan_hashes: list[str] = []
    remote_bucket_count_per_phase: dict[str, int] = {}
    fifo_wave_count = 0
    candidate_wave_count = 0
    p0_hidden_bundle_order_identical = True
    p0_probs_bundle_order_identical = True

    for plan_row in candidate_plans:
        layer_id = str(plan_row["plan_key"]["layer_id"])
        phase = str(plan_row["phase"])
        plan_contexts = tuple(
            ctx for ctx in contexts if str(ctx.layer_id) == layer_id and str(ctx.phase) == phase
        )
        if not plan_contexts:
            raise ValueError(f"missing phase contexts for layer={layer_id} phase={phase}")
        local_context = next(ctx for ctx in plan_contexts if int(ctx.global_rank) == int(plan_row["root_rank"]))
        baseline_policy = resolve_phase_policy(policy_name=args.baseline_policy, bucket_rows=bucket_rows)
        baseline_plan = baseline_policy.build_plan(local_context=local_context, global_contexts=plan_contexts)
        baseline_plan_hashes.append(baseline_plan.plan_hash)
        candidate_plan_hashes.append(str(plan_row["plan_hash"]))
        fifo_rows = _expected_execution_order_from_plan(baseline_plan.to_dict())
        cand_rows = [
            row
            for row in actual_execution
            if str(row["layer_id"]) == layer_id and str(row["phase"]) == phase
        ]
        baseline_execution_rows.extend(fifo_rows)
        candidate_execution_rows.extend(sorted(cand_rows, key=lambda item: int(item["execution_ordinal"])))
        bucket_ids = {str(task["task_id"]) for wave in plan_row.get("waves", []) for task in wave.get("bucket_tasks", [])}
        remote_bucket_count_per_phase[f"{layer_id}:{phase}"] = len(bucket_ids)
        fifo_wave_count += len(baseline_plan.waves)
        candidate_wave_count += len(plan_row.get("waves", []))
        if phase == "P0":
            expected_bucket_order = [str(row["bucket_id"]) for row in _expected_execution_order_from_plan(plan_row) if row["tensor_role"] == "hidden_states"]
            hidden_order = [str(row["bucket_id"]) for row in cand_rows if row["tensor_role"] == "hidden_states"]
            probs_order = [str(row["bucket_id"]) for row in cand_rows if row["tensor_role"] == "routing_probs"]
            p0_hidden_bundle_order_identical = p0_hidden_bundle_order_identical and (hidden_order == expected_bucket_order)
            p0_probs_bundle_order_identical = p0_probs_bundle_order_identical and (probs_order == expected_bucket_order)

    fifo_execution_order = _order_signature(baseline_execution_rows)
    candidate_execution_order = _order_signature(sorted(candidate_execution_rows, key=lambda item: int(item["execution_ordinal"])))
    execution_order_differs = fifo_execution_order != candidate_execution_order
    remote_bucket_count = sum(remote_bucket_count_per_phase.values())
    status = "ready"
    if remote_bucket_count < 2 or not execution_order_differs:
        status = "insufficient_schedule_space"

    payload = {
        "status": status,
        "baseline_policy": args.baseline_policy,
        "candidate_policy": candidate_policy_name,
        "fifo_plan_hash": stable_hash(baseline_plan_hashes),
        "candidate_plan_hash": stable_hash(candidate_plan_hashes),
        "fifo_wave_count": fifo_wave_count,
        "candidate_wave_count": candidate_wave_count,
        "fifo_execution_order": fifo_execution_order,
        "candidate_execution_order": candidate_execution_order,
        "execution_order_differs": execution_order_differs,
        "remote_bucket_count": remote_bucket_count,
        "remote_bucket_count_per_phase": remote_bucket_count_per_phase,
        "p0_hidden_bundle_order_identical": p0_hidden_bundle_order_identical,
        "p0_probs_bundle_order_identical": p0_probs_bundle_order_identical,
        "transport_mutation": True,
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
