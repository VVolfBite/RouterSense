#!/usr/bin/env python3
"""CPU-only probe for PreparedWindowPlan to online phase-plan compilation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json, write_jsonl
from rs.runtime.online.megatron_ep.execution.audit import ExecutionAuditInput, build_execution_audit
from rs.runtime.online.megatron_ep.p2_contracts import P2HintRequest
from rs.runtime.online.megatron_ep.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    FutureDemandHint,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    PhaseReadyContext,
    RuntimeIdentity,
    build_phase_ready_context,
)
from rs.runtime.online.megatron_ep.pending_window import compile_prepared_window_phase_plan
from rs.scheduling.contracts import (
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.validation import stable_hash


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="tests/fixtures/scheduling/p2_lookahead_sensitive_4rank.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="prepared_plan_trace_probe")
    parser.add_argument("--phase", choices=("P0", "P1"), default="P0")
    parser.add_argument("--bucket-rows", type=int, default=0)
    parser.add_argument("--p0-weight", type=float, default=1.0)
    parser.add_argument("--p1-reservation-weight", type=float, default=1.0)
    parser.add_argument("--p2-hint-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def _matrix(payload: dict[str, Any], key: str) -> tuple[tuple[int, ...], ...]:
    if key in payload:
        return tuple(tuple(int(value) for value in row) for row in payload[key])
    if key == "p2_next_dispatch_forecast_matrix" and "p2_next_dispatch_matrix" in payload:
        return tuple(tuple(int(value) for value in row) for row in payload["p2_next_dispatch_matrix"])
    raise KeyError(key)


def _build_problem(payload: dict[str, Any], *, p0_weight: float, p1_weight: float, p2_weight: float) -> MultiPhaseSchedulingProblem:
    p0 = _matrix(payload, "p0_dispatch_matrix")
    p1 = _matrix(payload, "p1_return_matrix")
    p2 = _matrix(payload, "p2_next_dispatch_forecast_matrix")
    digest = stable_hash({"p2_next_dispatch_forecast_matrix": p2})
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(),
        topology=LogicalTopology(num_gpus=int(payload.get("num_gpus", len(p0)))),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(payload.get("expert_compute_delay", 0.0)),
        ),
        forecast=ForecastPressure(
            source="trace_fixture",
            digest=digest,
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(len(p2), len(p2[0]) if p2 else 0),
            matrix_total_bytes=sum(sum(row) for row in p2),
            matrix=p2,
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_weight,
            p2_hint_weight=p2_weight,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def _contexts_from_matrix(*, phase: str, matrix: tuple[tuple[int, ...], ...], hint: FutureDemandHint) -> tuple[PhaseReadyContext, ...]:
    ep_group_ranks = tuple(range(len(matrix)))
    contexts: list[PhaseReadyContext] = []
    for rank, row in enumerate(matrix):
        col = tuple(int(matrix[src][rank]) for src in range(len(matrix)))
        if phase == "P0":
            input_splits = tuple(int(value) for value in row)
            output_splits = col
        else:
            input_splits = col
            output_splits = tuple(int(value) for value in row)
        contexts.append(
            _build_context(
                rank=rank,
                phase=phase,
                input_splits=input_splits,
                output_splits=output_splits,
                ep_group_ranks=ep_group_ranks,
                hint=hint,
            )
        )
    return tuple(contexts)


def _build_context(
    *,
    rank: int,
    phase: str,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    ep_group_ranks: tuple[int, ...],
    hint: FutureDemandHint,
) -> PhaseReadyContext:
    send_rows = sum(input_splits) if phase == "P0" else sum(output_splits)
    hidden_dim = 4
    hidden = torch.arange(max(send_rows, 1) * hidden_dim, dtype=torch.float16).reshape(max(send_rows, 1), hidden_dim)[:send_rows]
    packed_tensors = (hidden, hidden[:, :1].clone()) if phase == "P0" else (hidden,)
    return build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "1", "phase": phase, "rank": rank},
            runtime_identity=RuntimeIdentity(
                run_id="prepared-plan-probe",
                forward_epoch=0,
                layer_id="1",
                layer_name="module.decoder.layers.1.mlp",
                global_rank=rank,
                local_rank=rank,
                ep_group_ranks=ep_group_ranks,
                ep_group_root_rank=ep_group_ranks[0],
            ),
            topology={
                "global_rank": rank,
                "local_rank": rank,
                "node_index": 0,
                "hostname_digest": "cpu-probe",
                "device_index": rank,
                "ep_group_rank": ep_group_ranks.index(rank),
            },
            dispatcher_snapshot=DispatcherSnapshot(
                dispatcher_class="CPUProbeDispatcher",
                dispatcher_fingerprint={"dispatcher_class": "CPUProbeDispatcher"},
                expert_placement_hash="trace-fixture-placement",
                input_splits=input_splits,
                output_splits=output_splits,
            ),
            payload_contract=PhasePayloadContract(
                phase=phase,
                payload_roles=("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",),
                atomic_submit=phase == "P0",
            ),
            packed_tensors=packed_tensors,
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="trace_fixture_ready",
            payload_exists=True,
            p2_hint=hint,
        )
    )


def _synthetic_transport_events(plan: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for wave in plan.get("waves", []) or []:
        for task in wave.get("bucket_tasks", []) or []:
            payload_slices = task.get("payload_slices", []) or []
            if payload_slices:
                for payload in payload_slices:
                    rows.append(
                        {
                            "wave_id": int(wave["wave_id"]),
                            "task_id": task["task_id"],
                            "src_rank": int(task["src_rank"]),
                            "dst_rank": int(task["dst_rank"]),
                            "row_count": int(task["row_count"]),
                            "byte_count": int(payload["payload_byte_count"]),
                            "tensor_role": str(payload["tensor_role"]),
                            "plan_hash": plan.get("plan_hash", ""),
                            "phase": plan.get("phase", ""),
                            "layer_id": str(plan.get("plan_key", {}).get("layer_id", "")),
                        }
                    )
            else:
                rows.append(
                    {
                        "wave_id": int(wave["wave_id"]),
                        "task_id": task["task_id"],
                        "src_rank": int(task["src_rank"]),
                        "dst_rank": int(task["dst_rank"]),
                        "row_count": int(task["row_count"]),
                        "byte_count": int(task["byte_count"]),
                        "tensor_role": "hidden_states",
                        "plan_hash": plan.get("plan_hash", ""),
                        "phase": plan.get("phase", ""),
                        "layer_id": str(plan.get("plan_key", {}).get("layer_id", "")),
                    }
                )
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_path = Path(args.fixture)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    problem = _build_problem(
        payload,
        p0_weight=args.p0_weight,
        p1_weight=args.p1_reservation_weight,
        p2_weight=args.p2_hint_weight,
    )
    policy = RouterSenseMultiphaseLookaheadPolicy(
        information_mode="p0_p1_p2",
        p0_weight=args.p0_weight,
        p1_reservation_weight=args.p1_reservation_weight,
        p2_hint_weight=args.p2_hint_weight,
    )
    prepared = policy.build_prepared_window_plan(problem=problem, created_at_layer_id="0", applies_from_layer_id="1")
    shared_state = {
        "prepared_plan": prepared,
        "plan_created_at_us": int(time.time() * 1e6),
        "plan_source_layer": "module.decoder.layers.0.mlp",
    }
    hint = build_p2_hint_provider("calibrated_artifact", shared_state=shared_state).build_hint(
        P2HintRequest(
            plan_key={"layer_id": "1", "phase": args.phase},
            layer_id="1",
            phase=args.phase,
            global_rank=0,
            local_rank=0,
            ep_group_ranks=tuple(range(int(payload.get("num_gpus", len(problem.p0_dispatch_matrix))))),
        )
    )
    phase_matrix = problem.p0_dispatch_matrix if args.phase == "P0" else problem.p1_return_matrix
    contexts = _contexts_from_matrix(phase=args.phase, matrix=phase_matrix, hint=hint)
    execution_plan = compile_prepared_window_phase_plan(
        prepared_plan=prepared,
        local_context=contexts[0],
        global_contexts=contexts,
        bucket_rows=args.bucket_rows,
        p0_weight=args.p0_weight,
        p1_reservation_weight=args.p1_reservation_weight,
        p2_hint_weight=args.p2_hint_weight,
    )
    plan_dict = execution_plan.to_dict()
    transport_events = _synthetic_transport_events(plan_dict)
    audit = build_execution_audit(
        ExecutionAuditInput(
            execution_plan=execution_plan,
            transport_events=transport_events,
            phase_contract={"phase": args.phase, "layer_id": "1", "policy_enabled": True},
        )
    )
    arrival_record = {
        "layer_name": "module.decoder.layers.1.mlp",
        "phase": args.phase,
        "arrival_status": "before_commit",
        "source_layer": shared_state["plan_source_layer"],
        "has_prepared_plan": True,
        "window_key": prepared.window_key,
        "forecast_digest": prepared.forecast_digest,
    }

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": args.run_id,
            "run_kind": "prepared_plan_trace_probe",
            "fixture": str(fixture_path),
            "phase": args.phase,
            "bucket_rows": args.bucket_rows,
            "gpu_required": False,
            "megatron_required": False,
        },
    )
    write_json(run_dir / "prepared_window_plan.json", prepared.to_dict())
    write_json(run_dir / "future_demand_hint.json", hint.to_dict())
    write_json(run_dir / "phase_execution_plan.json", plan_dict)
    write_jsonl(run_dir / "synthetic_transport_execution.jsonl", transport_events)
    write_jsonl(run_dir / "plan_arrival_records.jsonl", [arrival_record])
    write_json(run_dir / "synthetic_execution_audit.json", audit.to_dict())
    write_json(
        run_dir / "probe_summary.json",
        {
            "compiled_from_prepared_plan": bool(execution_plan.metrics.get("compiled_from_prepared_plan", False)),
            "prepared_plan_order_preserved": bool(execution_plan.metrics.get("prepared_plan_order_preserved", False)),
            "hint_edges_available": int(execution_plan.metrics.get("hint_edges_available", 0) or 0),
            "hint_edges_consumed": int(execution_plan.metrics.get("hint_edges_consumed", 0) or 0),
            "hint_match_rate": float(execution_plan.metrics.get("hint_match_rate", 0.0) or 0.0),
            "audit_status": audit.status,
            "phase_wave_count": len(execution_plan.waves),
            "phase_task_count": sum(len(wave.bucket_tasks) for wave in execution_plan.waves),
        },
    )
    return 0 if audit.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
