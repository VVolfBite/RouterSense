from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any

from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave

from ..capabilities import PolicyCapabilities
from .common import (
    build_phase_serial_release_aware_plan,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
)


class ISLIPNoProgressError(RuntimeError):
    pass


class ISLIPRoundRobinPolicy:
    policy_name = "islip_round_robin"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int, max_rounds: int = 2, pointer_seed: str = "") -> None:
        self.bucket_rows = int(bucket_rows)
        self.max_rounds = int(max_rounds)
        self.pointer_seed = str(pointer_seed)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        p0_flows = list(flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True))
        p1_flows = list(flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True))
        p0_waves, p0_trace = _schedule_flows(
            p0_flows,
            ranks=tuple(range(problem.topology.num_gpus)),
            seed_payload={"policy": self.policy_name, "phase": "p0_dispatch", "matrix": problem.p0_dispatch_matrix, "seed": self.pointer_seed},
            start_wave_id=0,
            max_rounds=self.max_rounds,
        )
        p1_waves, p1_trace = _schedule_flows(
            p1_flows,
            ranks=tuple(range(problem.topology.num_gpus)),
            seed_payload={"policy": self.policy_name, "phase": "p1_return", "matrix": problem.p1_return_matrix, "seed": self.pointer_seed},
            start_wave_id=len(p0_waves),
            max_rounds=self.max_rounds,
        )
        p2_flows: list[FlowDemand] = []
        p2_waves: list[LogicalWave] = []
        p2_trace: list[dict[str, Any]] = []
        if include_real_p2_phase(problem):
            p2_flows = list(
                flows_from_matrix(
                    problem.p2_next_dispatch_forecast_matrix,
                    phase="p2_next_dispatch",
                    release_state="ready",
                    executable=True,
                )
            )
            p2_waves, p2_trace = _schedule_flows(
                p2_flows,
                ranks=tuple(range(problem.topology.num_gpus)),
                seed_payload={"policy": self.policy_name, "phase": "p2_next_dispatch", "matrix": problem.p2_next_dispatch_forecast_matrix, "seed": self.pointer_seed},
                start_wave_id=len(p0_waves) + len(p1_waves),
                max_rounds=self.max_rounds,
            )
        all_trace = p0_trace + p1_trace + p2_trace
        fallback_reason = ";".join(
            str(item.get("fallback_reason", ""))
            for item in all_trace
            if item.get("fallback_used")
        )
        base_plan = build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_islip_round_robin",
            tie_break_rule="stable rotating input/output pointers",
            priority_components={"islip_rounds": self.max_rounds},
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_islip_round_robin_v1",
            fallback_reason=fallback_reason,
        )
        return LogicalSchedulePlan(
            policy_name=base_plan.policy_name,
            waves=base_plan.waves,
            diagnostics={
                **base_plan.diagnostics,
                "islip_rounds": self.max_rounds,
                "islip_trace": all_trace,
                "p2_executable_flow_count": len(p2_flows),
            },
        )

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        transfer_layouts, all_tasks, build_stats = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
            return_stats=True,
        )
        schedule_started_ns = time.perf_counter_ns()
        waves, trace = _schedule_tasks(
            all_tasks,
            ranks=tuple(int(rank) for rank in local_context.ep_group_ranks),
            seed_payload={"policy": self.policy_name, "plan_key": local_context.plan_key, "phase": local_context.phase, "seed": self.pointer_seed},
            phase=local_context.phase,
            max_rounds=self.max_rounds,
        )
        pack_time_us = (time.perf_counter_ns() - schedule_started_ns) / 1000.0
        ordered_tasks = [task for wave in waves for task in wave.bucket_tasks]
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "phase_local_islip_round_robin",
            "logical_model": "discrete_bucket_phase_sync_wave",
            "bucket_order": [task.task_id for task in ordered_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "p1_dependency_used": False,
            "p2_forecast_used": False,
            "p2_source": local_context.p2_hint.hint_source,
            "evaluation_eligible": True,
            "islip_rounds": self.max_rounds,
            "islip_trace": trace,
            "priority_components": {"selection_rule": "deterministic iSLIP-style request/grant/accept"},
            "tie_break_rule": "stable rotating input/output pointers",
            "fallback_reason": ";".join(str(item.get("fallback_reason", "")) for item in trace if item.get("fallback_used")),
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=ordered_tasks,
            waves=tuple(waves),
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": int(len(waves)),
                "max_wave_task_count": int(max((len(wave.bucket_tasks) for wave in waves), default=0)),
                "task_count": int(len(all_tasks)),
            },
        )


def _stable_pointer(payload: Any, modulo: int, *, salt: str) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(json.dumps({"payload": payload, "salt": salt}, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _rotate(items: tuple[int, ...], pointer: int) -> tuple[int, ...]:
    if not items:
        return ()
    pointer %= len(items)
    return items[pointer:] + items[:pointer]


def _task_key(item: Any) -> tuple[int, int, int, str]:
    return (
        int(item.src_rank),
        int(item.dst_rank),
        int(getattr(item, "bucket_ordinal", 0)),
        str(getattr(item, "task_id", getattr(item, "flow_id", ""))),
    )


def _schedule_flows(
    flows: list[FlowDemand],
    *,
    ranks: tuple[int, ...],
    seed_payload: Any,
    start_wave_id: int,
    max_rounds: int,
) -> tuple[list[LogicalWave], list[dict[str, Any]]]:
    pending = sorted(flows, key=_task_key)
    waves: list[LogicalWave] = []
    traces: list[dict[str, Any]] = []
    wave_id = int(start_wave_id)
    while pending:
        chosen, trace = _select_matching(pending, ranks=ranks, seed_payload={**seed_payload, "wave_id": wave_id}, max_rounds=max_rounds)
        if not chosen:
            raise ISLIPNoProgressError("islip_round_robin made no progress with residual logical flows")
        chosen_ids = {flow.flow_id for flow in chosen}
        pending = [flow for flow in pending if flow.flow_id not in chosen_ids]
        waves.append(LogicalWave(wave_id=wave_id, flows=tuple(chosen), duration=float(max(int(flow.byte_count) for flow in chosen))))
        traces.append(trace)
        wave_id += 1
    return waves, traces


def _schedule_tasks(
    tasks: list[BucketTask],
    *,
    ranks: tuple[int, ...],
    seed_payload: Any,
    phase: str,
    max_rounds: int,
) -> tuple[list[PlanWave], list[dict[str, Any]]]:
    pending = sorted(tasks, key=_task_key)
    waves: list[PlanWave] = []
    traces: list[dict[str, Any]] = []
    wave_id = 0
    while pending:
        chosen, trace = _select_matching(pending, ranks=ranks, seed_payload={**seed_payload, "wave_id": wave_id}, max_rounds=max_rounds)
        if not chosen:
            raise ISLIPNoProgressError("islip_round_robin made no progress with residual bucket tasks")
        chosen_ids = {task.task_id for task in chosen}
        pending = [task for task in pending if task.task_id not in chosen_ids]
        waves.append(PlanWave(wave_id=wave_id, phase=phase, bucket_tasks=tuple(chosen)))
        traces.append(trace)
        wave_id += 1
    return waves, traces


def _select_matching(items: list[Any], *, ranks: tuple[int, ...], seed_payload: Any, max_rounds: int) -> tuple[list[Any], dict[str, Any]]:
    if not ranks and items:
        return [], {
            "wave_id": seed_payload.get("wave_id", 0) if isinstance(seed_payload, dict) else 0,
            "islip_rounds": int(max_rounds),
            "input_pointer": {},
            "output_pointer": {},
            "grant_trace": [],
            "accept_trace": [],
            "selected_edges": [],
            "fallback_used": False,
            "fallback_reason": "no_rank_ports_available",
        }
    by_src_dst: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for item in sorted(items, key=_task_key):
        by_src_dst[(int(item.src_rank), int(item.dst_rank))].append(item)
    rank_order = tuple(sorted(int(rank) for rank in ranks))
    input_pointer = {rank: _stable_pointer(seed_payload, len(rank_order), salt=f"input:{rank}") for rank in rank_order}
    output_pointer = {rank: _stable_pointer(seed_payload, len(rank_order), salt=f"output:{rank}") for rank in rank_order}
    grants: list[dict[str, Any]] = []
    accepts: list[dict[str, Any]] = []
    selected_edges: list[tuple[int, int]] = []
    selected_src: set[int] = set()
    selected_dst: set[int] = set()
    for round_id in range(max(1, int(max_rounds))):
        requests_by_dst: dict[int, list[int]] = defaultdict(list)
        for (src, dst), queue in by_src_dst.items():
            if queue and src not in selected_src and dst not in selected_dst:
                requests_by_dst[dst].append(src)
        granted_by_src: dict[int, list[int]] = defaultdict(list)
        for dst, srcs in sorted(requests_by_dst.items()):
            ordered_srcs = _rotate(rank_order, output_pointer.get(dst, 0))
            candidates = [src for src in ordered_srcs if src in srcs and src not in selected_src]
            if not candidates:
                continue
            src = candidates[0]
            grants.append({"round": round_id, "dst_rank": dst, "granted_src_rank": src, "requests": sorted(srcs), "output_pointer": output_pointer.get(dst, 0)})
            granted_by_src[src].append(dst)
        progress = False
        for src, dsts in sorted(granted_by_src.items()):
            ordered_dsts = _rotate(rank_order, input_pointer.get(src, 0))
            candidates = [dst for dst in ordered_dsts if dst in dsts and dst not in selected_dst]
            if not candidates:
                continue
            dst = candidates[0]
            if src in selected_src or dst in selected_dst:
                continue
            selected_edges.append((src, dst))
            selected_src.add(src)
            selected_dst.add(dst)
            input_pointer[src] = (rank_order.index(dst) + 1) % len(rank_order)
            output_pointer[dst] = (rank_order.index(src) + 1) % len(rank_order)
            accepts.append({"round": round_id, "src_rank": src, "accepted_dst_rank": dst, "grants": sorted(dsts), "input_pointer": input_pointer[src]})
            progress = True
        if not progress:
            break
    fallback_used = False
    fallback_reason = ""
    if not selected_edges and by_src_dst:
        selected_edges = _deterministic_maximal_edges(by_src_dst, rank_order)
        fallback_used = bool(selected_edges)
        fallback_reason = "no_progress_after_islip_rounds"
    selected = [by_src_dst[edge][0] for edge in selected_edges]
    trace = {
        "wave_id": seed_payload.get("wave_id", 0) if isinstance(seed_payload, dict) else 0,
        "islip_rounds": int(max_rounds),
        "input_pointer": dict(sorted(input_pointer.items())),
        "output_pointer": dict(sorted(output_pointer.items())),
        "grant_trace": grants,
        "accept_trace": accepts,
        "selected_edges": [{"src_rank": src, "dst_rank": dst} for src, dst in selected_edges],
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }
    return selected, trace


def _deterministic_maximal_edges(by_src_dst: dict[tuple[int, int], list[Any]], rank_order: tuple[int, ...]) -> list[tuple[int, int]]:
    del rank_order
    used_src: set[int] = set()
    used_dst: set[int] = set()
    edges: list[tuple[int, int]] = []
    for src, dst in sorted(by_src_dst):
        if not by_src_dst[(src, dst)]:
            continue
        if src in used_src or dst in used_dst:
            continue
        edges.append((src, dst))
        used_src.add(src)
        used_dst.add(dst)
    return edges


def _wave_diag(wave: LogicalWave, trace: dict[str, Any]) -> WaveDiagnostics:
    return WaveDiagnostics(
        wave_id=int(wave.wave_id),
        selected_flow_ids=tuple(flow.flow_id for flow in wave.flows),
        selected_edges=tuple({"phase": flow.phase, "src_rank": flow.src_rank, "dst_rank": flow.dst_rank, "byte_count": flow.byte_count} for flow in wave.flows),
        matching_weight=float(sum(int(flow.byte_count) for flow in wave.flows)),
        priority_components={"islip_trace": trace},
        remaining_bytes_before=float(sum(int(flow.byte_count) for flow in wave.flows)),
        remaining_bytes_after=0.0,
        ready_flow_count_before=len(wave.flows),
        blocked_flow_count_before=0,
        forecast_pressure_summary={},
        selection_reason="deterministic iSLIP-style request/grant/accept",
    )


__all__ = ["ISLIPNoProgressError", "ISLIPRoundRobinPolicy"]
