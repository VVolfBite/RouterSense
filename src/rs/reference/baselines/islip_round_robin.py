"""Persistent-state iSLIP-style offline reference.

The core follows request/grant/accept arbitration, keeps input/output round-robin
pointers across scheduling slots, and updates pointers only for matches accepted
in the first iteration. Variable-size MoE edges are treated as queued cells at
the selected logical bucket granularity.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_local.common import (
    build_phase_serial_release_aware_plan,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
)


class ISLIPNoProgressError(RuntimeError):
    pass


@dataclass
class _PointerState:
    rank_order: tuple[int, ...]
    input_pointer: dict[int, int]
    output_pointer: dict[int, int]


class ISLIPRoundRobinPolicy:
    policy_name = "islip_reference"
    policy_version = "v2-persistent"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=False,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int, max_rounds: int = 0, pointer_seed: str = "") -> None:
        self.bucket_rows = int(bucket_rows)
        self.max_rounds = int(max_rounds)
        self.pointer_seed = str(pointer_seed)

    def _rounds(self, rank_count: int) -> int:
        if self.max_rounds > 0:
            return min(max(1, self.max_rounds), max(1, rank_count))
        return min(max(1, int(math.ceil(math.log2(max(2, rank_count))))), max(1, rank_count))

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        ranks = tuple(range(int(problem.topology.num_gpus)))
        state = _initial_pointer_state(
            ranks,
            {"policy": self.policy_name, "seed": self.pointer_seed, "num_gpus": len(ranks)},
        )
        rounds = self._rounds(len(ranks))
        p0_waves, p0_trace = _schedule_flows(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            ranks=ranks,
            seed_payload={"policy": self.policy_name, "phase": "p0_dispatch", "seed": self.pointer_seed},
            start_wave_id=0,
            max_rounds=rounds,
            pointer_state=state,
        )
        p1_waves, p1_trace = _schedule_flows(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            ranks=ranks,
            seed_payload={"policy": self.policy_name, "phase": "p1_return", "seed": self.pointer_seed},
            start_wave_id=len(p0_waves),
            max_rounds=rounds,
            pointer_state=state,
        )
        p2_waves: list[LogicalWave] = []
        p2_trace: list[dict[str, Any]] = []
        p2_flow_count = 0
        if include_real_p2_phase(problem):
            p2_flows = flows_from_matrix(
                problem.p2_next_dispatch_forecast_matrix,
                phase="p2_next_dispatch",
                release_state="ready",
                executable=True,
            )
            p2_flow_count = len(p2_flows)
            p2_waves, p2_trace = _schedule_flows(
                p2_flows,
                ranks=ranks,
                seed_payload={"policy": self.policy_name, "phase": "p2_next_dispatch", "seed": self.pointer_seed},
                start_wave_id=len(p0_waves) + len(p1_waves),
                max_rounds=rounds,
                pointer_state=state,
            )
        trace = p0_trace + p1_trace + p2_trace
        base = build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_islip_persistent",
            tie_break_rule="persistent rotating input/output pointers",
            priority_components={
                "selection_rule": "iSLIP request/grant/accept",
                "iterations_per_slot": rounds,
                "pointer_update": "first iteration accepted matches only",
            },
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_islip_persistent_v2",
            fallback_reason=";".join(
                str(item.get("fallback_reason", "")) for item in trace if item.get("fallback_used")
            ),
        )
        return LogicalSchedulePlan(
            policy_name=base.policy_name,
            waves=base.waves,
            diagnostics={
                **base.diagnostics,
                "literature_mapping": "style",
                "islip_rounds": rounds,
                "pointer_state_persistent_across_waves": True,
                "pointer_update_first_iteration_only": True,
                "islip_trace": trace,
                "p2_executable_flow_count": p2_flow_count,
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
        ranks = tuple(int(rank) for rank in local_context.ep_group_ranks)
        rounds = self._rounds(len(ranks))
        state = _initial_pointer_state(
            ranks,
            {"policy": self.policy_name, "plan_key": local_context.plan_key, "seed": self.pointer_seed},
        )
        schedule_started_ns = time.perf_counter_ns()
        waves, trace = _schedule_tasks(
            all_tasks,
            ranks=ranks,
            seed_payload={"policy": self.policy_name, "plan_key": local_context.plan_key, "phase": local_context.phase},
            phase=local_context.phase,
            max_rounds=rounds,
            pointer_state=state,
        )
        pack_time_us = (time.perf_counter_ns() - schedule_started_ns) / 1000.0
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "phase_local_islip_persistent",
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "evaluation_eligible": True,
            "islip_rounds": rounds,
            "pointer_state_persistent_across_waves": True,
            "pointer_update_first_iteration_only": True,
            "islip_trace": trace,
            "priority_components": {"selection_rule": "persistent iSLIP request/grant/accept"},
            "tie_break_rule": "persistent rotating input/output pointers",
            "fallback_reason": ";".join(
                str(item.get("fallback_reason", "")) for item in trace if item.get("fallback_used")
            ),
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=all_tasks,
            waves=tuple(waves),
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": len(waves),
                "task_count": len(all_tasks),
            },
        )


def _stable_pointer(payload: Any, modulo: int, *, salt: str) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(
        json.dumps({"payload": payload, "salt": salt}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % modulo


def _initial_pointer_state(ranks: tuple[int, ...], seed_payload: Any) -> _PointerState:
    rank_order = tuple(sorted(int(rank) for rank in ranks))
    return _PointerState(
        rank_order=rank_order,
        input_pointer={rank: _stable_pointer(seed_payload, len(rank_order), salt=f"input:{rank}") for rank in rank_order},
        output_pointer={rank: _stable_pointer(seed_payload, len(rank_order), salt=f"output:{rank}") for rank in rank_order},
    )


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
    flows: tuple[FlowDemand, ...] | list[FlowDemand],
    *,
    ranks: tuple[int, ...],
    seed_payload: Any,
    start_wave_id: int,
    max_rounds: int,
    pointer_state: _PointerState | None = None,
) -> tuple[list[LogicalWave], list[dict[str, Any]]]:
    state = pointer_state or _initial_pointer_state(ranks, seed_payload)
    pending = sorted(list(flows), key=_task_key)
    waves: list[LogicalWave] = []
    traces: list[dict[str, Any]] = []
    wave_id = int(start_wave_id)
    while pending:
        chosen, trace = _select_matching(
            pending,
            ranks=ranks,
            seed_payload={"wave_id": wave_id, "seed": seed_payload},
            max_rounds=max_rounds,
            pointer_state=state,
        )
        if not chosen:
            raise ISLIPNoProgressError("islip_round_robin made no progress with residual logical flows")
        chosen_ids = {str(flow.flow_id) for flow in chosen}
        pending = [flow for flow in pending if str(flow.flow_id) not in chosen_ids]
        waves.append(
            LogicalWave(
                wave_id=wave_id,
                flows=tuple(chosen),
                duration=float(max(int(flow.byte_count) for flow in chosen)),
            )
        )
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
    pointer_state: _PointerState | None = None,
) -> tuple[list[PlanWave], list[dict[str, Any]]]:
    state = pointer_state or _initial_pointer_state(ranks, seed_payload)
    pending = sorted(tasks, key=_task_key)
    waves: list[PlanWave] = []
    traces: list[dict[str, Any]] = []
    while pending:
        chosen, trace = _select_matching(
            pending,
            ranks=ranks,
            seed_payload={"wave_id": len(waves), "seed": seed_payload},
            max_rounds=max_rounds,
            pointer_state=state,
        )
        if not chosen:
            raise ISLIPNoProgressError("islip_round_robin made no progress with residual bucket tasks")
        chosen_ids = {str(task.task_id) for task in chosen}
        pending = [task for task in pending if str(task.task_id) not in chosen_ids]
        waves.append(PlanWave(wave_id=len(waves), phase=phase, bucket_tasks=tuple(chosen)))
        traces.append(trace)
    return waves, traces


def _select_matching(
    items: list[Any],
    *,
    ranks: tuple[int, ...],
    seed_payload: Any,
    max_rounds: int,
    pointer_state: _PointerState | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    state = pointer_state or _initial_pointer_state(ranks, seed_payload)
    rank_order = state.rank_order
    before_input = dict(state.input_pointer)
    before_output = dict(state.output_pointer)
    by_src_dst: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for item in sorted(items, key=_task_key):
        by_src_dst[(int(item.src_rank), int(item.dst_rank))].append(item)
    if not rank_order and items:
        return [], {
            "wave_id": seed_payload.get("wave_id", 0) if isinstance(seed_payload, dict) else 0,
            "islip_rounds": int(max_rounds),
            "input_pointer_before": before_input,
            "output_pointer_before": before_output,
            "input_pointer_after": dict(state.input_pointer),
            "output_pointer_after": dict(state.output_pointer),
            "grant_trace": [],
            "accept_trace": [],
            "selected_edges": [],
            "fallback_used": False,
            "fallback_reason": "no_rank_ports_available",
        }

    selected_edges: list[tuple[int, int]] = []
    selected_src: set[int] = set()
    selected_dst: set[int] = set()
    grants: list[dict[str, Any]] = []
    accepts: list[dict[str, Any]] = []
    rounds_executed = 0
    for round_id in range(max(1, int(max_rounds))):
        rounds_executed += 1
        requests_by_dst: dict[int, list[int]] = defaultdict(list)
        for (src, dst), queue in by_src_dst.items():
            if queue and src not in selected_src and dst not in selected_dst and src in rank_order and dst in rank_order:
                requests_by_dst[dst].append(src)
        granted_by_src: dict[int, list[int]] = defaultdict(list)
        grant_choice: dict[tuple[int, int], int] = {}
        for dst, srcs in sorted(requests_by_dst.items()):
            candidates = [
                src for src in _rotate(rank_order, state.output_pointer.get(dst, 0))
                if src in srcs and src not in selected_src
            ]
            if not candidates:
                continue
            src = candidates[0]
            granted_by_src[src].append(dst)
            grant_choice[(src, dst)] = state.output_pointer.get(dst, 0)
            grants.append(
                {
                    "round": round_id,
                    "dst_rank": dst,
                    "granted_src_rank": src,
                    "requests": sorted(srcs),
                    "output_pointer": state.output_pointer.get(dst, 0),
                }
            )
        progress = False
        for src, dsts in sorted(granted_by_src.items()):
            candidates = [
                dst for dst in _rotate(rank_order, state.input_pointer.get(src, 0))
                if dst in dsts and dst not in selected_dst
            ]
            if not candidates:
                continue
            dst = candidates[0]
            if src in selected_src or dst in selected_dst:
                continue
            selected_edges.append((src, dst))
            selected_src.add(src)
            selected_dst.add(dst)
            if round_id == 0:
                state.input_pointer[src] = (rank_order.index(dst) + 1) % len(rank_order)
                state.output_pointer[dst] = (rank_order.index(src) + 1) % len(rank_order)
            accepts.append(
                {
                    "round": round_id,
                    "src_rank": src,
                    "accepted_dst_rank": dst,
                    "grants": sorted(dsts),
                    "pointers_updated": round_id == 0,
                    "grant_pointer_seen": grant_choice.get((src, dst), 0),
                }
            )
            progress = True
        if not progress:
            break

    fallback_used = False
    fallback_reason = ""
    if not selected_edges and by_src_dst:
        selected_edges = _deterministic_maximal_edges(by_src_dst)
        fallback_used = bool(selected_edges)
        fallback_reason = "no_progress_after_islip_rounds"
    selected = [by_src_dst[edge][0] for edge in selected_edges]
    return selected, {
        "wave_id": seed_payload.get("wave_id", 0) if isinstance(seed_payload, dict) else 0,
        "islip_rounds": int(max_rounds),
        "rounds_executed": rounds_executed,
        "input_pointer_before": dict(sorted(before_input.items())),
        "output_pointer_before": dict(sorted(before_output.items())),
        "input_pointer_after": dict(sorted(state.input_pointer.items())),
        "output_pointer_after": dict(sorted(state.output_pointer.items())),
        # Compatibility keys retained for older report readers.
        "input_pointer": dict(sorted(state.input_pointer.items())),
        "output_pointer": dict(sorted(state.output_pointer.items())),
        "grant_trace": grants,
        "accept_trace": accepts,
        "selected_edges": [{"src_rank": src, "dst_rank": dst} for src, dst in selected_edges],
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _deterministic_maximal_edges(by_src_dst: dict[tuple[int, int], list[Any]]) -> list[tuple[int, int]]:
    used_src: set[int] = set()
    used_dst: set[int] = set()
    selected: list[tuple[int, int]] = []
    for src, dst in sorted(by_src_dst):
        if not by_src_dst[(src, dst)] or src in used_src or dst in used_dst:
            continue
        selected.append((src, dst))
        used_src.add(src)
        used_dst.add(dst)
    return selected


__all__ = ["ISLIPNoProgressError", "ISLIPRoundRobinPolicy", "_schedule_flows"]
