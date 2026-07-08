from __future__ import annotations

import time

from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext, PlanWave

from ..capabilities import PolicyCapabilities
from ..matching import maximum_weight_bipartite_matching
from .common import build_phase_serial_release_aware_plan, build_transfer_layouts_and_tasks, finalize_execution_plan, include_real_p2_phase


class BirkhoffPhaseLocalPolicy:
    policy_name = "birkhoff_phase_local"
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

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def _logical_decompose(self, matrix: tuple[tuple[int, ...], ...], *, phase: str, start_wave_id: int) -> tuple[list[LogicalWave], list[WaveDiagnostics]]:
        residual = {
            (src_rank, dst_rank): int(byte_count)
            for src_rank, row in enumerate(matrix)
            for dst_rank, byte_count in enumerate(row)
            if src_rank != dst_rank and int(byte_count) > 0
        }
        waves: list[LogicalWave] = []
        diags: list[WaveDiagnostics] = []
        wave_id = start_wave_id
        ranks = tuple(range(len(matrix)))
        while residual:
            remaining_before = float(sum(residual.values()))

            def weight(src: int, dst: int) -> float:
                return 1.0 if residual.get((src, dst), 0) > 0 else 0.0

            edges = maximum_weight_bipartite_matching(sources=ranks, destinations=ranks, edge_weight=weight)
            chosen = [edge for edge in edges if edge in residual]
            if not chosen:
                raise ValueError("birkhoff_phase_local could not find a legal matching")
            quantum = min(int(residual[edge]) for edge in chosen)
            flows = []
            selected_edges = []
            for src_rank, dst_rank in chosen:
                flows.append(
                    FlowDemand(
                        flow_id=f"{phase}:{src_rank}->{dst_rank}:wave{wave_id}",
                        phase=phase,
                        src_rank=int(src_rank),
                        dst_rank=int(dst_rank),
                        byte_count=int(quantum),
                        release_state="ready",
                        is_executable=True,
                    )
                )
                selected_edges.append({"src_rank": int(src_rank), "dst_rank": int(dst_rank), "byte_count": int(quantum)})
                residual[(src_rank, dst_rank)] -= int(quantum)
                if residual[(src_rank, dst_rank)] <= 0:
                    residual.pop((src_rank, dst_rank), None)
            remaining_after = float(sum(residual.values()))
            waves.append(LogicalWave(wave_id=wave_id, flows=tuple(flows), duration=float(quantum)))
            diags.append(
                WaveDiagnostics(
                    wave_id=wave_id,
                    selected_flow_ids=tuple(flow.flow_id for flow in flows),
                    selected_edges=tuple(selected_edges),
                    matching_weight=float(len(chosen)),
                    priority_components={"coefficient": int(quantum)},
                    remaining_bytes_before=remaining_before,
                    remaining_bytes_after=remaining_after,
                    ready_flow_count_before=len(residual) + len(chosen),
                    blocked_flow_count_before=0,
                    forecast_pressure_summary={},
                    selection_reason="support matching with minimum residual coefficient",
                )
            )
            wave_id += 1
        return waves, diags

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        p0_waves, p0_diags = self._logical_decompose(problem.p0_dispatch_matrix, phase="p0_dispatch", start_wave_id=0)
        p1_waves, p1_diags = self._logical_decompose(problem.p1_return_matrix, phase="p1_return", start_wave_id=len(p0_waves))
        p2_waves: list[LogicalWave] = []
        p2_diags: list[WaveDiagnostics] = []
        if include_real_p2_phase(problem):
            p2_waves, p2_diags = self._logical_decompose(
                problem.p2_next_dispatch_forecast_matrix,
                phase="p2_next_dispatch",
                start_wave_id=len(p0_waves) + len(p1_waves),
            )
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_matrix_decomposition",
            tie_break_rule="lexicographic support matching",
            priority_components={
                "selection_rule": "support matching",
                "service_quantum": "minimum residual on matching",
                "phase_wave_diagnostics": [item.to_dict() for item in (p0_diags + p1_diags + p2_diags)],
            },
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_fluid",
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
        residual = {}
        for task in all_tasks:
            residual.setdefault((int(task.src_rank), int(task.dst_rank)), []).append(task)
        waves: list[PlanWave] = []
        wave_diags = []
        wave_id = 0
        ranks = tuple(int(rank) for rank in local_context.ep_group_ranks)
        while any(residual.values()):
            def weight(src: int, dst: int) -> float:
                return 1.0 if residual.get((src, dst), []) else 0.0

            edges = maximum_weight_bipartite_matching(sources=ranks, destinations=ranks, edge_weight=weight)
            chosen_tasks = []
            for edge in edges:
                queue = residual.get(edge, [])
                if not queue:
                    continue
                chosen_tasks.append(queue.pop(0))
            if not chosen_tasks:
                raise ValueError("birkhoff_phase_local could not select any bucket task")
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen_tasks)))
            wave_diags.append(float(sum(int(task.byte_count) for task in chosen_tasks)))
            wave_id += 1
        pack_time_us = (time.perf_counter_ns() - schedule_started_ns) / 1000.0
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "phase_local_matrix_decomposition",
            "bucket_order": [task.task_id for wave in waves for task in wave.bucket_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": wave_diags,
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "p1_dependency_used": False,
            "p2_forecast_used": False,
            "p2_source": local_context.p2_hint.hint_source,
            "evaluation_eligible": True,
            "priority_components": {"selection_rule": "support matching over non-empty edges"},
            "tie_break_rule": "lexicographic support matching",
            "fallback_reason": "",
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=[task for wave in waves for task in wave.bucket_tasks],
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
