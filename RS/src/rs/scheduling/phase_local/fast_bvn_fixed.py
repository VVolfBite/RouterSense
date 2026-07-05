from __future__ import annotations

from collections import defaultdict

from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave

from .fifo import build_transfer_layouts_and_tasks, finalize_execution_plan
from ..capabilities import PolicyCapabilities
from ..matching import maximum_weight_bipartite_matching


class FastBVNSingleTierPolicy:
    policy_name = "fast_bvn_single_tier"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        uses_p0=True,
        uses_p1=True,
        uses_p2=False,
        cross_phase=False,
        requires_topology=False,
        supports_sync_before_phase=True,
        supports_default_continue=False,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        if len(local_context.ep_group_ranks) > 8:
            raise ValueError("unsupported_policy_scale")
        transfer_layouts, all_tasks = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
        )
        edge_queues: dict[tuple[int, int], list[BucketTask]] = defaultdict(list)
        for task in all_tasks:
            edge_queues[(int(task.src_rank), int(task.dst_rank))].append(task)
        for tasks in edge_queues.values():
            tasks.sort(key=lambda task: (int(task.segment_ordinal), int(task.bucket_ordinal)))

        waves: list[PlanWave] = []
        wave_weights: list[float] = []
        matching_trace: list[dict] = []
        wave_id = 0
        srcs = tuple(int(rank) for rank in local_context.ep_group_ranks)
        dsts = tuple(int(rank) for rank in local_context.ep_group_ranks)
        while any(edge_queues.values()):
            remaining_before = float(sum(sum(int(task.byte_count) for task in tasks) for tasks in edge_queues.values()))

            def weight(src: int, dst: int) -> float:
                return float(sum(int(task.byte_count) for task in edge_queues.get((src, dst), [])))

            edges = maximum_weight_bipartite_matching(sources=srcs, destinations=dsts, edge_weight=weight)
            chosen_tasks: list[BucketTask] = []
            selected_edges: list[dict] = []
            total_weight = 0.0
            for src, dst in edges:
                queue = edge_queues.get((src, dst), [])
                if not queue:
                    continue
                task = queue.pop(0)
                chosen_tasks.append(task)
                edge_weight = weight(src, dst) + float(task.byte_count)
                total_weight += float(edge_weight)
                selected_edges.append({"src_rank": src, "dst_rank": dst, "bucket_id": task.task_id, "matching_weight": edge_weight})
            if not chosen_tasks:
                raise ValueError("fast_bvn_single_tier could not select any edge")
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen_tasks)))
            remaining_after = float(sum(sum(int(task.byte_count) for task in tasks) for tasks in edge_queues.values()))
            wave_weights.append(total_weight)
            matching_trace.append(
                {
                    "wave_id": wave_id,
                    "matching_weight": total_weight,
                    "remaining_weight_before": remaining_before,
                    "remaining_weight_after": remaining_after,
                    "selected_edges": selected_edges,
                }
            )
            wave_id += 1

        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_capabilities": self.capabilities.to_dict(),
            "bucket_order": [task.task_id for wave in waves for task in wave.bucket_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": wave_weights,
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "priority_components": {"matching_trace": matching_trace, "selection_rule": "maximum_weight_bipartite_matching"},
            "tie_break_rule": "sorted matched edges by src_rank,dst_rank after max weight",
            "fallback_reason": "",
            "evaluation_eligible": True,
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
        )
