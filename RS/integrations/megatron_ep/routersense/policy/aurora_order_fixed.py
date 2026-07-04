from __future__ import annotations

from collections import defaultdict

from integrations.megatron_ep.routersense.phase import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave

from .bucketed_fifo import build_transfer_layouts_and_tasks, finalize_execution_plan
from .capabilities import PolicyCapabilities


class AuroraOrderFixedPolicy:
    policy_name = "aurora_order_fixed"
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
        transfer_layouts, all_tasks = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
        )
        remaining = list(all_tasks)
        waves: list[PlanWave] = []
        wave_scores: list[float] = []
        selection_trace: list[dict] = []
        wave_id = 0
        while remaining:
            source_pressure = defaultdict(int)
            destination_pressure = defaultdict(int)
            for task in remaining:
                source_pressure[int(task.src_rank)] += int(task.byte_count)
                destination_pressure[int(task.dst_rank)] += int(task.byte_count)
            used_outgoing: set[int] = set()
            used_incoming: set[int] = set()
            selected: list[BucketTask] = []
            wave_weight = 0.0
            while True:
                candidates = [task for task in remaining if int(task.src_rank) not in used_outgoing and int(task.dst_rank) not in used_incoming]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda task: (
                        -max(source_pressure[int(task.src_rank)], destination_pressure[int(task.dst_rank)]),
                        -int(task.byte_count),
                        int(task.src_rank),
                        int(task.dst_rank),
                        int(task.bucket_ordinal),
                    )
                )
                chosen = candidates[0]
                src = int(chosen.src_rank)
                dst = int(chosen.dst_rank)
                score = float(max(source_pressure[src], destination_pressure[dst]))
                selection_trace.append(
                    {
                        "wave_id": wave_id,
                        "bucket_id": chosen.task_id,
                        "src_rank": src,
                        "dst_rank": dst,
                        "source_pressure": int(source_pressure[src]),
                        "destination_pressure": int(destination_pressure[dst]),
                        "priority_score": score,
                        "selection_reason": "max(source_pressure,destination_pressure)->byte_count->src->dst->bucket",
                    }
                )
                selected.append(chosen)
                wave_weight += float(chosen.byte_count)
                used_outgoing.add(src)
                used_incoming.add(dst)
                remaining.remove(chosen)
                source_pressure[src] -= int(chosen.byte_count)
                destination_pressure[dst] -= int(chosen.byte_count)
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(selected)))
            wave_scores.append(wave_weight)
            wave_id += 1

        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_capabilities": self.capabilities.to_dict(),
            "bucket_order": [task.task_id for wave in waves for task in wave.bucket_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": wave_scores,
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "priority_components": {
                "source_pressure": "remaining outgoing bytes at source",
                "destination_pressure": "remaining incoming bytes at destination",
                "priority": "max(source_pressure,destination_pressure)->flow bytes",
                "selection_trace": selection_trace,
            },
            "tie_break_rule": "src_rank,dst_rank,bucket_ordinal",
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
