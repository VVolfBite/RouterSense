from __future__ import annotations

from collections import defaultdict

from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave

from .bucketed_fifo import build_transfer_layouts_and_tasks, finalize_execution_plan
from .capabilities import PolicyCapabilities


class RouterSenseP0P1ReservationPolicy:
    policy_name = "routersense_p0p1_reservation"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        uses_p0=True,
        uses_p1=True,
        uses_p2=False,
        cross_phase=True,
        requires_topology=False,
        supports_sync_before_phase=True,
        supports_default_continue=False,
    )

    def __init__(self, *, bucket_rows: int, p0_weight: float = 1.0, p1_reservation_weight: float = 1.0) -> None:
        self.bucket_rows = int(bucket_rows)
        self.p0_weight = float(p0_weight)
        self.p1_reservation_weight = float(p1_reservation_weight)

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
        if local_context.phase == "P1":
            ordered_tasks = sorted(all_tasks, key=lambda task: (int(task.src_rank), int(task.dst_rank), int(task.segment_ordinal), int(task.bucket_ordinal)))
            diagnostics = {
                "bucket_order": [task.task_id for task in ordered_tasks],
                "wave_edges": [],
                "per_wave_matching_weight": [],
                "uses_current_phase_demand": True,
                "uses_p1_reservation": False,
                "uses_p2_hint": False,
                "priority_components": {"mode": "real_p1_only"},
                "tie_break_rule": "src_rank,dst_rank,segment_ordinal,bucket_ordinal",
                "fallback_reason": "",
                "evaluation_eligible": True,
                "p1_reservation_seen": False,
                "p1_reservation_bytes": 0,
                "p1_reservation_endpoint_pressure": {},
                "p1_reservation_influenced_p0_plan": False,
            }
            waves = []
            pending = ordered_tasks[:]
            wave_id = 0
            while pending:
                used_outgoing, used_incoming, chosen, remaining = set(), set(), [], []
                for task in pending:
                    if int(task.src_rank) in used_outgoing or int(task.dst_rank) in used_incoming:
                        remaining.append(task)
                        continue
                    chosen.append(task)
                    used_outgoing.add(int(task.src_rank))
                    used_incoming.add(int(task.dst_rank))
                waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen)))
                pending = remaining
                wave_id += 1
            diagnostics["wave_edges"] = [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves]
            diagnostics["per_wave_matching_weight"] = [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves]
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
            )

        future_out_pressure = defaultdict(int)
        future_in_pressure = defaultdict(int)
        for layout in transfer_layouts:
            if int(layout.src_rank) == int(layout.dst_rank):
                continue
            future_out_pressure[int(layout.dst_rank)] += int(layout.byte_count)
            future_in_pressure[int(layout.src_rank)] += int(layout.byte_count)

        ordered_tasks = sorted(
            all_tasks,
            key=lambda task: (
                -(
                    self.p0_weight * float(task.byte_count)
                    + self.p1_reservation_weight * float(future_out_pressure[int(task.dst_rank)] + future_in_pressure[int(task.src_rank)])
                ),
                -int(task.byte_count),
                int(task.src_rank),
                int(task.dst_rank),
                int(task.bucket_ordinal),
            ),
        )
        waves: list[PlanWave] = []
        pending = ordered_tasks[:]
        wave_id = 0
        while pending:
            used_outgoing, used_incoming, chosen, remaining = set(), set(), [], []
            for task in pending:
                if int(task.src_rank) in used_outgoing or int(task.dst_rank) in used_incoming:
                    remaining.append(task)
                    continue
                chosen.append(task)
                used_outgoing.add(int(task.src_rank))
                used_incoming.add(int(task.dst_rank))
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen)))
            pending = remaining
            wave_id += 1
        diagnostics = {
            "bucket_order": [task.task_id for task in ordered_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": True,
            "uses_p2_hint": False,
            "priority_components": {"p0_weight": self.p0_weight, "p1_reservation_weight": self.p1_reservation_weight},
            "tie_break_rule": "src_rank,dst_rank,bucket_ordinal",
            "fallback_reason": "",
            "evaluation_eligible": True,
            "p1_reservation_seen": True,
            "p1_reservation_bytes": int(sum(int(layout.byte_count) for layout in transfer_layouts if int(layout.src_rank) != int(layout.dst_rank))),
            "p1_reservation_endpoint_pressure": {
                "future_out_pressure": {str(k): int(v) for k, v in future_out_pressure.items()},
                "future_in_pressure": {str(k): int(v) for k, v in future_in_pressure.items()},
            },
            "p1_reservation_influenced_p0_plan": True,
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
        )
