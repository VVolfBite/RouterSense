from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem, PreparedWindowPlan
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from ..capabilities import PolicyCapabilities
from .scheduler_state import run_global_matching_scheduler


class UnsupportedOnlineMultiPhaseExecution(RuntimeError):
    pass


def _normalize_information_mode(mode: str) -> str:
    normalized = str(mode or "p0_p1_p2")
    if normalized not in {"p0_only", "p0_p1", "p0_p1_p2"}:
        raise ValueError(f"unsupported RouterSense information_mode {mode!r}")
    return normalized


def _scheduler_result_to_logical_plan(
    *,
    policy_name: str,
    policy_version: str,
    information_mode: str,
    capabilities: PolicyCapabilities,
    problem: MultiPhaseSchedulingProblem,
    result: dict[str, Any],
) -> LogicalSchedulePlan:
    by_wave: dict[int, list[dict[str, Any]]] = {}
    for entry in result["schedule"]:
        by_wave.setdefault(int(entry["wave_id"]), []).append(entry)
    waves: list[LogicalWave] = []
    per_wave: list[WaveDiagnostics] = []
    for wave_id in sorted(by_wave):
        rows = sorted(by_wave[wave_id], key=lambda item: (int(item["phase"]), int(item["src_gpu"]), int(item["dst_gpu"])))
        flows = []
        for item in rows:
            phase_id = int(item["phase"])
            phase = "p0_dispatch" if phase_id == 0 else "p1_return" if phase_id == 1 else "p2_next_dispatch_forecast"
            origin_flow_id = str(item.get("flow_id", f"phase{phase_id}_src{item['src_gpu']}_dst{item['dst_gpu']}"))
            segment_id = str(item.get("chunk_id", f"{origin_flow_id}_wave{wave_id}"))
            flows.append(
                {
                    "flow_id": segment_id,
                    "origin_flow_id": origin_flow_id,
                    "phase": phase,
                    "src_rank": int(item["src_gpu"]),
                    "dst_rank": int(item["dst_gpu"]),
                    "byte_count": int(round(float(item["served_volume"]))),
                    "served_volume": float(item.get("served_volume", 0.0)),
                    "residual_before": float(item.get("residual_before", item.get("served_volume", 0.0))),
                    "residual_after": float(item.get("residual_after", 0.0)),
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "wave_id": int(item.get("wave_id", wave_id)),
                }
            )
        logical_flows = tuple(
            __import__("rs.scheduling.contracts", fromlist=["FlowDemand"]).FlowDemand(
                flow_id=row["flow_id"],
                phase=row["phase"],
                src_rank=row["src_rank"],
                dst_rank=row["dst_rank"],
                byte_count=row["byte_count"],
                release_state="ready",
                is_executable=row["phase"] != "p2_next_dispatch_forecast",
                dependency_metadata={
                    "origin_flow_id": row["origin_flow_id"],
                    "segment_id": row["flow_id"],
                    "service_model": "fluid_wave",
                    "served_volume": row["served_volume"],
                    "residual_before": row["residual_before"],
                    "residual_after": row["residual_after"],
                    "start": row["start"],
                    "end": row["end"],
                    "wave_id": row["wave_id"],
                },
            )
            for row in flows
        )
        duration = max((int(row["byte_count"]) for row in flows), default=0)
        waves.append(LogicalWave(wave_id=wave_id, flows=logical_flows, duration=float(duration)))
        per_wave.append(
            WaveDiagnostics(
                wave_id=wave_id,
                selected_flow_ids=tuple(row["flow_id"] for row in flows),
                selected_edges=tuple(
                    {
                        "phase": row["phase"],
                        "src_rank": row["src_rank"],
                        "dst_rank": row["dst_rank"],
                        "byte_count": row["byte_count"],
                        "origin_flow_id": row["origin_flow_id"],
                        "segment_id": row["flow_id"],
                    }
                    for row in flows
                ),
                matching_weight=float(sum(int(row["byte_count"]) for row in flows)),
                priority_components={"mode": result["mode"]},
                remaining_bytes_before=float(sum(int(entry["served_volume"]) for entry in rows)),
                remaining_bytes_after=0.0,
                ready_flow_count_before=len(problem.flow_window.ready_flows),
                blocked_flow_count_before=len(problem.flow_window.blocked_flows),
                forecast_pressure_summary={"source": problem.forecast.source if problem.forecast is not None else "none"},
                selection_reason="multiphase ready-set matching",
            )
        )
    diagnostics = PolicyDiagnostics(
        policy_name=policy_name,
        policy_version=policy_version,
        information_mode=information_mode,
        tie_break_rule="score desc -> matching selection",
        wave_count=len(waves),
        logical_flow_count=sum(len(wave.flows) for wave in waves),
        ready_flow_count=len(problem.flow_window.ready_flows),
        blocked_flow_count=len(problem.flow_window.blocked_flows),
        forecast_flow_count=len(problem.flow_window.forecast_pressure),
        p1_dependency_used=information_mode in {"p0_p1", "p0_p1_p2"},
        p2_forecast_used=information_mode == "p0_p1_p2",
        p2_source=problem.forecast.source if problem.forecast is not None else "none",
        evaluation_eligible=capabilities.evaluation_eligible and bool(problem.forecast.evaluation_eligible if problem.forecast is not None else True),
        per_wave=tuple(per_wave),
        priority_components={
            "p0_weight": problem.options.p0_weight,
            "p1_reservation_weight": problem.options.p1_reservation_weight,
            "p2_hint_weight": problem.options.p2_hint_weight,
            "prediction_confidence": problem.options.prediction_confidence,
            "raw_strategy": result["strategy"],
        },
    )
    return LogicalSchedulePlan(
        policy_name=policy_name,
        waves=tuple(waves),
        diagnostics={
            **diagnostics.to_dict(),
            "mode": result["mode"],
            "prediction_used": result["prediction_used"],
            "makespan": result["makespan"],
            "solve_time_ms": result["solve_time_ms"],
            "audit": result["audit"],
            "raw_schedule": result["schedule"],
            "service_model": "fluid_wave",
            "future_information_mode": (
                "oracle_execution_window"
                if problem.options.scheduling_mode == "execution_window"
                else "heuristic_runtime_lookahead"
                if information_mode == "p0_p1_p2"
                and problem.forecast is not None
                and problem.forecast.source == "copy_current_dispatch"
                and problem.options.prediction_confidence > 0.0
                else "oracle_predicted_runtime_lookahead"
                if information_mode == "p0_p1_p2"
                and problem.forecast is not None
                and bool(problem.forecast.oracle)
                and problem.options.prediction_confidence > 0.0
                else "none"
            ),
            "evaluation_eligible": (
                False
                if problem.options.scheduling_mode == "execution_window"
                else False
                if information_mode == "p0_p1_p2"
                and problem.forecast is not None
                and bool(problem.forecast.oracle)
                and problem.options.prediction_confidence > 0.0
                else bool(capabilities.evaluation_eligible)
            ),
        },
    )


class RouterSenseMultiphaseLookaheadPolicy:
    policy_version = "v1"
    base_policy_name = "routersense_multiphase_lookahead"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=False,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=True,
        uses_p2_forecast=True,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def __init__(self, *, information_mode: str, p0_weight: float, p1_reservation_weight: float, p2_hint_weight: float) -> None:
        self.information_mode = _normalize_information_mode(information_mode)
        self.policy_name = f"{self.base_policy_name}:{self.information_mode}"
        self.p0_weight = float(p0_weight)
        self.p1_reservation_weight = float(p1_reservation_weight)
        self.p2_hint_weight = float(p2_hint_weight)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        prediction_confidence = float(problem.options.prediction_confidence)
        barrier_weight = self.p1_reservation_weight if self.information_mode in {"p0_p1", "p0_p1_p2"} else 0.0
        prediction_weight = self.p2_hint_weight if self.information_mode == "p0_p1_p2" else 0.0
        if self.information_mode != "p0_p1_p2":
            prediction_confidence = 0.0
        prediction_matrix = None
        if problem.forecast is not None:
            metadata = dict(problem.forecast.metadata or {})
            hint = metadata.get("planning_hint_matrix")
            prediction_matrix = (
                [list(row) for row in problem.forecast.matrix]
                if hint is None
                else [list(row) for row in canonicalize_remote_matrix(hint)]
            )
        result = run_global_matching_scheduler(
            [list(row) for row in problem.p0_dispatch_matrix],
            [list(row) for row in problem.p1_return_matrix],
            [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
            int(problem.topology.num_gpus),
            strategy="routersense_multiphase_lookahead",
            mode=problem.options.scheduling_mode,
            prediction_confidence=prediction_confidence,
            expert_compute_delay=float(problem.release_model.expert_compute_delay),
            exact_matching=True,
            wave_quantum=None,
            max_waves=256,
            residual_weight=float(self.p0_weight),
            barrier_weight=float(barrier_weight),
            age_weight=0.1,
            prediction_weight=float(prediction_weight),
            adaptive_prices=False,
            price_step=0.0,
            price_decay=0.0,
            price_clip=0.0,
            iteration_budget=1,
            atomic=False,
            prediction_matrix=prediction_matrix,
        )
        return _scheduler_result_to_logical_plan(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            information_mode=self.information_mode,
            capabilities=self.capabilities,
            problem=problem,
            result=result,
        )

    def build_prepared_window_plan(
        self,
        *,
        problem: MultiPhaseSchedulingProblem,
        created_at_layer_id: str,
        applies_from_layer_id: str,
    ) -> PreparedWindowPlan:
        logical_plan = self.build_logical_plan(problem)
        window_key = hashlib.sha256(
            json.dumps(
                {
                    "policy": self.policy_name,
                    "created_at_layer_id": created_at_layer_id,
                    "applies_from_layer_id": applies_from_layer_id,
                    "forecast_digest": problem.forecast.digest if problem.forecast is not None else "",
                    "plan": logical_plan.to_dict(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return PreparedWindowPlan(
            window_key=window_key,
            forecast_digest=problem.forecast.digest if problem.forecast is not None else "",
            logical_plan=logical_plan,
            created_at_layer_id=str(created_at_layer_id),
            applies_from_layer_id=str(applies_from_layer_id),
            execution_capability_required="multiphase_pending_window",
            forecast_matrix=tuple(tuple(int(value) for value in row) for row in problem.p2_next_dispatch_forecast_matrix),
        )
