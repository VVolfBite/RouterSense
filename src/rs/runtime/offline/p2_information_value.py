"""Offline simulation of P2 information timing.

This module separates three concepts that are easy to conflate in an offline
replay:

* ``perfect``: the true P2 matrix is known before P0/P1 scheduling starts;
* ``reactive``: each true P2 source row is revealed only after that rank has
  completed its P1 inbound barrier;
* ``predicted``: an advisory P2 matrix may influence P0/P1 decisions, while
  executable P2 bytes are still created only when the corresponding true row
  is revealed.

The dynamic simulator is intentionally model-agnostic.  It consumes only the
three traffic matrices, the phase-release dependency, and a scheduling-family
kernel specification.  Predicted traffic never becomes executable traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Literal

from rs.scheduling.families import get_family_kernel_spec
from rs.scheduling.multiphase.critical_frontier import critical_frontier_candidates
from rs.scheduling.multiphase.dependency_model import (
    collect_real_flows,
    inbound_remaining,
    outbound_loads,
)
from rs.scheduling.multiphase.flow_model import (
    EXECUTION_WINDOW_MODE,
    RUNTIME_LOOKAHEAD_MODE,
    ResidualFlowState,
)
from rs.scheduling.multiphase.matching import (
    greedy_maximal_matching,
    maximum_weight_matching,
)
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler
from rs.scheduling.multiphase.scoring import ready_flow_candidates

P2InformationMode = Literal["perfect", "reactive", "predicted"]


@dataclass(frozen=True)
class P2InformationResult:
    family_id: str
    information_mode: P2InformationMode
    makespan: float
    planning_time_ms: float
    wave_count: int
    valid: bool
    schedule: tuple[dict[str, Any], ...]
    audit: dict[str, Any]
    revealed_rows: int

    def to_dict(self, *, include_schedule: bool = False) -> dict[str, Any]:
        row = {
            "family_id": self.family_id,
            "information_mode": self.information_mode,
            "makespan": self.makespan,
            "planning_time_ms": self.planning_time_ms,
            "wave_count": self.wave_count,
            "valid": self.valid,
            "revealed_rows": self.revealed_rows,
            "audit": self.audit,
        }
        if include_schedule:
            row["schedule"] = list(self.schedule)
        return row


def _zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _copy_matrix(matrix: list[list[int]]) -> list[list[int]]:
    return [[int(value) for value in row] for row in matrix]


def _validate_square_matrices(*matrices: list[list[int]]) -> int:
    if not matrices:
        raise ValueError("at least one matrix is required")
    size = len(matrices[0])
    if size <= 0:
        raise ValueError("traffic matrices must be non-empty")
    for matrix in matrices:
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("all traffic matrices must have the same square shape")
        if any(int(value) < 0 for row in matrix for value in row):
            raise ValueError("traffic matrices must be non-negative")
    return size


def _critical_frontier_ready(
    *,
    spec,
    flows: list[ResidualFlowState],
    residual: dict[str, float],
    ready_since: dict[str, float],
    current_time: float,
    release_time: dict[tuple[int, int], float],
    inbound: dict[tuple[int, int], float],
    downstream_load: dict[tuple[int, int], float],
    future_matrix: list[list[int]],
    prediction_confidence: float,
) -> list[dict[str, Any]]:
    return critical_frontier_candidates(
        flows=flows,
        residual=residual,
        ready_since=ready_since,
        current_time=current_time,
        release_time=release_time,
        inbound_remaining=inbound,
        downstream_load=downstream_load,
        age_scale=max(1.0, current_time + 1.0),
        residual_weight=float(spec.residual_weight),
        barrier_weight=float(spec.barrier_weight),
        age_weight=float(spec.age_weight),
        prediction_weight=float(spec.prediction_weight),
        release_gain_weight=float(spec.release_gain_weight),
        endpoint_pressure_weight=float(spec.endpoint_pressure_weight),
        critical_path_weight=float(spec.critical_path_weight),
        transitive_unlock_weight=float(spec.transitive_unlock_weight),
        endpoint_dual_weight=float(spec.endpoint_dual_weight),
        duplex_pair_weight=float(spec.duplex_pair_weight),
        dual_temperature=float(spec.dual_temperature),
        transitive_tail_weight=float(spec.transitive_tail_weight),
        destination_hotspot_weight=float(spec.destination_hotspot_weight),
        size_bias_power=float(spec.size_bias_power),
        mode=RUNTIME_LOOKAHEAD_MODE,
        prediction_confidence=float(prediction_confidence),
        future_matrix=future_matrix,
        base_score_lookup=None,
        base_priority_weight=0.0,
    )


def _weighted_ready(
    *,
    spec,
    flows: list[ResidualFlowState],
    residual: dict[str, float],
    ready_since: dict[str, float],
    current_time: float,
    release_time: dict[tuple[int, int], float],
    inbound: dict[tuple[int, int], float],
    downstream_load: dict[tuple[int, int], float],
    prediction_confidence: float,
) -> list[dict[str, Any]]:
    return ready_flow_candidates(
        flows=flows,
        residual=residual,
        ready_since=ready_since,
        current_time=current_time,
        release_time=release_time,
        inbound_remaining=inbound,
        downstream_load=downstream_load,
        age_scale=max(1.0, current_time + 1.0),
        residual_weight=float(spec.residual_weight),
        barrier_weight=float(spec.barrier_weight),
        age_weight=float(spec.age_weight),
        prediction_weight=float(spec.prediction_weight),
        endpoint_pressure_weight=float(spec.endpoint_pressure_weight),
        release_gain_weight=float(spec.release_gain_weight),
        mode=RUNTIME_LOOKAHEAD_MODE,
        prediction_confidence=float(prediction_confidence),
        base_score_lookup=None,
        base_priority_weight=0.0,
    )


def simulate_p2_information(
    *,
    p0_dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_truth_matrix: list[list[int]],
    family_id: str,
    information_mode: P2InformationMode,
    p2_forecast_matrix: list[list[int]] | None = None,
    prediction_confidence: float = 1.0,
    expert_compute_delay: float = 0.0,
    max_waves: int = 4096,
) -> P2InformationResult:
    """Simulate one P0/P1/P2 window under a controlled information scope."""

    p0 = _copy_matrix(p0_dispatch_matrix)
    p1 = _copy_matrix(p1_return_matrix)
    p2_truth = _copy_matrix(p2_truth_matrix)
    num_gpus = _validate_square_matrices(p0, p1, p2_truth)
    spec = get_family_kernel_spec(family_id)

    if information_mode == "perfect":
        started = time.perf_counter()
        result = run_global_matching_scheduler(
            p0,
            p1,
            p2_truth,
            num_gpus,
            strategy=f"p2_information:{spec.family_id}:perfect",
            mode=EXECUTION_WINDOW_MODE,
            prediction_confidence=1.0,
            expert_compute_delay=float(expert_compute_delay),
            exact_matching=bool(spec.exact_matching),
            wave_quantum=None,
            max_waves=int(max_waves),
            residual_weight=float(spec.residual_weight),
            barrier_weight=float(spec.barrier_weight),
            age_weight=float(spec.age_weight),
            prediction_weight=float(spec.prediction_weight),
            endpoint_pressure_weight=float(spec.endpoint_pressure_weight),
            release_gain_weight=float(spec.release_gain_weight),
            adaptive_prices=bool(spec.adaptive_prices),
            price_step=float(spec.price_step),
            price_decay=float(spec.price_decay),
            price_clip=float(spec.price_clip),
            iteration_budget=int(spec.iteration_budget),
            atomic=bool(spec.atomic),
            prediction_matrix=p2_truth,
            scoring_model=str(spec.scoring_model),
            critical_path_weight=float(spec.critical_path_weight),
            transitive_unlock_weight=float(spec.transitive_unlock_weight),
            endpoint_dual_weight=float(spec.endpoint_dual_weight),
            duplex_pair_weight=float(spec.duplex_pair_weight),
            dual_temperature=float(spec.dual_temperature),
            transitive_tail_weight=float(spec.transitive_tail_weight),
            destination_hotspot_weight=float(spec.destination_hotspot_weight),
            size_bias_power=float(spec.size_bias_power),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        audit = dict(result.get("audit", {}))
        return P2InformationResult(
            family_id=spec.family_id,
            information_mode="perfect",
            makespan=float(result.get("makespan", 0.0)),
            planning_time_ms=elapsed_ms,
            wave_count=int(result.get("wave_count", 0)),
            valid=bool(audit.get("valid", False)),
            schedule=tuple(result.get("schedule", ())),
            audit=audit,
            revealed_rows=num_gpus,
        )

    if information_mode not in {"reactive", "predicted"}:
        raise ValueError(f"unsupported P2 information mode {information_mode!r}")
    if information_mode == "predicted" and p2_forecast_matrix is None:
        raise ValueError("predicted mode requires p2_forecast_matrix")
    if p2_forecast_matrix is not None:
        forecast = _copy_matrix(p2_forecast_matrix)
        _validate_square_matrices(p0, p1, p2_truth, forecast)
    else:
        forecast = _zero_matrix(num_gpus)

    started = time.perf_counter()
    zero = _zero_matrix(num_gpus)
    flows = collect_real_flows(p0, p1, zero, mode=EXECUTION_WINDOW_MODE)
    residual = {flow.flow_id: float(flow.volume) for flow in flows}
    inbound = inbound_remaining(flows, num_gpus)
    release_time = {
        (phase, gpu): (0.0 if phase == 0 else math.inf)
        for phase in range(3)
        for gpu in range(num_gpus)
    }
    for gpu in range(num_gpus):
        if inbound.get((0, gpu), 0.0) <= 1e-9:
            release_time[(1, gpu)] = float(expert_compute_delay)
    barrier_done = {
        (phase, gpu): 0.0
        for phase in range(3)
        for gpu in range(num_gpus)
    }
    ready_since: dict[str, float] = {}
    revealed: set[int] = set()
    schedule: list[dict[str, Any]] = []
    current_time = 0.0
    wave_count = 0

    def reveal_row(source: int, reveal_time: float) -> None:
        if source in revealed:
            return
        revealed.add(source)
        release_time[(2, source)] = float(reveal_time)
        for destination, volume_value in enumerate(p2_truth[source]):
            volume = float(volume_value)
            if source == destination or volume <= 0.0:
                continue
            flow = ResidualFlowState(
                flow_id=f"phase2_src{source}_dst{destination}",
                phase=2,
                src_gpu=source,
                dst_gpu=destination,
                volume=volume,
            )
            flows.append(flow)
            residual[flow.flow_id] = volume
            inbound[(2, destination)] = inbound.get((2, destination), 0.0) + volume

    for gpu in range(num_gpus):
        if inbound.get((1, gpu), 0.0) <= 1e-9:
            reveal_row(gpu, 0.0)

    while (
        any(value > 1e-9 for value in residual.values()) or len(revealed) < num_gpus
    ) and wave_count < max(1, int(max_waves)):
        future_view = _zero_matrix(num_gpus)
        if information_mode == "predicted":
            for source, row in enumerate(forecast):
                if source not in revealed:
                    future_view[source] = list(row)
        downstream = outbound_loads(
            p0,
            p1,
            future_view,
            mode=RUNTIME_LOOKAHEAD_MODE,
            prediction_confidence=(
                float(prediction_confidence) if information_mode == "predicted" else 0.0
            ),
        )
        common = {
            "spec": spec,
            "flows": flows,
            "residual": residual,
            "ready_since": ready_since,
            "current_time": current_time,
            "release_time": release_time,
            "inbound": inbound,
            "downstream_load": downstream,
            "prediction_confidence": (
                float(prediction_confidence) if information_mode == "predicted" else 0.0
            ),
        }
        if spec.scoring_model == "critical_frontier":
            ready = _critical_frontier_ready(
                **common,
                future_matrix=future_view,
            )
        elif spec.scoring_model == "weighted_components":
            ready = _weighted_ready(**common)
        else:
            raise ValueError(
                f"dynamic P2 simulator does not support scoring model {spec.scoring_model!r}"
            )

        if not ready:
            future_releases = [
                value
                for value in release_time.values()
                if value < math.inf and value > current_time + 1e-9
            ]
            if not future_releases:
                break
            current_time = min(future_releases)
            continue

        chosen = (
            maximum_weight_matching(ready, num_gpus)
            if spec.exact_matching
            else greedy_maximal_matching(ready)
        )
        if not chosen:
            break
        duration = max(
            min(float(candidate["residual"]) for candidate in chosen),
            1e-6,
        )
        wave_end = current_time
        rows_to_reveal: list[tuple[int, float]] = []
        for candidate in chosen:
            flow_id = str(candidate["flow_id"])
            phase = int(candidate["phase"])
            source = int(candidate["src_gpu"])
            destination = int(candidate["dst_gpu"])
            served = (
                max(0.0, residual[flow_id])
                if spec.atomic
                else duration
            )
            residual[flow_id] = max(0.0, residual[flow_id] - served)
            inbound[(phase, destination)] = max(
                0.0,
                inbound.get((phase, destination), 0.0) - served,
            )
            end_time = current_time + served
            wave_end = max(wave_end, end_time)
            barrier_done[(phase, destination)] = max(
                barrier_done[(phase, destination)],
                end_time,
            )
            if phase < 2 and inbound[(phase, destination)] <= 1e-9:
                next_release = barrier_done[(phase, destination)] + (
                    float(expert_compute_delay) if phase == 0 else 0.0
                )
                release_time[(phase + 1, destination)] = next_release
                if phase == 1:
                    rows_to_reveal.append((destination, next_release))
            schedule.append(
                {
                    "chunk_id": f"{flow_id}_wave{wave_count}",
                    "flow_id": flow_id,
                    "phase": phase,
                    "size": float(served),
                    "served_volume": float(served),
                    "src": source,
                    "dst": destination,
                    "src_gpu": source,
                    "dst_gpu": destination,
                    "start": float(current_time),
                    "end": float(end_time),
                    "wave_id": int(wave_count),
                    "priority": [
                        float(candidate.get("score", 0.0)),
                        float(candidate.get("barrier_urgency", 0.0)),
                        float(candidate.get("age", 0.0)),
                        float(candidate.get("prediction_bonus", 0.0)),
                        float(candidate.get("base_priority", 0.0)),
                    ],
                }
            )
        current_time = wave_end if spec.atomic else current_time + duration
        for source, reveal_time in rows_to_reveal:
            reveal_row(source, reveal_time)
        wave_count += 1

    makespan = max((float(row["end"]) for row in schedule), default=0.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    audit = replay_and_audit_schedule(
        schedule=schedule,
        dispatch_matrix=p0,
        combine_matrix=p1,
        next_dispatch_matrix=p2_truth,
        num_gpus=num_gpus,
        expert_compute_delay=float(expert_compute_delay),
        mode=EXECUTION_WINDOW_MODE,
        scheduler_name=f"p2_information:{spec.family_id}:{information_mode}",
        planning_time_ms=elapsed_ms,
        reported_makespan=makespan,
        prediction_used=information_mode == "predicted",
    )
    all_revealed = len(revealed) == num_gpus
    all_served = not any(value > 1e-9 for value in residual.values())
    valid = bool(audit.get("valid", False)) and all_revealed and all_served
    if not all_revealed or not all_served:
        audit = {
            **audit,
            "valid": False,
            "validation_errors": [
                *list(audit.get("validation_errors", ())),
                *([] if all_revealed else ["not all P2 source rows were revealed"]),
                *([] if all_served else ["residual traffic remained after simulation"]),
            ],
        }
    return P2InformationResult(
        family_id=spec.family_id,
        information_mode=information_mode,
        makespan=makespan,
        planning_time_ms=elapsed_ms,
        wave_count=wave_count,
        valid=valid,
        schedule=tuple(schedule),
        audit=audit,
        revealed_rows=len(revealed),
    )


__all__ = [
    "P2InformationMode",
    "P2InformationResult",
    "simulate_p2_information",
]
