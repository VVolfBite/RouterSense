"""Derived phase-local B-side policies for paired RouterSense families."""

from __future__ import annotations

from typing import Any

from rs.scheduling.contracts import LogicalWave, MultiPhaseSchedulingProblem

from ..capabilities import PolicyCapabilities
from .common import build_phase_serial_release_aware_plan, flows_from_matrix, include_real_p2_phase


def _row_sums(matrix: tuple[tuple[int, ...], ...]) -> list[int]:
    return [int(sum(int(value) for value in row)) for row in matrix]


def _col_sums(matrix: tuple[tuple[int, ...], ...]) -> list[int]:
    if not matrix:
        return []
    width = len(matrix[0])
    return [int(sum(int(matrix[src][dst]) for src in range(len(matrix)))) for dst in range(width)]


class _DerivedPhaseLocalPolicy:
    policy_version = "v1"
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
    policy_name = "derived_phase_local"
    information_mode = "phase_local_family"
    tie_break_rule = "family_score desc, byte_count desc, src_rank,dst_rank"
    service_model = "phase_local_family"
    family_note = ""

    def __init__(self, *, bucket_rows: int = 0) -> None:
        self.bucket_rows = int(bucket_rows)

    def _score(self, *, phase: str, src_rank: int, dst_rank: int, byte_count: int, matrix: tuple[tuple[int, ...], ...]) -> tuple[float, float, int, int]:
        return (float(byte_count), float(byte_count), src_rank, dst_rank)

    def _pack(self, flows: tuple[Any, ...], *, matrix: tuple[tuple[int, ...], ...], phase: str, start_wave_id: int) -> tuple[LogicalWave, ...]:
        pending = sorted(
            flows,
            key=lambda flow: (
                -self._score(
                    phase=phase,
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(flow.byte_count),
                    matrix=matrix,
                )[0],
                -int(flow.byte_count),
                int(flow.src_rank),
                int(flow.dst_rank),
            ),
        )
        waves: list[LogicalWave] = []
        wave_id = int(start_wave_id)
        while pending:
            used_src: set[int] = set()
            used_dst: set[int] = set()
            chosen: list[Any] = []
            remaining: list[Any] = []
            for flow in pending:
                src = int(flow.src_rank)
                dst = int(flow.dst_rank)
                if src in used_src or dst in used_dst:
                    remaining.append(flow)
                    continue
                chosen.append(flow)
                used_src.add(src)
                used_dst.add(dst)
            waves.append(
                LogicalWave(
                    wave_id=wave_id,
                    flows=tuple(chosen),
                    duration=float(max((int(flow.byte_count) for flow in chosen), default=0)),
                )
            )
            pending = remaining
            wave_id += 1
        return tuple(waves)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem):  # type: ignore[no-untyped-def]
        p0_flows = flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True)
        p1_flows = flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True)
        p2_flows = ()
        if include_real_p2_phase(problem):
            p2_flows = flows_from_matrix(
                problem.p2_next_dispatch_forecast_matrix,
                phase="p2_next_dispatch",
                release_state="ready",
                executable=True,
            )
        p0_waves = self._pack(p0_flows, matrix=problem.p0_dispatch_matrix, phase="p0_dispatch", start_wave_id=0)
        p1_waves = self._pack(p1_flows, matrix=problem.p1_return_matrix, phase="p1_return", start_wave_id=len(p0_waves))
        p2_waves = self._pack(
            p2_flows,
            matrix=problem.p2_next_dispatch_forecast_matrix,
            phase="p2_next_dispatch",
            start_wave_id=len(p0_waves) + len(p1_waves),
        )
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode=self.information_mode,
            tie_break_rule=self.tie_break_rule,
            priority_components={"family_note": self.family_note},
            p0_waves=p0_waves,
            p1_waves=p1_waves,
            p2_waves=p2_waves,
            service_model=self.service_model,
        )


class BGatedGreedyMaximalPolicy(_DerivedPhaseLocalPolicy):
    policy_name = "B_gated_greedy_maximal"
    information_mode = "phase_local_gated_greedy"
    service_model = "phase_local_gated_greedy"
    family_note = "derived from U_gated_greedy_maximal without joint coupling"


class BGatedMaxweightMatchingPolicy(_DerivedPhaseLocalPolicy):
    policy_name = "B_gated_maxweight_matching"
    information_mode = "phase_local_gated_maxweight"
    service_model = "phase_local_gated_maxweight"
    family_note = "derived from U_gated_maxweight_matching without downstream pressure"

    def _score(self, *, phase: str, src_rank: int, dst_rank: int, byte_count: int, matrix: tuple[tuple[int, ...], ...]) -> tuple[float, float, int, int]:
        rows = _row_sums(matrix)
        cols = _col_sums(matrix)
        local_weight = int((rows[src_rank] if src_rank < len(rows) else 0) + (cols[dst_rank] if dst_rank < len(cols) else 0))
        return (float(byte_count + local_weight), float(local_weight), src_rank, dst_rank)


class BBarrierCriticalityMatchingPolicy(_DerivedPhaseLocalPolicy):
    policy_name = "B_barrier_criticality_matching"
    information_mode = "phase_local_barrier_criticality"
    service_model = "phase_local_barrier_criticality"
    family_note = "derived from U_barrier_criticality_global_matching without cross-phase dependency"

    def _score(self, *, phase: str, src_rank: int, dst_rank: int, byte_count: int, matrix: tuple[tuple[int, ...], ...]) -> tuple[float, float, int, int]:
        rows = _row_sums(matrix)
        cols = _col_sums(matrix)
        barrier_pressure = int((rows[dst_rank] if dst_rank < len(rows) else 0) + (cols[dst_rank] if dst_rank < len(cols) else 0))
        return (float(byte_count + 2 * barrier_pressure), float(barrier_pressure), src_rank, dst_rank)
