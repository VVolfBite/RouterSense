"""Offline fluid Birkhoff-von Neumann crossbar reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.matching import maximum_weight_bipartite_matching
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_col_sums_remote, matrix_diagonal_report, matrix_nonzero_remote_edge_count, matrix_remote_bytes, matrix_row_sums_remote


@dataclass(frozen=True)
class FluidBVNCertificate:
    reference_model: str
    num_ports: int
    max_source_load: int
    max_destination_load: int
    fluid_optimal_horizon: int
    emitted_real_service_horizon: int
    dummy_idle_service_horizon: int
    coverage_verified: bool
    matching_constraints_verified: bool
    certificate_verified: bool
    self_bytes_ignored: int
    remote_bytes: int
    matrix_diagonal_nonzero_count: int
    waves: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_model": self.reference_model,
            "num_ports": self.num_ports,
            "max_source_load": self.max_source_load,
            "max_destination_load": self.max_destination_load,
            "fluid_optimal_horizon": self.fluid_optimal_horizon,
            "emitted_real_service_horizon": self.emitted_real_service_horizon,
            "dummy_idle_service_horizon": self.dummy_idle_service_horizon,
            "coverage_verified": self.coverage_verified,
            "matching_constraints_verified": self.matching_constraints_verified,
            "certificate_verified": self.certificate_verified,
            "self_bytes_ignored": self.self_bytes_ignored,
            "remote_bytes": self.remote_bytes,
            "matrix_diagonal_nonzero_count": self.matrix_diagonal_nonzero_count,
            "waves": list(self.waves),
        }


class BirkhoffVonNeumannFluidReference:
    policy_name = "birkhoff_von_neumann_fluid"
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

    def __init__(self, *, bucket_rows: int = 0) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        p0_waves, p0_certificate = decompose_fluid_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", start_wave_id=0)
        p1_waves, p1_certificate = decompose_fluid_matrix(problem.p1_return_matrix, phase="p1_return", start_wave_id=len(p0_waves))
        waves = tuple(p0_waves + p1_waves)
        certificate = {
            "reference_model": "birkhoff_von_neumann_fluid",
            "evaluation_model": "offline_fluid_crossbar",
            "runtime_latency_comparable": False,
            "online_executor_compatible": False,
            "phase_certificates": {
                "p0_dispatch": p0_certificate.to_dict(),
                "p1_return": p1_certificate.to_dict(),
            },
            "certificate_verified": p0_certificate.certificate_verified and p1_certificate.certificate_verified,
        }
        per_wave = []
        for wave in waves:
            per_wave.append(
                WaveDiagnostics(
                    wave_id=int(wave.wave_id),
                    selected_flow_ids=tuple(flow.flow_id for flow in wave.flows),
                    selected_edges=tuple(
                        {"phase": flow.phase, "src_rank": flow.src_rank, "dst_rank": flow.dst_rank, "byte_count": flow.byte_count}
                        for flow in wave.flows
                    ),
                    matching_weight=float(sum(flow.byte_count for flow in wave.flows)),
                    priority_components={"service_quantum": float(wave.duration)},
                    remaining_bytes_before=float(sum(flow.byte_count for flow in wave.flows)),
                    remaining_bytes_after=0.0,
                    ready_flow_count_before=len(wave.flows),
                    blocked_flow_count_before=0,
                    forecast_pressure_summary={},
                    selection_reason="fluid BvN matching decomposition",
                )
            )
        diag = PolicyDiagnostics(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            information_mode="offline_fluid_crossbar",
            tie_break_rule="maximum real residual support, lexicographic matching",
            wave_count=len(waves),
            logical_flow_count=sum(len(wave.flows) for wave in waves),
            ready_flow_count=matrix_nonzero_remote_edge_count(problem.p0_dispatch_matrix),
            blocked_flow_count=matrix_nonzero_remote_edge_count(problem.p1_return_matrix),
            forecast_flow_count=len(problem.flow_window.forecast_pressure),
            p1_dependency_used=False,
            p2_forecast_used=False,
            p2_source=problem.forecast.source if problem.forecast is not None else "none",
            evaluation_eligible=True,
            per_wave=tuple(per_wave),
            priority_components={"reference_model": "offline_fluid_crossbar"},
        )
        return LogicalSchedulePlan(
            policy_name=self.policy_name,
            waves=waves,
            diagnostics={
                **diag.to_dict(),
                "logical_model": "offline_fluid_crossbar",
                "evaluation_model": "offline_fluid_crossbar",
                "runtime_latency_comparable": False,
                "online_executor_compatible": False,
                "certificate": certificate,
            },
        )


def decompose_fluid_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    start_wave_id: int = 0,
) -> tuple[list[LogicalWave], FluidBVNCertificate]:
    raw_matrix = matrix
    diag = matrix_diagonal_report(raw_matrix)
    matrix = canonicalize_remote_matrix(raw_matrix)
    n = len(matrix)
    residual = {
        (src, dst): int(value)
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    }
    row_loads = list(matrix_row_sums_remote(matrix))
    col_loads = list(matrix_col_sums_remote(matrix))
    horizon = max(max(row_loads, default=0), max(col_loads, default=0))
    waves: list[LogicalWave] = []
    wave_records: list[dict[str, Any]] = []
    wave_id = int(start_wave_id)
    emitted_horizon = 0
    ranks = tuple(range(n))
    while residual:
        before = sum(residual.values())

        def weight(src: int, dst: int) -> float:
            if src == dst:
                return 0.0
            return float(residual.get((src, dst), 0))

        matching = tuple(edge for edge in maximum_weight_bipartite_matching(sources=ranks, destinations=ranks, edge_weight=weight) if edge in residual)
        if not matching:
            raise ValueError("birkhoff_von_neumann_fluid could not make progress")
        quantum = min(residual[edge] for edge in matching)
        flows = []
        for src, dst in matching:
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src}->{dst}:fluid:{wave_id}",
                    phase=phase,
                    src_rank=int(src),
                    dst_rank=int(dst),
                    byte_count=int(quantum),
                    release_state="ready",
                    is_executable=True,
                )
            )
            residual[(src, dst)] -= int(quantum)
            if residual[(src, dst)] <= 0:
                residual.pop((src, dst), None)
        emitted_horizon += int(quantum)
        waves.append(LogicalWave(wave_id=wave_id, flows=tuple(flows), duration=float(quantum)))
        wave_records.append(
            {
                "wave_id": wave_id,
                "service_quantum": int(quantum),
                "real_edges": [{"src_rank": src, "dst_rank": dst, "byte_count": int(quantum)} for src, dst in matching],
                "dummy_edges": _dummy_edge_count(n, matching),
                "remaining_before": before,
                "remaining_after": sum(residual.values()),
            }
        )
        wave_id += 1
    dummy_idle = max(0, int(horizon) - int(emitted_horizon))
    certificate = FluidBVNCertificate(
        reference_model="birkhoff_von_neumann_fluid",
        num_ports=n,
        max_source_load=max(row_loads, default=0),
        max_destination_load=max(col_loads, default=0),
        fluid_optimal_horizon=int(horizon),
        emitted_real_service_horizon=int(emitted_horizon),
        dummy_idle_service_horizon=int(dummy_idle),
        coverage_verified=_coverage_verified(matrix, waves),
        matching_constraints_verified=_matching_verified(waves),
        certificate_verified=False,
        self_bytes_ignored=int(diag["self_bytes"]),
        remote_bytes=matrix_remote_bytes(matrix),
        matrix_diagonal_nonzero_count=int(diag["diagonal_nonzero_count"]),
        waves=tuple(wave_records),
    )
    certificate = FluidBVNCertificate(
        **{**certificate.to_dict(), "certificate_verified": certificate.coverage_verified and certificate.matching_constraints_verified and emitted_horizon <= horizon}
    )
    return waves, certificate


def _dummy_edge_count(n: int, matching: tuple[tuple[int, int], ...]) -> int:
    used_src = {src for src, _ in matching}
    used_dst = {dst for _, dst in matching}
    return max(0, n - max(len(used_src), len(used_dst)))


def _coverage_verified(matrix: tuple[tuple[int, ...], ...], waves: list[LogicalWave]) -> bool:
    covered: dict[tuple[int, int], int] = {}
    for wave in waves:
        for flow in wave.flows:
            covered[(flow.src_rank, flow.dst_rank)] = covered.get((flow.src_rank, flow.dst_rank), 0) + int(flow.byte_count)
    expected = {
        (src, dst): int(value)
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    }
    return covered == expected


def _matching_verified(waves: list[LogicalWave]) -> bool:
    for wave in waves:
        srcs = [flow.src_rank for flow in wave.flows]
        dsts = [flow.dst_rank for flow in wave.flows]
        if len(srcs) != len(set(srcs)) or len(dsts) != len(set(dsts)):
            return False
    return True


__all__ = ["BirkhoffVonNeumannFluidReference", "FluidBVNCertificate", "decompose_fluid_matrix"]
