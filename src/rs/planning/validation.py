from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanningRequest, WindowPlan


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    reasons: tuple[str, ...]


def _matrix_entries(matrix: tuple[tuple[int, ...], ...], *, phase: str) -> dict[tuple[str, int, int], int]:
    entries: dict[tuple[str, int, int], int] = {}
    for src_rank, row in enumerate(matrix):
        for dst_rank, row_count in enumerate(row):
            if src_rank == dst_rank or int(row_count) <= 0:
                continue
            entries[(str(phase), int(src_rank), int(dst_rank))] = int(row_count)
    return entries


def _expected_executable_coverage(request: PlanningRequest) -> dict[tuple[str, int, int], int]:
    coverage = _matrix_entries(request.traffic.p0_dispatch_rows, phase="p0_dispatch")
    coverage.update(_matrix_entries(request.traffic.p1_return_rows, phase="p1_return"))
    if str(request.p2_semantics) == "executable_actual":
        coverage.update(_matrix_entries(request.prediction_hint.target_dispatch_rows, phase="p2_next_dispatch"))
    return coverage


def _expected_advisory_coverage(request: PlanningRequest) -> dict[tuple[str, int, int], int]:
    if str(request.p2_semantics) != "advisory_hint":
        return {}
    return _matrix_entries(request.prediction_hint.target_dispatch_rows, phase="p2_next_dispatch_forecast")


def _actual_plan_coverage(plan: WindowPlan, *, executable: bool) -> dict[tuple[str, int, int], int]:
    coverage: dict[tuple[str, int, int], int] = {}
    for wave in plan.waves:
        for flow in wave.flows:
            if bool(flow.executable) != bool(executable):
                continue
            key = (str(flow.phase), int(flow.src_rank), int(flow.dst_rank))
            coverage[key] = int(coverage.get(key, 0)) + int(flow.row_count)
    return coverage


def _compare_coverage(
    *,
    expected: dict[tuple[str, int, int], int],
    actual: dict[tuple[str, int, int], int],
    missing_reason: str,
    extra_reason: str,
    duplicate_reason: str,
) -> None:
    for key, expected_rows in expected.items():
        actual_rows = int(actual.get(key, 0))
        if actual_rows < int(expected_rows):
            raise ValueError(f"{missing_reason}:{key}:{actual_rows}/{expected_rows}")
        if actual_rows > int(expected_rows):
            raise ValueError(f"{duplicate_reason}:{key}:{actual_rows}/{expected_rows}")
    for key, actual_rows in actual.items():
        if key not in expected and int(actual_rows) > 0:
            raise ValueError(f"{extra_reason}:{key}:{actual_rows}")


def validate_window_plan_for_request(plan: WindowPlan, request: PlanningRequest) -> None:
    request.validate()
    plan.validate()
    if str(plan.request_digest) != str(request.semantic_digest()):
        raise ValueError("request_digest_mismatch")
    world_size = int(request.topology.world_size)
    wave_ids = [int(wave.wave_id) for wave in plan.waves]
    if len(set(wave_ids)) != len(wave_ids):
        raise ValueError("duplicate_wave_id")
    if wave_ids != sorted(wave_ids):
        raise ValueError("wave_ids_not_sorted")
    if len(plan.waves) > int(request.constraints.max_waves):
        raise ValueError("max_waves_exceeded")
    allowed_phases = {
        "p0_only": {"p0_dispatch"},
        "p0_p1": {"p0_dispatch", "p1_return"},
        "p0_p1_p2": {"p0_dispatch", "p1_return", "p2_next_dispatch_forecast", "p2_next_dispatch"},
    }[str(request.information_mode)]
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        for flow in wave.flows:
            if not 0 <= int(flow.src_rank) < world_size:
                raise ValueError("src_rank_out_of_range")
            if not 0 <= int(flow.dst_rank) < world_size:
                raise ValueError("dst_rank_out_of_range")
            if str(flow.phase) not in allowed_phases:
                raise ValueError("phase_not_allowed_for_information_mode")
            if str(request.p2_semantics) == "absent" and str(flow.phase).startswith("p2_"):
                raise ValueError("p2_present_when_absent")
            if str(request.p2_semantics) == "advisory_hint" and str(flow.phase) == "p2_next_dispatch" and bool(flow.executable):
                raise ValueError("advisory_p2_cannot_be_executable")
            if str(request.p2_semantics) == "executable_actual" and str(flow.phase) == "p2_next_dispatch_forecast":
                raise ValueError("execution_window_cannot_use_forecast_p2")
            if bool(flow.executable):
                if int(flow.src_rank) in used_src:
                    raise ValueError("multiple_outgoing_in_wave")
                if int(flow.dst_rank) in used_dst:
                    raise ValueError("multiple_incoming_in_wave")
                used_src.add(int(flow.src_rank))
                used_dst.add(int(flow.dst_rank))
    _compare_coverage(
        expected=_expected_executable_coverage(request),
        actual=_actual_plan_coverage(plan, executable=True),
        missing_reason="missing_executable_coverage",
        extra_reason="unexpected_executable_flow",
        duplicate_reason="duplicate_executable_coverage",
    )
    advisory_actual = _actual_plan_coverage(plan, executable=False)
    if advisory_actual:
        _compare_coverage(
            expected=_expected_advisory_coverage(request),
            actual=advisory_actual,
            missing_reason="missing_advisory_coverage",
            extra_reason="unexpected_advisory_flow",
            duplicate_reason="duplicate_advisory_coverage",
        )
