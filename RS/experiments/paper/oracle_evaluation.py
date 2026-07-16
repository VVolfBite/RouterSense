from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import RecordMetadata, ScheduleEvaluationRecord
from .scheduling_evaluation import _exact_record, build_paper_execution_window_problem


@dataclass(frozen=True)
class OracleFixture:
    fixture_id: str
    rank_count: int
    bucket_rows: int
    expert_compute_delay: float
    p0_dispatch_matrix: tuple[tuple[int, ...], ...]
    p1_return_matrix: tuple[tuple[int, ...], ...]
    p2_next_dispatch_matrix: tuple[tuple[int, ...], ...]
    expected_relation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALLOWED_RELATIONS = {"joint_strictly_better", "tie", "unsupported"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _matrix(payload: Any, *, rank_count: int, name: str) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(v) for v in row) for row in payload)
    if len(rows) != int(rank_count) or any(len(row) != int(rank_count) for row in rows):
        raise ValueError(f"{name} must be {rank_count}x{rank_count}")
    if any(v < 0 for row in rows for v in row):
        raise ValueError(f"{name} must contain non-negative integers")
    return rows


def load_oracle_fixture(path: Path) -> OracleFixture:
    payload = _load_json(path)
    rank_count = int(payload["rank_count"])
    expected_relation = str(payload["expected_relation"])
    if expected_relation not in ALLOWED_RELATIONS:
        raise ValueError(f"unsupported expected_relation {expected_relation!r}")
    fixture = OracleFixture(
        fixture_id=str(payload["fixture_id"]),
        rank_count=rank_count,
        bucket_rows=int(payload["bucket_rows"]),
        expert_compute_delay=float(payload["expert_compute_delay"]),
        p0_dispatch_matrix=_matrix(payload["p0_dispatch_matrix"], rank_count=rank_count, name="p0_dispatch_matrix"),
        p1_return_matrix=_matrix(payload["p1_return_matrix"], rank_count=rank_count, name="p1_return_matrix"),
        p2_next_dispatch_matrix=_matrix(payload["p2_next_dispatch_matrix"], rank_count=rank_count, name="p2_next_dispatch_matrix"),
        expected_relation=expected_relation,
    )
    if fixture.p1_return_matrix != tuple(
        tuple(int(fixture.p0_dispatch_matrix[src][dst]) for src in range(fixture.rank_count))
        for dst in range(fixture.rank_count)
    ):
        raise ValueError("p1_return_matrix must equal transpose(p0_dispatch_matrix)")
    return fixture


def classify_oracle_relation(
    local_record: ScheduleEvaluationRecord,
    joint_record: ScheduleEvaluationRecord,
    tolerance: float,
) -> str:
    if not (local_record.comparable and joint_record.comparable):
        return "not_comparable"
    local = float(local_record.objective or 0.0)
    joint = float(joint_record.objective or 0.0)
    if joint < local - float(tolerance):
        return "joint_strictly_better"
    if abs(joint - local) <= float(tolerance):
        return "tie"
    return "dominance_violation"


def evaluate_oracle_case(
    fixture: OracleFixture,
    metadata: RecordMetadata,
    cost_model_id: str,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    if float(fixture.expert_compute_delay) != 0.0:
        unsupported_record = {
            "fixture_id": fixture.fixture_id,
            "supported": False,
            "solver_status": "unsupported_compute_delay",
            "certified_optimal": False,
            "objective": None,
            "best_bound": None,
            "optimality_gap": None,
            "plan_digest": None,
            "comparable": False,
            "coverage_valid": False,
            "validation_errors": ["unsupported_compute_delay"],
            "metadata": metadata.to_dict(),
        }
        return {
            "case_id": fixture.fixture_id,
            "status": "PASS" if fixture.expected_relation == "unsupported" else "FAIL",
            "input": fixture.to_dict(),
            "o_local": unsupported_record,
            "o_joint": unsupported_record,
            "comparison": {
                "expected_relation": fixture.expected_relation,
                "observed_relation": "not_comparable",
                "dominance_violation": False,
                "tolerance": float(tolerance),
            },
        }
    replay_window, _hint, problem = build_paper_execution_window_problem(
        fixture_id=fixture.fixture_id,
        layer_id=0,
        p0_matrix=fixture.p0_dispatch_matrix,
        p1_matrix=fixture.p1_return_matrix,
        p2_matrix=fixture.p2_next_dispatch_matrix,
        bucket_rows=int(fixture.bucket_rows),
        expert_compute_delay=float(fixture.expert_compute_delay),
    )
    local_record = _exact_record(
        instance_id=fixture.fixture_id,
        policy_id="O_local",
        meta={"heuristic_family": "oracle", "oracle_like": False},
        problem=problem,
        metadata=metadata,
        cost_model_id=cost_model_id,
        scope="local",
    )
    joint_record = _exact_record(
        instance_id=fixture.fixture_id,
        policy_id="O_joint",
        meta={"heuristic_family": "oracle", "oracle_like": False},
        problem=problem,
        metadata=metadata,
        cost_model_id=cost_model_id,
        scope="joint",
    )
    relation = classify_oracle_relation(local_record, joint_record, tolerance)
    if fixture.expected_relation == "unsupported":
        status = "PASS" if relation == "not_comparable" else "FAIL"
    else:
        status = "PASS" if relation == fixture.expected_relation else "FAIL"
    return {
        "case_id": fixture.fixture_id,
        "status": status,
        "input": {
            **fixture.to_dict(),
            "replay_window": {
                "fixture_id": replay_window.fixture_id,
                "window_id": replay_window.window_id,
                "layer_id": replay_window.layer_id,
                "group_size": replay_window.group_size,
            },
        },
        "o_local": local_record.to_dict(),
        "o_joint": joint_record.to_dict(),
        "comparison": {
            "expected_relation": fixture.expected_relation,
            "observed_relation": relation,
            "dominance_violation": relation == "dominance_violation",
            "tolerance": float(tolerance),
        },
    }


def evaluate_oracle_control_set(
    fixture_dir: Path,
    metadata: RecordMetadata,
    cost_model_id: str,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    expected_files = {
        "joint_advantage": fixture_dir / "joint_advantage.json",
        "tie": fixture_dir / "tie.json",
        "unsupported": fixture_dir / "unsupported.json",
    }
    fixtures = {name: load_oracle_fixture(path) for name, path in expected_files.items()}
    case_results = [
        evaluate_oracle_case(fixture, metadata, cost_model_id, tolerance=tolerance)
        for fixture in fixtures.values()
    ]
    cases = {
        "joint_advantage": {
            "status": next(item for item in case_results if item["case_id"] == fixtures["joint_advantage"].fixture_id)["status"],
            "observed_relation": next(item for item in case_results if item["case_id"] == fixtures["joint_advantage"].fixture_id)["comparison"]["observed_relation"],
        },
        "tie": {
            "status": next(item for item in case_results if item["case_id"] == fixtures["tie"].fixture_id)["status"],
            "observed_relation": next(item for item in case_results if item["case_id"] == fixtures["tie"].fixture_id)["comparison"]["observed_relation"],
        },
        "unsupported": {
            "status": next(item for item in case_results if item["case_id"] == fixtures["unsupported"].fixture_id)["status"],
            "observed_relation": next(item for item in case_results if item["case_id"] == fixtures["unsupported"].fixture_id)["comparison"]["observed_relation"],
        },
    }
    records: list[dict[str, Any]] = []
    dominance_violation_count = 0
    for item in case_results:
        records.extend([item["o_local"], item["o_joint"]])
        dominance_violation_count += int(bool(item["comparison"]["dominance_violation"]))
    ready = (
        cases["joint_advantage"]["status"] == "PASS"
        and cases["tie"]["status"] == "PASS"
        and cases["unsupported"]["status"] == "PASS"
        and dominance_violation_count == 0
    )
    return {
        "schema_version": "paper_oracle_control_summary.v1",
        "status": "READY" if ready else "PARTIAL",
        "final_commit": metadata.commit,
        "case_count": 3,
        "cases": cases,
        "dominance_violation_count": int(dominance_violation_count),
        "records": records,
        "case_results": case_results,
        "tolerance": float(tolerance),
    }
