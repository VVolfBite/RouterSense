from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from rs.planning import PlannerRegistry
from rs.prediction import PredictionRegistry
from rs.scheduling.algorithm_catalog import pair_status_summary


ALLOWED_STATUS = {"READY", "PARTIAL", "MISSING", "SEMANTICALLY_INVALID", "ENVIRONMENT_BLOCKED"}


def _status(status: str, **kwargs: Any) -> dict[str, Any]:
    if status not in ALLOWED_STATUS:
        raise ValueError(status)
    return {"status": status, **kwargs}


def run_capability_audit(*, repo_root: Path) -> dict[str, Any]:
    model_path = Path(os.environ.get("RS_MODEL_PATH", r"D:\models\OLMoE-1B-7B-0924-Instruct"))
    pair_summary = pair_status_summary()
    rows = {
        "trace": _status(
            "READY" if model_path.exists() else "ENVIRONMENT_BLOCKED",
            public_api="experiments.offline.collect_router_trace + rs.runtime.offline.trace.olmoe",
            requires_formal_code_change=False,
            missing_public_api=None if model_path.exists() else "model path not available",
            suggested_milestone=None if model_path.exists() else "restore single-GPU model path",
        ),
        "traffic_builder": _status(
            "READY",
            public_api="rs.runtime.offline.replay_unified",
            requires_formal_code_change=False,
        ),
        "O_local": _status(
            "READY",
            public_api="birkhoff_von_neumann_fluid via resolve_policy / replay_unified",
            requires_formal_code_change=False,
        ),
        "O_joint": _status(
            "PARTIAL",
            public_api="exact_small_instance_reference small-instance exact reference only",
            requires_formal_code_change=False,
            suggested_milestone="promote real-trace exact comparable oracle path",
        ),
        "local_policies": _status(
            "READY" if pair_summary["ready_pair_count"] > 0 else "PARTIAL",
            public_api="rs.scheduling.algorithm_catalog + PlannerRegistry",
            requires_formal_code_change=False,
        ),
        "joint_policies": _status(
            "READY" if pair_summary["ready_pair_count"] > 0 else "PARTIAL",
            public_api="rs.scheduling.algorithm_catalog + PlannerRegistry",
            requires_formal_code_change=False,
        ),
        "predictor": _status(
            "PARTIAL",
            public_api="PredictionRegistry zero/copy_current/history available; formal predicted paper path not frozen",
            requires_formal_code_change=False,
            suggested_milestone="promote non-oracle predictor with no-future-leakage contract",
        ),
        "publication_store_timing": _status(
            "PARTIAL",
            public_api="CanonicalPlanPublisher/TargetPlanStore types exist but frozen public timeline extraction is incomplete",
            requires_formal_code_change=True,
            suggested_milestone="expose stable publication/store timestamps through public measurement API",
        ),
        "materialization": _status(
            "READY",
            public_api="CommonPlanMaterializer/CommonPlanValidator",
            requires_formal_code_change=False,
        ),
        "gloo_execution": _status(
            "PARTIAL",
            public_api="distributed Gloo runners exist; paper harness keeps runtime correctness smoke at single-process level this round",
            requires_formal_code_change=False,
            suggested_milestone="wire 4-rank paper runtime-correctness wrapper",
        ),
        "plan_identity": _status(
            "READY",
            public_api="PublishedPlan/MaterializedPlan digests + validator contracts",
            requires_formal_code_change=False,
        ),
        "tensor_parity": _status(
            "PARTIAL",
            public_api="execution/runtime unit coverage exists; paper harness smoke does not yet execute 4-rank tensor replay",
            requires_formal_code_change=False,
            suggested_milestone="add Gloo tensor replay wrapper under experiments.paper.runtime-correctness",
        ),
    }
    return rows


def render_capability_markdown(matrix: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# CAPABILITY_AUDIT",
        "",
        "| capability | status | public_api | requires_formal_code_change | suggested_milestone |",
        "|---|---|---|---|---|",
    ]
    for key, row in matrix.items():
        lines.append(
            f"| {key} | {row['status']} | {row.get('public_api', '')} | "
            f"{row.get('requires_formal_code_change', '')} | {row.get('suggested_milestone', '')} |"
        )
    return "\n".join(lines) + "\n"
