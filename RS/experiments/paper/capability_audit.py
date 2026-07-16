from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ALLOWED_STATUS = {"READY", "PARTIAL", "MISSING", "SEMANTICALLY_INVALID", "ENVIRONMENT_BLOCKED"}


def _status(status: str, **kwargs: Any) -> dict[str, Any]:
    if status not in ALLOWED_STATUS:
        raise ValueError(status)
    return {"status": status, **kwargs}


def baseline_capability_matrix() -> dict[str, dict[str, Any]]:
    model_path = Path(os.environ.get("RS_MODEL_PATH", r"D:\models\OLMoE-1B-7B-0924-Instruct"))
    return {
        "trace_wrapper": _status(
            "PARTIAL" if model_path.exists() else "ENVIRONMENT_BLOCKED",
            public_entrypoint="python -m experiments.paper.cli capture-trace",
            evidence="wrapper present; requires real collector execution",
        ),
        "real_trace_ingest": _status(
            "MISSING",
            public_entrypoint="python -m experiments.paper.cli build-traffic --input <trace_bundle>",
            evidence="no executed external trace-bundle smoke yet",
        ),
        "traffic_builder": _status(
            "PARTIAL",
            public_entrypoint="python -m experiments.paper.cli build-traffic --input <trace_bundle>",
            evidence="real trace to TrafficInstance builder not yet evidenced",
        ),
        "O_local": _status(
            "SEMANTICALLY_INVALID",
            public_entrypoint="none",
            reason="no phase-local exact solver under same discrete objective as O_joint",
        ),
        "O_joint": _status(
            "PARTIAL",
            public_entrypoint="exact_small_instance_reference",
            evidence="tiny exact reference exists; replay comparable coverage still partial",
        ),
        "local_policy_registry": _status(
            "READY",
            public_entrypoint="PlannerRegistry / scheduling catalog",
            evidence="import + contract tests",
        ),
        "joint_policy_evaluation": _status(
            "PARTIAL",
            public_entrypoint="python -m experiments.paper.cli scheduling",
            evidence="fail-closed evaluator exists; invalid/unsupported cases still need wider smoke coverage",
        ),
        "predictor": _status(
            "PARTIAL",
            public_entrypoint="python -m experiments.paper.cli prediction",
            evidence="perfect/zero/shuffled baselines executable; formal predicted path missing",
        ),
        "publication_store_timing": _status(
            "PARTIAL",
            public_entrypoint="none",
            evidence="frozen public timeline extraction incomplete",
        ),
        "materialization": _status(
            "PARTIAL",
            public_entrypoint="python -m experiments.paper.cli runtime-correctness",
            evidence="materialization contract smoke only",
        ),
        "gloo_execution_wrapper": _status(
            "MISSING",
            public_entrypoint="python -m experiments.paper.cli runtime-correctness",
            evidence="real paper wrapper not yet executed",
        ),
        "executed_plan_identity": _status(
            "MISSING",
            public_entrypoint="python -m experiments.paper.cli runtime-correctness",
            evidence="no executed plan digest from runner evidence",
        ),
        "tensor_parity": _status(
            "MISSING",
            public_entrypoint="python -m experiments.paper.cli runtime-correctness",
            evidence="no paper wrapper parity evidence yet",
        ),
    }


def apply_capability_evidence(
    matrix: dict[str, dict[str, Any]],
    *,
    scheduling_summary: dict[str, Any] | None = None,
    prediction_summary: dict[str, Any] | None = None,
    runtime_summary: dict[str, Any] | None = None,
    build_traffic_summary: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    result = {key: dict(value) for key, value in matrix.items()}
    if build_traffic_summary is not None:
        status = str(build_traffic_summary.get("status", ""))
        if status == "OK":
            result["real_trace_ingest"] = _status("READY", public_entrypoint="python -m experiments.paper.cli build-traffic", evidence="external trace bundle consumed")
            result["traffic_builder"] = _status("READY", public_entrypoint="python -m experiments.paper.cli build-traffic", evidence="real compact route counts -> TrafficInstance smoke passed")
        elif status == "ENVIRONMENT_BLOCKED":
            result["real_trace_ingest"] = _status("ENVIRONMENT_BLOCKED", public_entrypoint="python -m experiments.paper.cli build-traffic", evidence="external trace bundle unavailable")
    if runtime_summary is not None:
        runtime_status = str(runtime_summary.get("status", ""))
        if runtime_status in {"MATERIALIZATION_CONTRACT_SMOKE", "PARTIAL_GLOO_ENVIRONMENT_BLOCKED", "PARTIAL_EXECUTED_DIGEST_MISSING", "RUNTIME_CORRECTNESS"}:
            result["materialization"] = _status("READY", public_entrypoint="python -m experiments.paper.cli runtime-correctness", evidence=runtime_status)
        if runtime_status in {"PARTIAL_EXECUTED_DIGEST_MISSING", "RUNTIME_CORRECTNESS"}:
            result["gloo_execution_wrapper"] = _status("PARTIAL", public_entrypoint="python -m experiments.paper.cli runtime-correctness", evidence=runtime_status)
        if runtime_status == "RUNTIME_CORRECTNESS":
            result["executed_plan_identity"] = _status("READY", public_entrypoint="python -m experiments.paper.cli runtime-correctness", evidence="executed digest matched materialized digest")
            result["tensor_parity"] = _status("READY", public_entrypoint="python -m experiments.paper.cli runtime-correctness", evidence="runtime parity passed")
        elif runtime_summary.get("tensor_parity_pass") is True:
            result["tensor_parity"] = _status("PARTIAL", public_entrypoint="python -m experiments.paper.cli runtime-correctness", evidence="tensor parity observed but executed digest missing")
    return result


def render_capability_markdown(matrix: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# CAPABILITY_AUDIT",
        "",
        "| capability | status | public_entrypoint | evidence | reason |",
        "|---|---|---|---|---|",
    ]
    for key, row in matrix.items():
        lines.append(
            f"| {key} | {row['status']} | {row.get('public_entrypoint', '')} | "
            f"{row.get('evidence', '')} | {row.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"
