from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_result_bundle(
    *,
    branch: str,
    commit: str,
    config_digest: str,
    claim_scope: str,
    status: str,
    scheduling_summary: dict[str, Any] | None,
    prediction_summary: dict[str, Any] | None,
    hiding_summary: dict[str, Any] | None,
    runtime_summary: dict[str, Any] | None,
    artifact_index: dict[str, str],
) -> dict[str, Any]:
    missing_capabilities: list[str] = []
    failure_reasons: list[str] = []
    record_counts = {
        "scheduling": len((scheduling_summary or {}).get("records", [])),
        "prediction": len((prediction_summary or {}).get("records", [])),
        "hiding": len((hiding_summary or {}).get("records", [])),
        "runtime": len((runtime_summary or {}).get("records", [])),
    }
    if prediction_summary and prediction_summary.get("status") == "PARTIAL_MISSING_PREDICTED":
        missing_capabilities.append("predicted_paper_path")
    if runtime_summary and runtime_summary.get("status") != "RUNTIME_CORRECTNESS":
        failure_reasons.append(str(runtime_summary.get("status")))
    comparable_count = len([row for row in (scheduling_summary or {}).get("records", []) if bool(row.get("comparable"))])
    invalid_count = len([row for row in (scheduling_summary or {}).get("records", []) if not bool(row.get("comparable", False))])
    return {
        "schema_version": "paper_result_bundle.v1",
        "run_identity": {
            "branch": branch,
            "commit": commit,
            "config_digest": config_digest,
            "claim_scope": claim_scope,
        },
        "status": status,
        "correctness_eligibility": True,
        "performance_eligibility": False,
        "runtime_correctness_eligible": bool(runtime_summary and runtime_summary.get("status") == "RUNTIME_CORRECTNESS"),
        "record_counts": record_counts,
        "comparable_count": comparable_count,
        "invalid_count": invalid_count,
        "missing_capabilities": missing_capabilities,
        "failure_reasons": failure_reasons,
        "artifact_index": artifact_index,
    }
