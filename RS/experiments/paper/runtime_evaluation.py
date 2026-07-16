from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan

from .adapters.execution_adapter import actual_phase_context_from_ready_context, build_phase_contexts_from_matrix
from .adapters.publication_adapter import materialize_and_validate, publish_plan
from .contracts import RecordMetadata, RuntimeEvaluationRecord


def _window_plan() -> WindowPlan:
    return WindowPlan(
        planner_id="paper_materialization_contract_smoke",
        planner_family="joint",
        request_digest="paper-runtime:0->1",
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(
                        flow_id="p0_0_1",
                        phase="p0_dispatch",
                        src_rank=0,
                        dst_rank=1,
                        row_count=4,
                        release_state="ready",
                        executable=True,
                    ),
                ),
                estimated_duration=4.0,
            ),
        ),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )


def evaluate_materialization_contract_smoke(*, metadata: RecordMetadata) -> dict[str, Any]:
    contexts = build_phase_contexts_from_matrix(phase="P0", matrix=((0, 4), (0, 0)))
    actual_context = actual_phase_context_from_ready_context(contexts[0], phase="P0")
    published = publish_plan(window_plan=_window_plan(), world_size=2)
    materialized, validation = materialize_and_validate(
        published_plan=published,
        actual_context=actual_context,
    )
    submitted = sum(len(batch.slices) for batch in materialized.batches)
    record = RuntimeEvaluationRecord(
        instance_id="paper-materialization-contract-smoke",
        requested_policy_id=None,
        selected_policy_id=None,
        published_plan_digest=str(published.published_plan_digest),
        materialized_plan_digest=str(materialized.materialized_plan_digest),
        executed_plan_digest=None,
        execution_backend_id=None,
        submitted_tasks=int(submitted),
        completed_tasks=None,
        unresolved_tasks=None,
        fallback_count=None,
        reference_output_digest=None,
        executed_output_digest=None,
        parity_status="NOT_EXECUTED",
        communication_makespan_ms=None,
        visible_control_ms=None,
        runtime_status="MATERIALIZATION_CONTRACT_SMOKE" if validation.valid else "INVALID",
        metadata=metadata,
        evidence={"validation": validation.to_dict()},
    )
    return {
        "status": "MATERIALIZATION_CONTRACT_SMOKE" if validation.valid else "INVALID",
        "records": [record.to_dict()],
        "validation": validation.to_dict(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_runner_script() -> Path:
    return Path(__file__).resolve().parents[1] / "distributed" / "run_m123_integrated_publication_execution_gloo.py"


def _runtime_status_from_summary(summary: dict[str, Any], parity: dict[str, Any]) -> str:
    if summary.get("executed_task_manifest_digest") in {None, ""}:
        return "EXECUTED_DIGEST_MISSING"
    if summary.get("materialized_task_manifest_digest") in {None, ""}:
        return "MATERIALIZED_DIGEST_MISSING"
    if summary.get("executed_task_manifest_digest") != summary.get("materialized_task_manifest_digest"):
        return "PLAN_IDENTITY_MISMATCH"
    if not bool(parity.get("allclose", False)):
        return "TENSOR_PARITY_FAILED"
    submitted = sum(int(rank.get("submitted_task_count", 0) or 0) for rank in summary.get("ranks", []))
    completed = sum(int(rank.get("completed_task_count", 0) or 0) for rank in summary.get("ranks", []))
    unresolved = sum(int(rank.get("unresolved_task_count", 0) or 0) for rank in summary.get("ranks", []))
    if submitted != completed or unresolved != 0:
        return "INCOMPLETE_TASKS"
    fallback = sum(int(rank.get("phase_sync_fallback_count", 0) or 0) for rank in summary.get("ranks", []))
    if fallback != 0:
        return "FALLBACK_OCCURRED"
    return "RUNTIME_CORRECTNESS"


def _run_formal_gloo_backend(
    *,
    metadata: RecordMetadata,
    execution_backend: str,
    policy_name: str,
    matrix_bundle_path: str = "",
    trace_sample_id: str | None = None,
    traffic_instance_id: str | None = None,
) -> RuntimeEvaluationRecord:
    script = _formal_runner_script()
    with tempfile.TemporaryDirectory(prefix=f"rs_paper_{execution_backend}_") as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "summary.json"
        output_dir = tmp / "runner"
        command = [
            sys.executable,
            str(script),
            "--quiet",
            "--execution-backend",
            str(execution_backend),
            "--policy-name",
            str(policy_name),
            "--output-dir",
            str(output_dir),
            "--summary-path",
            str(summary_path),
        ]
        if str(matrix_bundle_path).strip():
            command.extend(["--matrix-bundle", str(matrix_bundle_path)])
        proc = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return RuntimeEvaluationRecord(
                instance_id=f"paper-gloo-runtime-wrapper:{execution_backend}",
                requested_policy_id=str(policy_name),
                selected_policy_id=str(policy_name),
                published_plan_digest=None,
                materialized_plan_digest=None,
                executed_plan_digest=None,
                execution_backend_id=str(execution_backend),
                submitted_tasks=None,
                completed_tasks=None,
                unresolved_tasks=None,
                fallback_count=None,
                reference_output_digest=None,
                executed_output_digest=None,
                parity_status="NOT_EXECUTED",
                communication_makespan_ms=None,
                visible_control_ms=None,
                runtime_status="ENVIRONMENT_BLOCKED",
                metadata=metadata,
                evidence={"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode},
            )
        summary = _load_json(summary_path)
        run_dir = Path(str(summary["run_dir"]))
        parity_path = Path(str(summary["parity_path"]))
        executed_manifest_path = Path(str(summary["executed_task_manifest_path"]))
        materialized_manifest_path = Path(str(summary["materialized_task_manifest_path"]))
        parity = _load_json(parity_path) if parity_path.exists() else {"allclose": False, "status": "PARITY_ARTIFACT_MISSING"}
        runtime_status = _runtime_status_from_summary(summary, parity) if parity_path.exists() else "PARITY_ARTIFACT_MISSING"
        submitted = sum(int(rank.get("submitted_task_count", 0) or 0) for rank in summary.get("ranks", []))
        completed = sum(int(rank.get("completed_task_count", 0) or 0) for rank in summary.get("ranks", []))
        unresolved = sum(int(rank.get("unresolved_task_count", 0) or 0) for rank in summary.get("ranks", []))
        fallback = sum(int(rank.get("phase_sync_fallback_count", 0) or 0) for rank in summary.get("ranks", []))
        return RuntimeEvaluationRecord(
            instance_id=f"paper-gloo-runtime-wrapper:{execution_backend}",
            requested_policy_id=str(policy_name),
            selected_policy_id=str(policy_name),
            published_plan_digest=str(summary["ranks"][0].get("published_plan_digest")) if summary.get("ranks") else None,
            materialized_plan_digest=str(summary.get("materialized_task_manifest_digest")) if summary.get("materialized_task_manifest_digest") is not None else None,
            executed_plan_digest=str(summary.get("executed_task_manifest_digest")) if summary.get("executed_task_manifest_digest") is not None else None,
            execution_backend_id=str(summary.get("execution_backend")),
            submitted_tasks=submitted,
            completed_tasks=completed,
            unresolved_tasks=unresolved,
            fallback_count=fallback,
            reference_output_digest=str(parity.get("reference_final_digest")) if parity.get("reference_final_digest") is not None else None,
            executed_output_digest=str(parity.get("executed_final_digest")) if parity.get("executed_final_digest") is not None else None,
            parity_status=str(parity.get("status", "PARITY_ARTIFACT_MISSING")),
            communication_makespan_ms=None,
            visible_control_ms=None,
            runtime_status=runtime_status,
            metadata=metadata,
            evidence={
                "summary_path": str(summary_path),
                "run_dir": str(run_dir),
                "runner_summary": summary,
                "parity_path": str(parity_path),
                "executed_task_manifest_path": str(executed_manifest_path),
                "materialized_task_manifest_path": str(materialized_manifest_path),
                "trace_sample_id": trace_sample_id,
                "traffic_instance_id": traffic_instance_id,
            },
        )


def evaluate_runtime_correctness_with_gloo(
    *,
    metadata: RecordMetadata,
    policy_name: str = "U_barrier_criticality_global_matching",
    matrix_bundle_path: str = "",
    trace_sample_id: str | None = None,
    traffic_instance_id: str | None = None,
) -> dict[str, Any]:
    phase_sync = _run_formal_gloo_backend(
        metadata=metadata,
        execution_backend="phase_sync",
        policy_name=policy_name,
        matrix_bundle_path=matrix_bundle_path,
        trace_sample_id=trace_sample_id,
        traffic_instance_id=traffic_instance_id,
    )
    async_release = _run_formal_gloo_backend(
        metadata=metadata,
        execution_backend="async_release",
        policy_name=policy_name,
        matrix_bundle_path=matrix_bundle_path,
        trace_sample_id=trace_sample_id,
        traffic_instance_id=traffic_instance_id,
    )
    statuses = {phase_sync.runtime_status, async_release.runtime_status}
    overall_status = "RUNTIME_CORRECTNESS" if statuses == {"RUNTIME_CORRECTNESS"} else sorted(statuses)[0]
    tensor_parity_pass = phase_sync.parity_status == "PASS" and async_release.parity_status == "PASS"
    return {
        "status": overall_status,
        "records": [phase_sync.to_dict(), async_release.to_dict()],
        "tensor_parity_pass": tensor_parity_pass,
    }

