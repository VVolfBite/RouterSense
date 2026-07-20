"""Prediction artifact helpers for the formal offline runtime."""

from __future__ import annotations

from typing import Any

from rs.core.contracts.result import EligibilityResult, OFFLINE_PIPELINE, ResultBundle, RunIdentity
from rs.core.contracts.trace import FutureInformationMode, TraceOrigin
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle


def build_calibrated_counterfactual_result(
    *,
    run_id: str,
    future_information_mode: str,
    extra: dict[str, Any] | None = None,
) -> ResultBundle:
    details = {
        "is_real_ep_runtime": False,
        "source_ownership_mode": "observed_online_native_ep",
        "expert_residency_mode": "observed_online_native_ep",
        "transport_backend": "calibrated_simulator",
        "deployable_scheduler_candidate": future_information_mode != FutureInformationMode.ORACLE_FULL_TRACE,
        **(extra or {}),
    }
    return build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id=run_id,
                pipeline=OFFLINE_PIPELINE,
                claim_scope="calibrated_offline_counterfactual",
                trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
                future_information_mode=future_information_mode,
            ),
            status="invalid",
            correctness_status="invalid",
            performance_status="ineligible",
            commit_sha="legacy-offline-artifact",
            git_clean=False,
            instrumentation_mode="off",
            audit_evidence_level="summary_only",
            measurement_complete=False,
            summary={
                "all_work_completed": False,
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "execution_outcome_count": 0,
                "prediction_evaluation_complete": False,
                "offline_replay_complete": False,
            },
            details=details,
            extensions={},
        )
    )


__all__ = ["build_calibrated_counterfactual_result"]
