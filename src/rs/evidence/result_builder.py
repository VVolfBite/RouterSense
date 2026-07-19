from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rs.core.contracts.result import EligibilityResult
from rs.core.contracts.result import ResultBundle, RunIdentity
from rs.evidence.eligibility import evaluate_result_bundle_eligibility


@dataclass(frozen=True)
class ResultBundleDraft:
    run_identity: RunIdentity
    status: str
    correctness_status: str
    performance_status: str
    commit_sha: str
    git_clean: bool
    instrumentation_mode: str
    audit_evidence_level: str
    measurement_complete: bool
    summary: Mapping[str, object]
    details: Mapping[str, object]
    extensions: Mapping[str, object]


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be an explicit boolean")
    return value


def build_result_bundle(draft: ResultBundleDraft) -> ResultBundle:
    placeholder = EligibilityResult(
        correctness_eligible=False,
        performance_eligible=False,
        prediction_evaluation_eligible=False,
        offline_replay_eligible=False,
        preparation_claim_eligible=False,
        correctness_reasons=("pending_evaluation",),
        performance_reasons=("pending_evaluation",),
        prediction_reasons=("pending_evaluation",),
        offline_replay_reasons=("pending_evaluation",),
        preparation_claim_reasons=("pending_evaluation",),
    )
    provisional = ResultBundle(
        run_identity=draft.run_identity,
        status=str(draft.status),
        correctness_status=str(draft.correctness_status),
        performance_status=str(draft.performance_status),
        pipeline=str(draft.run_identity.pipeline),
        commit_sha=str(draft.commit_sha),
        git_clean=_require_bool(draft.git_clean, field_name="git_clean"),
        instrumentation_mode=str(draft.instrumentation_mode),
        audit_evidence_level=str(draft.audit_evidence_level),
        measurement_complete=_require_bool(draft.measurement_complete, field_name="measurement_complete"),
        eligibility=placeholder,
        summary=dict(draft.summary),
        details=dict(draft.details),
        extensions=dict(draft.extensions),
    )
    final_eligibility = evaluate_result_bundle_eligibility(provisional)
    finalized = ResultBundle(
        run_identity=provisional.run_identity,
        status=provisional.status,
        correctness_status=provisional.correctness_status,
        performance_status="eligible" if final_eligibility.performance_eligible else "ineligible",
        pipeline=provisional.pipeline,
        commit_sha=provisional.commit_sha,
        git_clean=provisional.git_clean,
        instrumentation_mode=provisional.instrumentation_mode,
        audit_evidence_level=provisional.audit_evidence_level,
        measurement_complete=provisional.measurement_complete,
        eligibility=final_eligibility,
        summary=provisional.summary,
        details=provisional.details,
        extensions=provisional.extensions,
    )
    finalized.validate()
    return finalized
