"""Result-envelope contracts for RouteSense experiments and runtime outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


OFFLINE_PIPELINE = "offline"
ONLINE_PIPELINE = "online"
LEGACY_TRACE_REPLAY_PIPELINE = "legacy"
RESULT_BUNDLE_SCHEMA_VERSION = "result_bundle.v2"
ALLOWED_INSTRUMENTATION_MODES = {"off", "contract", "perf_light", "debug"}
ALLOWED_RESULT_STATUSES = {"success", "failure", "invalid"}
ALLOWED_CORRECTNESS_STATUSES = {"valid", "invalid", "unknown"}
ALLOWED_PERFORMANCE_STATUSES = {"eligible", "ineligible", "unknown"}
ALLOWED_AUDIT_LEVELS = {"full", "summary_only", "unavailable"}
RESERVED_RESULT_EXTENSION_KEYS = {
    "schema_version",
    "run_identity",
    "status",
    "correctness_status",
    "performance_status",
    "pipeline",
    "commit_sha",
    "git_clean",
    "instrumentation_mode",
    "audit_evidence_level",
    "measurement_complete",
    "eligibility",
    "summary",
    "details",
    "extensions",
}


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    pipeline: str
    claim_scope: str
    trace_origin: str
    future_information_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RunIdentity":
        return cls(
            run_id=str(payload.get("run_id", "")),
            pipeline=str(payload.get("pipeline", "")),
            claim_scope=str(payload.get("claim_scope", "")),
            trace_origin=str(payload.get("trace_origin", "")),
            future_information_mode=str(payload.get("future_information_mode", "")),
        )

    def validate(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("run_identity.run_id must be non-empty")
        if not str(self.pipeline).strip():
            raise ValueError("run_identity.pipeline must be non-empty")
        if not str(self.claim_scope).strip():
            raise ValueError("run_identity.claim_scope must be non-empty")
        if not str(self.trace_origin).strip():
            raise ValueError("run_identity.trace_origin must be non-empty")
        if not str(self.future_information_mode).strip():
            raise ValueError("run_identity.future_information_mode must be non-empty")


@dataclass(frozen=True)
class EligibilityResult:
    correctness_eligible: bool
    performance_eligible: bool
    prediction_evaluation_eligible: bool
    offline_replay_eligible: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EligibilityResult":
        return cls(
            correctness_eligible=bool(payload.get("correctness_eligible", False)),
            performance_eligible=bool(payload.get("performance_eligible", False)),
            prediction_evaluation_eligible=bool(payload.get("prediction_evaluation_eligible", False)),
            offline_replay_eligible=bool(payload.get("offline_replay_eligible", False)),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
        )


@dataclass(frozen=True)
class ResultBundle:
    run_identity: RunIdentity
    status: str
    eligibility: EligibilityResult | None
    schema_version: str = RESULT_BUNDLE_SCHEMA_VERSION
    correctness_status: str = "unknown"
    performance_status: str = "unknown"
    pipeline: str = ""
    commit_sha: str = ""
    git_clean: bool | None = None
    instrumentation_mode: str = "off"
    audit_evidence_level: str = "unavailable"
    measurement_complete: bool | None = None
    summary: Mapping[str, object] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)
    extensions: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        self.run_identity.validate()
        if str(self.schema_version) != RESULT_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported result bundle schema_version")
        if str(self.status) not in ALLOWED_RESULT_STATUSES:
            raise ValueError("result bundle status is invalid")
        if str(self.correctness_status) not in ALLOWED_CORRECTNESS_STATUSES:
            raise ValueError("correctness_status is invalid")
        if str(self.performance_status) not in ALLOWED_PERFORMANCE_STATUSES:
            raise ValueError("performance_status is invalid")
        if not str(self.pipeline or self.run_identity.pipeline).strip():
            raise ValueError("pipeline must be non-empty")
        if not str(self.commit_sha).strip():
            raise ValueError("commit_sha must be non-empty")
        if self.git_clean is None:
            raise ValueError("git_clean must be explicit")
        if str(self.instrumentation_mode) not in ALLOWED_INSTRUMENTATION_MODES:
            raise ValueError("instrumentation_mode is invalid")
        if str(self.audit_evidence_level) not in ALLOWED_AUDIT_LEVELS:
            raise ValueError("audit_evidence_level is invalid")
        if "all_work_completed" not in self.summary:
            raise ValueError("summary must include all_work_completed")
        if "fallback_count" not in self.summary:
            raise ValueError("summary must include fallback_count")
        if "timeout_count" not in self.summary:
            raise ValueError("summary must include timeout_count")
        if "check_failure_count" not in self.summary:
            raise ValueError("summary must include check_failure_count")
        for key in self.extensions:
            if str(key) in RESERVED_RESULT_EXTENSION_KEYS:
                raise ValueError(f"extensions key conflicts with reserved field: {key}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ResultBundle":
        run_identity = RunIdentity.from_dict(payload.get("run_identity", {}))
        eligibility_payload = payload.get("eligibility")
        bundle = cls(
            run_identity=run_identity,
            status=str(payload.get("status", "")),
            eligibility=EligibilityResult.from_dict(eligibility_payload) if isinstance(eligibility_payload, Mapping) else None,
            schema_version=str(payload.get("schema_version", "")),
            correctness_status=str(payload.get("correctness_status", "unknown")),
            performance_status=str(payload.get("performance_status", "unknown")),
            pipeline=str(payload.get("pipeline", run_identity.pipeline)),
            commit_sha=str(payload.get("commit_sha", "")),
            git_clean=payload.get("git_clean") if isinstance(payload.get("git_clean"), bool) else None,
            instrumentation_mode=str(payload.get("instrumentation_mode", "off")),
            audit_evidence_level=str(payload.get("audit_evidence_level", "unavailable")),
            measurement_complete=payload.get("measurement_complete") if isinstance(payload.get("measurement_complete"), bool) else None,
            summary=dict(payload.get("summary", {})) if isinstance(payload.get("summary"), Mapping) else {},
            details=dict(payload.get("details", {})) if isinstance(payload.get("details"), Mapping) else {},
            extensions=dict(payload.get("extensions", {})) if isinstance(payload.get("extensions"), Mapping) else {},
        )
        bundle.validate()
        return bundle

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": str(self.schema_version),
            "run_identity": self.run_identity.to_dict(),
            "status": str(self.status),
            "correctness_status": str(self.correctness_status),
            "performance_status": str(self.performance_status),
            "pipeline": str(self.pipeline or self.run_identity.pipeline),
            "commit_sha": str(self.commit_sha),
            "git_clean": bool(self.git_clean),
            "instrumentation_mode": str(self.instrumentation_mode),
            "audit_evidence_level": str(self.audit_evidence_level),
            "measurement_complete": bool(self.measurement_complete),
            "eligibility": None if self.eligibility is None else self.eligibility.to_dict(),
            "summary": dict(self.summary),
            "details": dict(self.details),
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    schema_digest: str
    artifact_count: int
    artifact_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_result_envelope(
    *,
    run_id: str,
    pipeline: str,
    claim_scope: str,
    trace_origin: str,
    future_information_mode: str,
    is_real_ep_runtime: bool,
    source_ownership_mode: str,
    expert_residency_mode: str,
    transport_backend: str,
    correctness_status: str,
    performance_claim_eligible: bool,
    execution_mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if extra:
        conflicting = RESERVED_RESULT_EXTENSION_KEYS.intersection(str(key) for key in extra)
        if conflicting:
            raise ValueError(f"extra keys conflict with reserved result fields: {sorted(conflicting)}")
    payload = {
        "run_identity": RunIdentity(
            run_id=run_id,
            pipeline=pipeline,
            claim_scope=claim_scope,
            trace_origin=trace_origin,
            future_information_mode=future_information_mode,
        ).to_dict(),
        "pipeline": pipeline,
        "claim_scope": claim_scope,
        "trace_origin": trace_origin,
        "future_information_mode": future_information_mode,
        "is_real_ep_runtime": is_real_ep_runtime,
        "source_ownership_mode": source_ownership_mode,
        "expert_residency_mode": expert_residency_mode,
        "transport_backend": transport_backend,
        "correctness_status": correctness_status,
        "performance_claim_eligible": performance_claim_eligible,
    }
    if execution_mode is not None:
        payload["execution_mode"] = execution_mode
    if extra:
        payload["extensions"] = dict(extra)
    return payload


__all__ = [
    "ArtifactManifest",
    "EligibilityResult",
    "LEGACY_TRACE_REPLAY_PIPELINE",
    "OFFLINE_PIPELINE",
    "ONLINE_PIPELINE",
    "ResultBundle",
    "RunIdentity",
    "build_result_envelope",
]
