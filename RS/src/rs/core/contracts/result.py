"""Canonical result contracts for RouterSense runtime and experiment outputs."""

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
    preparation_claim_eligible: bool = False
    correctness_reasons: tuple[str, ...] = ()
    performance_reasons: tuple[str, ...] = ()
    prediction_reasons: tuple[str, ...] = ()
    offline_replay_reasons: tuple[str, ...] = ()
    preparation_claim_reasons: tuple[str, ...] = ()

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(
            list(self.correctness_reasons)
            + [f"performance:{item}" for item in self.performance_reasons]
            + [f"prediction:{item}" for item in self.prediction_reasons]
            + [f"offline:{item}" for item in self.offline_replay_reasons]
            + [f"preparation:{item}" for item in self.preparation_claim_reasons]
        )

    def validate(self) -> None:
        checks = (
            (self.correctness_eligible, self.correctness_reasons, "correctness"),
            (self.performance_eligible, self.performance_reasons, "performance"),
            (self.prediction_evaluation_eligible, self.prediction_reasons, "prediction"),
            (self.offline_replay_eligible, self.offline_replay_reasons, "offline_replay"),
            (self.preparation_claim_eligible, self.preparation_claim_reasons, "preparation_claim"),
        )
        for eligible, reasons, label in checks:
            normalized = tuple(str(item) for item in reasons)
            if bool(eligible) and normalized:
                raise ValueError(f"{label} eligible cannot carry rejection reasons")
            if any(not item.strip() for item in normalized):
                raise ValueError(f"{label} reasons must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EligibilityResult":
        legacy_reasons = tuple(str(item) for item in payload.get("reasons", ()))
        result = cls(
            correctness_eligible=bool(payload.get("correctness_eligible", False)),
            performance_eligible=bool(payload.get("performance_eligible", False)),
            prediction_evaluation_eligible=bool(payload.get("prediction_evaluation_eligible", False)),
            offline_replay_eligible=bool(payload.get("offline_replay_eligible", False)),
            preparation_claim_eligible=bool(payload.get("preparation_claim_eligible", False)),
            correctness_reasons=tuple(str(item) for item in payload.get("correctness_reasons", ())),
            performance_reasons=tuple(str(item) for item in payload.get("performance_reasons", ())),
            prediction_reasons=tuple(str(item) for item in payload.get("prediction_reasons", ())),
            offline_replay_reasons=tuple(str(item) for item in payload.get("offline_replay_reasons", ())),
            preparation_claim_reasons=tuple(str(item) for item in payload.get("preparation_claim_reasons", ())),
        )
        if legacy_reasons and not result.reasons:
            correctness: list[str] = []
            performance: list[str] = []
            prediction: list[str] = []
            offline: list[str] = []
            preparation: list[str] = []
            for item in legacy_reasons:
                if item.startswith("performance:"):
                    performance.append(item.split(":", 1)[1])
                elif item.startswith("prediction:"):
                    prediction.append(item.split(":", 1)[1])
                elif item.startswith("offline:"):
                    offline.append(item.split(":", 1)[1])
                elif item.startswith("preparation:"):
                    preparation.append(item.split(":", 1)[1])
                else:
                    correctness.append(item)
            result = cls(
                correctness_eligible=result.correctness_eligible,
                performance_eligible=result.performance_eligible,
                prediction_evaluation_eligible=result.prediction_evaluation_eligible,
                offline_replay_eligible=result.offline_replay_eligible,
                preparation_claim_eligible=result.preparation_claim_eligible,
                correctness_reasons=tuple(correctness),
                performance_reasons=tuple(performance),
                prediction_reasons=tuple(prediction),
                offline_replay_reasons=tuple(offline),
                preparation_claim_reasons=tuple(preparation),
            )
        result.validate()
        return result


@dataclass(frozen=True)
class ResultBundle:
    run_identity: RunIdentity
    status: str
    eligibility: EligibilityResult
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
        self.eligibility.validate()
        if str(self.schema_version) != RESULT_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported result bundle schema_version")
        if str(self.status) not in ALLOWED_RESULT_STATUSES:
            raise ValueError("result bundle status is invalid")
        if str(self.correctness_status) not in ALLOWED_CORRECTNESS_STATUSES:
            raise ValueError("correctness_status is invalid")
        if str(self.performance_status) not in ALLOWED_PERFORMANCE_STATUSES:
            raise ValueError("performance_status is invalid")
        if str(self.pipeline or self.run_identity.pipeline).strip() != str(self.run_identity.pipeline):
            raise ValueError("pipeline must exactly match run_identity.pipeline")
        commit_sha = str(self.commit_sha).strip()
        if not commit_sha:
            raise ValueError("commit_sha must be non-empty")
        if commit_sha.lower() == "unknown":
            raise ValueError("commit_sha must not be unknown")
        if self.git_clean is None:
            raise ValueError("git_clean must be explicit")
        if str(self.instrumentation_mode) not in ALLOWED_INSTRUMENTATION_MODES:
            raise ValueError("instrumentation_mode is invalid")
        if str(self.audit_evidence_level) not in ALLOWED_AUDIT_LEVELS:
            raise ValueError("audit_evidence_level is invalid")
        if self.measurement_complete is None:
            raise ValueError("measurement_complete must be explicit")
        required_summary = {
            "all_work_completed",
            "fallback_count",
            "timeout_count",
            "check_failure_count",
            "execution_outcome_count",
        }
        missing_summary = sorted(item for item in required_summary if item not in self.summary)
        if missing_summary:
            raise ValueError(f"summary missing required keys: {missing_summary}")
        if "status" in self.summary and str(self.summary["status"]) != str(self.status):
            raise ValueError("summary status conflicts with result bundle status")
        if "correctness_status" in self.summary and str(self.summary["correctness_status"]) != str(self.correctness_status):
            raise ValueError("summary correctness_status conflicts with result bundle correctness_status")
        if "performance_status" in self.summary and str(self.summary["performance_status"]) != str(self.performance_status):
            raise ValueError("summary performance_status conflicts with result bundle performance_status")
        if "status" in self.details and str(self.details["status"]) != str(self.status):
            raise ValueError("details status conflicts with result bundle status")
        if "correctness_status" in self.details and str(self.details["correctness_status"]) != str(self.correctness_status):
            raise ValueError("details correctness_status conflicts with result bundle correctness_status")
        if "performance_status" in self.details and str(self.details["performance_status"]) != str(self.performance_status):
            raise ValueError("details performance_status conflicts with result bundle performance_status")
        for key in self.extensions:
            if str(key) in RESERVED_RESULT_EXTENSION_KEYS:
                raise ValueError(f"extensions key conflicts with reserved field: {key}")
        if str(self.performance_status) == "eligible" and not bool(self.eligibility.performance_eligible):
            raise ValueError("performance_status conflicts with eligibility")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ResultBundle":
        run_identity = RunIdentity.from_dict(payload.get("run_identity", {}))
        eligibility_payload = payload.get("eligibility")
        bundle = cls(
            run_identity=run_identity,
            status=str(payload.get("status", "")),
            eligibility=EligibilityResult.from_dict(eligibility_payload)
            if isinstance(eligibility_payload, Mapping)
            else EligibilityResult(
                correctness_eligible=False,
                performance_eligible=False,
                prediction_evaluation_eligible=False,
                offline_replay_eligible=False,
                preparation_claim_eligible=False,
                correctness_reasons=("missing_eligibility",),
                performance_reasons=("missing_eligibility",),
                prediction_reasons=("missing_eligibility",),
                offline_replay_reasons=("missing_eligibility",),
                preparation_claim_reasons=("missing_eligibility",),
            ),
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
            "pipeline": str(self.run_identity.pipeline),
            "commit_sha": str(self.commit_sha),
            "git_clean": self.git_clean,
            "instrumentation_mode": str(self.instrumentation_mode),
            "audit_evidence_level": str(self.audit_evidence_level),
            "measurement_complete": self.measurement_complete,
            "eligibility": self.eligibility.to_dict(),
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


__all__ = [
    "ArtifactManifest",
    "EligibilityResult",
    "LEGACY_TRACE_REPLAY_PIPELINE",
    "OFFLINE_PIPELINE",
    "ONLINE_PIPELINE",
    "RESERVED_RESULT_EXTENSION_KEYS",
    "RESULT_BUNDLE_SCHEMA_VERSION",
    "ResultBundle",
    "RunIdentity",
]
