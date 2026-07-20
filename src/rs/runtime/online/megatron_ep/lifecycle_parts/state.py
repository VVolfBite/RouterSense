"""State and evidence records used by the Megatron EP lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ReleaseStateLedger:
    run_id: str
    forward_generation: int
    microbatch_id: str
    completed_payload_roles_by_phase: dict[tuple[str, str, int], set[str]] = field(default_factory=dict)
    satisfied_release_ids: set[str] = field(default_factory=set)

    def reset(self, *, run_id: str, forward_generation: int, microbatch_id: str) -> None:
        self.run_id = str(run_id)
        self.forward_generation = int(forward_generation)
        self.microbatch_id = str(microbatch_id)
        self.completed_payload_roles_by_phase.clear()
        self.satisfied_release_ids.clear()

    def record_payload_completion(
        self,
        *,
        layer_id: str,
        phase: str,
        local_group_rank: int,
        payload_role: str,
        required_payload_roles: tuple[str, ...],
    ) -> tuple[str, ...]:
        key = (str(layer_id), str(phase), int(local_group_rank))
        completed = self.completed_payload_roles_by_phase.setdefault(key, set())
        completed.add(str(payload_role))
        required = {str(item) for item in required_payload_roles}
        if not required.issubset(completed):
            return ()
        if str(phase) == "P0":
            release_id = f"release:{str(layer_id)}:p0_inbound_complete:{int(local_group_rank)}"
        elif str(phase) == "P1":
            release_id = f"release:{str(layer_id)}:p1_inbound_complete:{int(local_group_rank)}"
        else:
            return ()
        if release_id in self.satisfied_release_ids:
            return ()
        self.satisfied_release_ids.add(release_id)
        return (release_id,)


@dataclass
class RuntimeEvidenceCounters:
    fallback_count: int = 0
    preparation_miss_count: int = 0
    provisional_execution_count: int = 0
    timeout_count: int = 0
    check_failure_count: int = 0
    materialization_failure_count: int = 0
    execution_failure_count: int = 0
    cleanup_failure_count: int = 0


@dataclass
class ExpectedEvidence:
    claim_scope: str = "formal"
    selected_layers: set[str] = field(default_factory=set)
    expected_phase_payload_roles: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    expected_execution_count: int = 0
    measurement_required: bool = True
    performance_claim_requested: bool = False
    prediction_claim_requested: bool = False
    offline_claim_requested: bool = False

    def reset(
        self,
        *,
        claim_scope: str,
        selected_layers: set[str],
        measurement_required: bool,
        performance_claim_requested: bool,
        prediction_claim_requested: bool,
        offline_claim_requested: bool,
    ) -> None:
        self.claim_scope = str(claim_scope)
        self.selected_layers = {str(item) for item in selected_layers}
        self.expected_phase_payload_roles.clear()
        self.expected_execution_count = 0
        self.measurement_required = bool(measurement_required)
        self.performance_claim_requested = bool(performance_claim_requested)
        self.prediction_claim_requested = bool(prediction_claim_requested)
        self.offline_claim_requested = bool(offline_claim_requested)

    def expect_phase_payload_roles(
        self,
        *,
        layer_id: str,
        phase: str,
        payload_roles: tuple[str, ...],
    ) -> None:
        if not payload_roles:
            return
        key = (str(layer_id), str(phase))
        expected = self.expected_phase_payload_roles.setdefault(key, set())
        for payload_role in payload_roles:
            if str(payload_role) not in expected:
                expected.add(str(payload_role))
                self.expected_execution_count += 1


@dataclass(frozen=True)
class RuntimePredictionCompatResult:
    predictor_id: str
    matrix: tuple[tuple[int, ...], ...]
    matrix_digest: str
    confidence: float
    predictor_version: str = "v1"
    evaluation_eligible: bool = True
    is_oracle: bool = False
    valid: bool = True
    error: str = ""
    fallback: bool = False

    @property
    def predictor_name(self) -> str:
        return self.predictor_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "matrix_digest": str(self.matrix_digest),
            "predictor_name": str(self.predictor_name),
            "predictor_id": str(self.predictor_id),
            "predictor_version": str(self.predictor_version),
            "confidence": float(self.confidence),
            "evaluation_eligible": bool(self.evaluation_eligible),
            "is_oracle": bool(self.is_oracle),
            "valid": bool(self.valid),
            "error": str(self.error),
            "fallback": bool(self.fallback),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimePredictionCompatResult":
        return cls(
            predictor_id=str(payload.get("predictor_id", payload.get("predictor_name", ""))),
            matrix=tuple(tuple(int(value) for value in row) for row in payload.get("matrix", [])),
            matrix_digest=str(payload.get("matrix_digest", "")),
            predictor_version=str(payload.get("predictor_version", "v1")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            evaluation_eligible=bool(payload.get("evaluation_eligible", True)),
            is_oracle=bool(payload.get("is_oracle", False)),
            valid=bool(payload.get("valid", False)),
            error=str(payload.get("error", "")),
            fallback=bool(payload.get("fallback", False)),
        )

__all__ = [
    "ReleaseStateLedger",
    "RuntimeEvidenceCounters",
    "ExpectedEvidence",
    "RuntimePredictionCompatResult",
]
