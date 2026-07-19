from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts.result import EligibilityResult, ResultBundle, RunIdentity
from rs.core.contracts.trace import ReferenceTraceBundle


@dataclass(frozen=True)
class EvidenceEligibility:
    correctness_eligible: bool
    performance_eligible: bool
    prediction_evaluation_eligible: bool
    offline_replay_eligible: bool
    reasons: tuple[str, ...]

    def to_result(self) -> EligibilityResult:
        return EligibilityResult(
            correctness_eligible=bool(self.correctness_eligible),
            performance_eligible=bool(self.performance_eligible),
            prediction_evaluation_eligible=bool(self.prediction_evaluation_eligible),
            offline_replay_eligible=bool(self.offline_replay_eligible),
            preparation_claim_eligible=False,
            correctness_reasons=tuple(str(item) for item in self.reasons),
            performance_reasons=(),
            prediction_reasons=(),
            offline_replay_reasons=(),
            preparation_claim_reasons=(),
        )


__all__ = ["EvidenceEligibility", "ReferenceTraceBundle", "ResultBundle", "RunIdentity"]
