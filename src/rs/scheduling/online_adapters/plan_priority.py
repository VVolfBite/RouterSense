"""Priority artifact builders for formally paired Joint/Local plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class PriorityEntry:
    phase: str
    src_rank: int
    dst_rank: int
    byte_count: int
    priority_score: float
    wave_id: int
    bucket_hint: int
    release_dependency: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanPriorityArtifact:
    source_policy: str
    joint_policy: str
    local_policy: str
    selected_policy: str
    fallback_to_local: bool
    heuristic_family: str
    predictor_name: str
    p2_source: str
    granularity_mode: str = "dynamic_bucket_current"
    priority_entries: tuple[PriorityEntry, ...] = ()
    source_plan_digest: str = ""
    same_information_guard: bool = True
    heavy_solver_used_offline: bool = False
    online_consumption_mode: str = "phase_sync_ordering_hint"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["priority_entries"] = [entry.to_dict() for entry in self.priority_entries]
        payload["priority_digest"] = self.priority_digest
        return payload

    @property
    def priority_digest(self) -> str:
        return _stable_digest(
            {
                "source_policy": self.source_policy,
                "selected_policy": self.selected_policy,
                "fallback_to_local": self.fallback_to_local,
                "heuristic_family": self.heuristic_family,
                "granularity_mode": self.granularity_mode,
                "priority_entries": [entry.to_dict() for entry in self.priority_entries],
            }
        )


def build_priority_artifact_from_plan(
    *,
    problem: MultiPhaseSchedulingProblem,
    plan: LogicalSchedulePlan,
    heuristic_family: str,
    predictor_name: str,
    p2_source: str,
) -> PlanPriorityArtifact:
    entries: list[PriorityEntry] = []
    total_entries = sum(len(wave.flows) for wave in plan.waves)
    ordinal = total_entries
    for wave in plan.waves:
        for flow in wave.flows:
            entries.append(
                PriorityEntry(
                    phase=str(flow.phase),
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(flow.byte_count),
                    priority_score=float(ordinal),
                    wave_id=int(wave.wave_id),
                    bucket_hint=int(max(0, flow.byte_count)),
                    release_dependency=str(flow.dependency_metadata.get("release_dependency", "none")),
                )
            )
            ordinal -= 1
    return PlanPriorityArtifact(
        source_policy=str(plan.policy_name),
        joint_policy=str(plan.diagnostics.get("joint_policy", "")),
        local_policy=str(plan.diagnostics.get("local_policy", "")),
        selected_policy=str(plan.diagnostics.get("selected_policy", plan.policy_name)),
        fallback_to_local=bool(plan.diagnostics.get("fallback_to_local", False)),
        heuristic_family=heuristic_family,
        predictor_name=predictor_name,
        p2_source=p2_source,
        priority_entries=tuple(entries),
        source_plan_digest=_stable_digest(plan.to_dict()),
        heavy_solver_used_offline=False,
        online_consumption_mode="phase_sync_ordering_hint",
        metadata={
            "topology_gpus": int(problem.topology.num_gpus),
            "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
            "same_information_guard": bool(plan.diagnostics.get("same_information_guard", True)),
        },
    )


__all__ = ["PlanPriorityArtifact", "PriorityEntry", "build_priority_artifact_from_plan"]
