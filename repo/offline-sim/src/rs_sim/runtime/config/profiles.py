from __future__ import annotations

"""Immutable runtime performance profile handoff.

One bundle owns every deterministic service-time input used by the formal
Current-P12 runtime.  Callers may select one bundle, but cannot override
transport, receiver, local assembly, or planning costs independently.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rs_sim.backend.resources.costs import LinearReceiverCostModel
from rs_sim.contracts.digest import stable_digest
from rs_sim.scheduler.execution.lines import PlanningCostModel
from rs_sim.transport.config.profiles import (
    PROFILE_KIND_CALIBRATED,
    PROFILE_KIND_SYNTHETIC,
    TransportProfileBundle,
    transport_profile_bundle_from_json_dict,
    make_synthetic_profile_sensitivity_set,
)

RUNTIME_PROFILE_SCHEMA = "RS_SIM_RUNTIME_PROFILE_BUNDLE"
_ALLOWED_KINDS = frozenset({PROFILE_KIND_SYNTHETIC, PROFILE_KIND_CALIBRATED})


def _semantic_payload(
    *,
    profile_id: str,
    profile_kind: str,
    profile_provenance: str,
    performance_eligible: bool,
    transport_profile: TransportProfileBundle,
    receiver_cost_model: LinearReceiverCostModel,
    planning_cost_model: PlanningCostModel,
    source_digests: tuple[str, ...],
    assumptions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "profile_kind": profile_kind,
        "profile_provenance": profile_provenance,
        "performance_eligible": performance_eligible,
        "transport_profile_digest": transport_profile.bundle_digest,
        "receiver_cost_model": asdict(receiver_cost_model),
        "planning_cost_model": asdict(planning_cost_model),
        "source_digests": source_digests,
        "assumptions": assumptions,
    }


@dataclass(frozen=True, slots=True)
class RuntimeProfileBundle:
    profile_id: str
    profile_kind: str
    profile_provenance: str
    transport_profile: TransportProfileBundle
    receiver_cost_model: LinearReceiverCostModel
    planning_cost_model: PlanningCostModel
    source_digests: tuple[str, ...]
    assumptions: tuple[str, ...]
    profile_digest: str
    performance_eligible: bool = False
    schema_version: str = RUNTIME_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_PROFILE_SCHEMA:
            raise ValueError("unsupported runtime profile schema")
        for name in ("profile_id", "profile_provenance", "profile_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if self.profile_kind not in _ALLOWED_KINDS:
            raise ValueError("profile_kind must be SYNTHETIC or CALIBRATED")
        if not isinstance(self.transport_profile, TransportProfileBundle):
            raise TypeError("transport_profile must be TransportProfileBundle")
        if not isinstance(self.receiver_cost_model, LinearReceiverCostModel):
            raise TypeError("receiver_cost_model must be LinearReceiverCostModel")
        if not isinstance(self.planning_cost_model, PlanningCostModel):
            raise TypeError("planning_cost_model must be PlanningCostModel")
        if tuple(sorted(set(self.source_digests))) != self.source_digests:
            raise ValueError("source_digests must be sorted and unique")
        if tuple(sorted(set(self.assumptions))) != self.assumptions:
            raise ValueError("assumptions must be sorted and unique")
        if any(not isinstance(value, str) or not value for value in self.source_digests):
            raise TypeError("source_digests must contain non-empty strings")
        if any(not isinstance(value, str) or not value for value in self.assumptions):
            raise TypeError("assumptions must contain non-empty strings")
        if self.profile_kind == PROFILE_KIND_SYNTHETIC and self.performance_eligible:
            raise ValueError("synthetic runtime profiles cannot be performance eligible")
        if self.performance_eligible and not self.transport_profile.performance_eligible:
            raise ValueError(
                "performance-eligible runtime profile requires an eligible transport profile"
            )
        if self.profile_kind == PROFILE_KIND_CALIBRATED and not self.source_digests:
            raise ValueError("calibrated runtime profiles require source_digests")
        expected = stable_digest(
            _semantic_payload(
                profile_id=self.profile_id,
                profile_kind=self.profile_kind,
                profile_provenance=self.profile_provenance,
                performance_eligible=self.performance_eligible,
                transport_profile=self.transport_profile,
                receiver_cost_model=self.receiver_cost_model,
                planning_cost_model=self.planning_cost_model,
                source_digests=self.source_digests,
                assumptions=self.assumptions,
            ),
            domain="RS_SIM_RUNTIME_PROFILE_BUNDLE",
        )
        if self.profile_digest != expected:
            raise ValueError("runtime profile digest mismatch")

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "runtime_profile_schema": self.schema_version,
            "runtime_profile_id": self.profile_id,
            "runtime_profile_digest": self.profile_digest,
            "runtime_profile_kind": self.profile_kind,
            "runtime_profile_provenance": self.profile_provenance,
            "runtime_profile_performance_eligible": self.performance_eligible,
            "runtime_profile_source_digests": self.source_digests,
            "runtime_profile_assumptions": self.assumptions,
            "receiver_cost_model": asdict(self.receiver_cost_model),
            "receiver_cost_model_digest": stable_digest(
                self.receiver_cost_model, domain="RS_SIM_RECEIVER_COST_MODEL"
            ),
            "planning_cost_model": asdict(self.planning_cost_model),
            "planning_cost_model_digest": self.planning_cost_model.model_digest,
            **self.transport_profile.manifest_fragment(),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "profile_provenance": self.profile_provenance,
            "performance_eligible": self.performance_eligible,
            "source_digests": list(self.source_digests),
            "assumptions": list(self.assumptions),
            "transport_profile": self.transport_profile.to_json_dict(),
            "receiver_cost_model": asdict(self.receiver_cost_model),
            "planning_cost_model": asdict(self.planning_cost_model),
            "profile_digest": self.profile_digest,
        }


def make_runtime_profile_bundle(
    *,
    profile_id: str,
    profile_kind: str,
    profile_provenance: str,
    transport_profile: TransportProfileBundle,
    receiver_cost_model: LinearReceiverCostModel,
    planning_cost_model: PlanningCostModel,
    source_digests: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
    performance_eligible: bool = False,
) -> RuntimeProfileBundle:
    source_digests = tuple(sorted(set(source_digests)))
    assumptions = tuple(sorted(set(assumptions)))
    semantic = _semantic_payload(
        profile_id=str(profile_id),
        profile_kind=str(profile_kind),
        profile_provenance=str(profile_provenance),
        performance_eligible=bool(performance_eligible),
        transport_profile=transport_profile,
        receiver_cost_model=receiver_cost_model,
        planning_cost_model=planning_cost_model,
        source_digests=source_digests,
        assumptions=assumptions,
    )
    return RuntimeProfileBundle(
        profile_id=str(profile_id),
        profile_kind=str(profile_kind),
        profile_provenance=str(profile_provenance),
        transport_profile=transport_profile,
        receiver_cost_model=receiver_cost_model,
        planning_cost_model=planning_cost_model,
        source_digests=source_digests,
        assumptions=assumptions,
        profile_digest=stable_digest(
            semantic, domain="RS_SIM_RUNTIME_PROFILE_BUNDLE"
        ),
        performance_eligible=bool(performance_eligible),
    )


def make_default_synthetic_runtime_profile(
    *, max_batch_tasks: int, local_assembly_latency_ns: int = 5
) -> RuntimeProfileBundle:
    transport = make_synthetic_profile_sensitivity_set(
        profile_set_id="rs-sim-current-p12-synthetic",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        max_batch_tasks=int(max_batch_tasks),
        local_assembly_latency_ns=int(local_assembly_latency_ns),
    ).as_profile_bundle()
    return make_runtime_profile_bundle(
        profile_id="rs-sim-current-p12-synthetic",
        profile_kind=PROFILE_KIND_SYNTHETIC,
        profile_provenance="SYNTHETIC_TEST_ONLY",
        transport_profile=transport,
        receiver_cost_model=LinearReceiverCostModel(
            posting_fixed_ns=3,
            posting_bytes_per_ns=4096,
            drain_fixed_ns=2,
            drain_bytes_per_ns=4096,
        ),
        planning_cost_model=PlanningCostModel(
            prediction_base_ns=31,
            prediction_per_observation_ns=2,
            prediction_per_task_ns=1,
            control_base_ns=41,
            control_per_observation_ns=2,
            control_per_task_ns=1,
            control_per_phase_ns=3,
            binding_base_ns=13,
            binding_per_task_ns=1,
            binding_per_phase_ns=2,
            zero_cost_mode=False,
        ),
        assumptions=(
            "CORRECTNESS_AND_RELATIVE_SENSITIVITY_ONLY",
            "NOT_HARDWARE_CALIBRATED",
            "PLANNING_LINES_GLOBAL_FIFO_NONPREEMPTIVE",
            "RECEIVER_DRAIN_SINGLE_FIFO_PER_DESTINATION",
            "RECEIVER_POSTING_SINGLE_FIFO_PER_DESTINATION",
            "RECEIVER_STAGING_RESERVED_AT_POSTING_START",
        ),
        performance_eligible=False,
    )


def _reject_unknown(payload: dict[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _receiver_from_json(payload: Any) -> LinearReceiverCostModel:
    if not isinstance(payload, dict):
        raise ValueError("receiver_cost_model must be an object")
    allowed = {
        "posting_fixed_ns",
        "posting_bytes_per_ns",
        "drain_fixed_ns",
        "drain_bytes_per_ns",
    }
    _reject_unknown(payload, allowed, name="receiver_cost_model")
    if set(payload) != allowed:
        raise ValueError("receiver_cost_model must define all fields")
    return LinearReceiverCostModel(**{key: int(payload[key]) for key in allowed})


def _planning_from_json(payload: Any) -> PlanningCostModel:
    if not isinstance(payload, dict):
        raise ValueError("planning_cost_model must be an object")
    allowed = {
        "prediction_base_ns",
        "prediction_per_observation_ns",
        "prediction_per_task_ns",
        "control_base_ns",
        "control_per_observation_ns",
        "control_per_task_ns",
        "control_per_phase_ns",
        "binding_base_ns",
        "binding_per_task_ns",
        "binding_per_phase_ns",
        "zero_cost_mode",
    }
    _reject_unknown(payload, allowed, name="planning_cost_model")
    if set(payload) != allowed:
        raise ValueError("planning_cost_model must define all fields")
    values = {key: int(payload[key]) for key in allowed - {"zero_cost_mode"}}
    values["zero_cost_mode"] = _require_bool(
        payload["zero_cost_mode"], name="planning_cost_model.zero_cost_mode"
    )
    return PlanningCostModel(**values)


def load_runtime_profile_bundle_json(path: str | Path) -> RuntimeProfileBundle:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime profile must be a JSON object")
    allowed = {
        "schema_version",
        "profile_id",
        "profile_kind",
        "profile_provenance",
        "performance_eligible",
        "source_digests",
        "assumptions",
        "transport_profile",
        "receiver_cost_model",
        "planning_cost_model",
        "profile_digest",
    }
    _reject_unknown(payload, allowed, name="runtime profile")
    if set(payload) != allowed:
        raise ValueError("runtime profile must define all schema fields")
    if payload["schema_version"] != RUNTIME_PROFILE_SCHEMA:
        raise ValueError("unsupported runtime profile JSON schema")

    transport_payload = payload["transport_profile"]
    if not isinstance(transport_payload, dict):
        raise ValueError("transport_profile must be an object")
    transport = transport_profile_bundle_from_json_dict(transport_payload)

    bundle = make_runtime_profile_bundle(
        profile_id=str(payload["profile_id"]),
        profile_kind=str(payload["profile_kind"]),
        profile_provenance=str(payload["profile_provenance"]),
        transport_profile=transport,
        receiver_cost_model=_receiver_from_json(payload["receiver_cost_model"]),
        planning_cost_model=_planning_from_json(payload["planning_cost_model"]),
        source_digests=tuple(str(value) for value in payload["source_digests"]),
        assumptions=tuple(str(value) for value in payload["assumptions"]),
        performance_eligible=_require_bool(
            payload["performance_eligible"], name="performance_eligible"
        ),
    )
    if str(payload["profile_digest"]) != bundle.profile_digest:
        raise ValueError("runtime profile JSON digest mismatch")
    return bundle


def write_runtime_profile_bundle_json(
    path: str | Path, bundle: RuntimeProfileBundle
) -> None:
    if not isinstance(bundle, RuntimeProfileBundle):
        raise TypeError("bundle must be RuntimeProfileBundle")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(bundle.to_json_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RUNTIME_PROFILE_SCHEMA",
    "RuntimeProfileBundle",
    "load_runtime_profile_bundle_json",
    "make_default_synthetic_runtime_profile",
    "make_runtime_profile_bundle",
    "write_runtime_profile_bundle_json",
]
