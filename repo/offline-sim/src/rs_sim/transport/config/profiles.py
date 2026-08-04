from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rs_sim.contracts.factories import (
    control_plane_profile_digest,
    hardware_profile_digest,
    make_control_plane_profile,
    make_hardware_profile,
)
from rs_sim.contracts.schema import ControlPlaneProfile, HardwareProfile, LinkClass
from rs_sim.contracts.digest import stable_digest


TRANSPORT_PROFILE_SCHEMA = "TRANSPORT_PROFILE_BUNDLE"
PROFILE_KIND_SYNTHETIC = "SYNTHETIC"
PROFILE_KIND_CALIBRATED = "CALIBRATED"

BANDWIDTH_MODE_FIXED_PER_LANE = "FIXED_PER_LANE"
BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE = (
    "VISIBLE_FABRIC_EQUAL_SHARE_WITHIN_BATCH"
)
_ALLOWED_BANDWIDTH_MODES = frozenset(
    {
        BANDWIDTH_MODE_FIXED_PER_LANE,
        BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE,
    }
)


@dataclass(frozen=True, slots=True)
class BandwidthContentionSensitivity:
    """Explicit transport bandwidth sensitivity configuration.

    The optional equal-share mode changes service time only.  It derives a
    contention group from the public ``NetworkTopology.nic_id_by_lane`` map
    and never creates a hidden reservation, wait queue, or policy-visible task.
    The default mode preserves the frozen fixed-per-lane behavior.
    """

    mode: str = BANDWIDTH_MODE_FIXED_PER_LANE
    group_source: str = "TOPOLOGY_NIC_ID_BY_LANE"
    share_policy: str = "EQUAL_SHARE_WITHIN_CONFIRMED_BATCH"

    def __post_init__(self) -> None:
        if self.mode not in _ALLOWED_BANDWIDTH_MODES:
            raise ValueError(f"unsupported bandwidth contention mode: {self.mode!r}")
        if self.group_source != "TOPOLOGY_NIC_ID_BY_LANE":
            raise ValueError("group_source must be TOPOLOGY_NIC_ID_BY_LANE")
        if self.share_policy != "EQUAL_SHARE_WITHIN_CONFIRMED_BATCH":
            raise ValueError(
                "share_policy must be EQUAL_SHARE_WITHIN_CONFIRMED_BATCH"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != BANDWIDTH_MODE_FIXED_PER_LANE

    @property
    def config_digest(self) -> str:
        return stable_digest(self, domain="TRANSPORT_BANDWIDTH_CONTENTION")

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "bandwidth_contention_mode": self.mode,
            "bandwidth_contention_enabled": self.enabled,
            "bandwidth_contention_group_source": self.group_source,
            "bandwidth_contention_share_policy": self.share_policy,
            "bandwidth_contention_config_digest": self.config_digest,
            "bandwidth_contention_adds_hidden_resources": False,
            "bandwidth_contention_changes_admission": False,
        }


def fixed_per_lane_bandwidth_sensitivity() -> BandwidthContentionSensitivity:
    return BandwidthContentionSensitivity()


def visible_fabric_equal_share_sensitivity() -> BandwidthContentionSensitivity:
    return BandwidthContentionSensitivity(
        mode=BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE
    )


def _profile_bundle_semantic(
    *,
    bundle_id: str,
    profile_kind: str,
    profile_provenance: str,
    performance_eligible: bool,
    local_assembly_latency_ns: int,
    hardware_profile: HardwareProfile,
    control_profile: ControlPlaneProfile,
    bandwidth_contention: BandwidthContentionSensitivity,
    source_digests: tuple[str, ...],
    assumptions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": TRANSPORT_PROFILE_SCHEMA,
        "bundle_id": bundle_id,
        "profile_kind": profile_kind,
        "profile_provenance": profile_provenance,
        "performance_eligible": performance_eligible,
        "local_assembly_latency_ns": local_assembly_latency_ns,
        "hardware_profile": hardware_profile,
        "control_profile": control_profile,
        "bandwidth_contention": bandwidth_contention,
        "source_digests": source_digests,
        "assumptions": assumptions,
    }


@dataclass(frozen=True, slots=True)
class TransportProfileBundle:
    """Resolved transport profile handoff for the unified runtime.

    The bundle stores already-resolved integer service parameters.  It does not
    fit measured samples and therefore cannot invent calibration.  A calibrated
    bundle must cite external source digests and may be performance eligible
    only when both component profiles are explicitly eligible.
    """

    bundle_id: str
    profile_kind: str
    profile_provenance: str
    local_assembly_latency_ns: int
    hardware_profile: HardwareProfile
    control_profile: ControlPlaneProfile
    bandwidth_contention: BandwidthContentionSensitivity
    source_digests: tuple[str, ...]
    assumptions: tuple[str, ...]
    bundle_digest: str
    performance_eligible: bool = False
    schema_version: str = TRANSPORT_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRANSPORT_PROFILE_SCHEMA:
            raise ValueError("unsupported transport profile bundle schema")
        for name in ("bundle_id", "profile_provenance", "bundle_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty str")
        if self.profile_kind not in {PROFILE_KIND_SYNTHETIC, PROFILE_KIND_CALIBRATED}:
            raise ValueError("profile_kind must be SYNTHETIC or CALIBRATED")
        if (
            not isinstance(self.local_assembly_latency_ns, int)
            or isinstance(self.local_assembly_latency_ns, bool)
            or self.local_assembly_latency_ns < 0
        ):
            raise ValueError("local_assembly_latency_ns must be a non-negative int")
        if not isinstance(self.hardware_profile, HardwareProfile):
            raise TypeError("hardware_profile must be HardwareProfile")
        if not isinstance(self.control_profile, ControlPlaneProfile):
            raise TypeError("control_profile must be ControlPlaneProfile")
        if hardware_profile_digest(self.hardware_profile) != self.hardware_profile.profile_digest:
            raise ValueError("hardware profile digest mismatch")
        if control_plane_profile_digest(self.control_profile) != self.control_profile.profile_digest:
            raise ValueError("control profile digest mismatch")
        if not isinstance(
            self.bandwidth_contention, BandwidthContentionSensitivity
        ):
            raise TypeError(
                "bandwidth_contention must be BandwidthContentionSensitivity"
            )
        if not isinstance(self.source_digests, tuple) or any(
            not isinstance(value, str) or not value for value in self.source_digests
        ):
            raise TypeError("source_digests must be a tuple of non-empty strings")
        if tuple(sorted(set(self.source_digests))) != self.source_digests:
            raise ValueError("source_digests must be sorted and unique")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(value, str) or not value for value in self.assumptions
        ):
            raise TypeError("assumptions must be a tuple of non-empty strings")
        if tuple(sorted(set(self.assumptions))) != self.assumptions:
            raise ValueError("assumptions must be sorted and unique")
        expected_eligible = bool(
            self.hardware_profile.performance_eligible
            and self.control_profile.performance_eligible
        )
        if self.performance_eligible != expected_eligible:
            raise ValueError(
                "bundle performance_eligible must equal both component profiles"
            )
        if self.profile_kind == PROFILE_KIND_SYNTHETIC:
            if self.performance_eligible:
                raise ValueError("synthetic profile bundles cannot be performance eligible")
        else:
            if not self.source_digests:
                raise ValueError("calibrated profile bundles require source_digests")
            provenances = (
                self.profile_provenance,
                self.hardware_profile.profile_provenance,
                self.control_profile.profile_provenance,
            )
            if any("SYNTHETIC" in value.upper() for value in provenances):
                raise ValueError(
                    "calibrated profile provenance and component provenance cannot be synthetic"
                )
        semantic = _profile_bundle_semantic(
            bundle_id=self.bundle_id,
            profile_kind=self.profile_kind,
            profile_provenance=self.profile_provenance,
            performance_eligible=self.performance_eligible,
            local_assembly_latency_ns=self.local_assembly_latency_ns,
            hardware_profile=self.hardware_profile,
            control_profile=self.control_profile,
            bandwidth_contention=self.bandwidth_contention,
            source_digests=self.source_digests,
            assumptions=self.assumptions,
        )
        expected_digest = stable_digest(
            semantic, domain="TRANSPORT_PROFILE_BUNDLE"
        )
        if self.bundle_digest != expected_digest:
            raise ValueError("transport profile bundle digest mismatch")

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "transport_profile_bundle_schema": self.schema_version,
            "transport_profile_bundle_id": self.bundle_id,
            "transport_profile_bundle_digest": self.bundle_digest,
            "transport_profile_kind": self.profile_kind,
            "transport_profile_provenance": self.profile_provenance,
            "transport_profile_performance_eligible": self.performance_eligible,
            "transport_profile_source_digests": self.source_digests,
            "transport_profile_assumptions": self.assumptions,
            "local_assembly_latency_ns": self.local_assembly_latency_ns,
            "local_assembly_enters_data_plane": False,
            "hardware_profile_id": self.hardware_profile.profile_id,
            "hardware_profile_digest": self.hardware_profile.profile_digest,
            "hardware_profile_provenance": self.hardware_profile.profile_provenance,
            "hardware_profile_performance_eligible": (
                self.hardware_profile.performance_eligible
            ),
            "control_profile_id": self.control_profile.profile_id,
            "control_profile_digest": self.control_profile.profile_digest,
            "control_profile_provenance": self.control_profile.profile_provenance,
            "control_profile_performance_eligible": (
                self.control_profile.performance_eligible
            ),
            **self.bandwidth_contention.manifest_fragment(),
        }

    def to_json_dict(self) -> dict[str, Any]:
        def link_rows(rows: tuple[tuple[LinkClass, int], ...]) -> list[list[Any]]:
            return [[link_class.value, value] for link_class, value in rows]

        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "profile_kind": self.profile_kind,
            "profile_provenance": self.profile_provenance,
            "performance_eligible": self.performance_eligible,
            "local_assembly_latency_ns": self.local_assembly_latency_ns,
            "source_digests": list(self.source_digests),
            "assumptions": list(self.assumptions),
            "bandwidth_contention": {
                "mode": self.bandwidth_contention.mode,
                "group_source": self.bandwidth_contention.group_source,
                "share_policy": self.bandwidth_contention.share_policy,
            },
            "hardware_profile": {
                "profile_id": self.hardware_profile.profile_id,
                "profile_digest": self.hardware_profile.profile_digest,
                "profile_provenance": self.hardware_profile.profile_provenance,
                "performance_eligible": self.hardware_profile.performance_eligible,
                "max_batch_tasks": self.hardware_profile.max_batch_tasks,
                "launch_delay_ns_by_link_class": link_rows(
                    self.hardware_profile.launch_delay_ns_by_link_class
                ),
                "fixed_latency_ns_by_link_class": link_rows(
                    self.hardware_profile.fixed_latency_ns_by_link_class
                ),
                "bandwidth_bytes_per_second_by_link_class": link_rows(
                    self.hardware_profile.bandwidth_bytes_per_second_by_link_class
                ),
            },
            "control_profile": {
                "profile_id": self.control_profile.profile_id,
                "profile_digest": self.control_profile.profile_digest,
                "profile_provenance": self.control_profile.profile_provenance,
                "performance_eligible": self.control_profile.performance_eligible,
                "fixed_latency_ns": self.control_profile.fixed_latency_ns,
                "bandwidth_bytes_per_second": (
                    self.control_profile.bandwidth_bytes_per_second
                ),
            },
            "bundle_digest": self.bundle_digest,
        }


@runtime_checkable
class TransportProfileProvider(Protocol):
    def load_transport_profile_bundle(self) -> TransportProfileBundle:
        """Return one immutable resolved profile bundle."""


@dataclass(frozen=True, slots=True)
class StaticTransportProfileProvider:
    bundle: TransportProfileBundle

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, TransportProfileBundle):
            raise TypeError("bundle must be TransportProfileBundle")

    def load_transport_profile_bundle(self) -> TransportProfileBundle:
        return self.bundle


def make_transport_profile_bundle(
    *,
    bundle_id: str,
    profile_kind: str,
    profile_provenance: str,
    local_assembly_latency_ns: int,
    hardware_profile: HardwareProfile,
    control_profile: ControlPlaneProfile,
    bandwidth_contention: BandwidthContentionSensitivity | None = None,
    source_digests: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
) -> TransportProfileBundle:
    source_digests = tuple(sorted(set(source_digests)))
    assumptions = tuple(sorted(set(assumptions)))
    contention = bandwidth_contention or fixed_per_lane_bandwidth_sensitivity()
    performance_eligible = bool(
        hardware_profile.performance_eligible and control_profile.performance_eligible
    )
    semantic = _profile_bundle_semantic(
        bundle_id=str(bundle_id),
        profile_kind=str(profile_kind),
        profile_provenance=str(profile_provenance),
        performance_eligible=performance_eligible,
        local_assembly_latency_ns=int(local_assembly_latency_ns),
        hardware_profile=hardware_profile,
        control_profile=control_profile,
        bandwidth_contention=contention,
        source_digests=source_digests,
        assumptions=assumptions,
    )
    return TransportProfileBundle(
        bundle_id=str(bundle_id),
        profile_kind=str(profile_kind),
        profile_provenance=str(profile_provenance),
        local_assembly_latency_ns=int(local_assembly_latency_ns),
        hardware_profile=hardware_profile,
        control_profile=control_profile,
        bandwidth_contention=contention,
        source_digests=source_digests,
        assumptions=assumptions,
        bundle_digest=stable_digest(
            semantic, domain="TRANSPORT_PROFILE_BUNDLE"
        ),
        performance_eligible=performance_eligible,
    )


def make_calibrated_transport_profile_bundle(
    *,
    bundle_id: str,
    profile_provenance: str,
    source_digests: tuple[str, ...],
    hardware_profile: HardwareProfile,
    control_profile: ControlPlaneProfile,
    local_assembly_latency_ns: int = 0,
    bandwidth_contention: BandwidthContentionSensitivity | None = None,
    assumptions: tuple[str, ...] = (),
) -> TransportProfileBundle:
    """Wrap externally fitted measured profiles without inventing calibration."""

    return make_transport_profile_bundle(
        bundle_id=bundle_id,
        profile_kind=PROFILE_KIND_CALIBRATED,
        profile_provenance=profile_provenance,
        local_assembly_latency_ns=local_assembly_latency_ns,
        hardware_profile=hardware_profile,
        control_profile=control_profile,
        bandwidth_contention=bandwidth_contention,
        source_digests=source_digests,
        assumptions=assumptions,
    )



def _require_json_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _parse_link_rows(value: Any, *, name: str) -> tuple[tuple[LinkClass, int], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    rows: list[tuple[LinkClass, int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{name} rows must be [link_class, integer]")
        rows.append((LinkClass(str(row[0])), int(row[1])))
    return tuple(rows)


def transport_profile_bundle_from_json_dict(payload: Any) -> TransportProfileBundle:
    """Parse one strict resolved transport profile object."""

    if not isinstance(payload, dict):
        raise ValueError("transport profile bundle must be a JSON object")
    allowed = {
        "schema_version", "bundle_id", "profile_kind", "profile_provenance",
        "performance_eligible", "local_assembly_latency_ns", "source_digests",
        "assumptions", "bandwidth_contention", "hardware_profile",
        "control_profile", "bundle_digest",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown transport profile fields: " + ", ".join(unknown))
    if set(payload) != allowed:
        missing = sorted(allowed - set(payload))
        raise ValueError("missing transport profile fields: " + ", ".join(missing))
    if payload.get("schema_version") != TRANSPORT_PROFILE_SCHEMA:
        raise ValueError("unsupported transport profile bundle JSON schema")
    hardware_data = payload["hardware_profile"]
    control_data = payload["control_profile"]
    if not isinstance(hardware_data, dict) or not isinstance(control_data, dict):
        raise ValueError("hardware_profile and control_profile must be objects")
    hardware_allowed = {
        "profile_id", "profile_digest", "profile_provenance",
        "performance_eligible", "max_batch_tasks",
        "launch_delay_ns_by_link_class", "fixed_latency_ns_by_link_class",
        "bandwidth_bytes_per_second_by_link_class",
    }
    control_allowed = {
        "profile_id", "profile_digest", "profile_provenance",
        "performance_eligible", "fixed_latency_ns",
        "bandwidth_bytes_per_second",
    }
    if set(hardware_data) != hardware_allowed:
        raise ValueError("hardware_profile fields do not match the strict schema")
    if set(control_data) != control_allowed:
        raise ValueError("control_profile fields do not match the strict schema")
    hardware = make_hardware_profile(
        profile_id=str(hardware_data["profile_id"]),
        profile_provenance=str(hardware_data["profile_provenance"]),
        performance_eligible=_require_json_bool(hardware_data["performance_eligible"], name="hardware_profile.performance_eligible"),
        max_batch_tasks=int(hardware_data["max_batch_tasks"]),
        launch_delay_ns_by_link_class=_parse_link_rows(
            hardware_data["launch_delay_ns_by_link_class"],
            name="launch_delay_ns_by_link_class",
        ),
        fixed_latency_ns_by_link_class=_parse_link_rows(
            hardware_data["fixed_latency_ns_by_link_class"],
            name="fixed_latency_ns_by_link_class",
        ),
        bandwidth_bytes_per_second_by_link_class=_parse_link_rows(
            hardware_data["bandwidth_bytes_per_second_by_link_class"],
            name="bandwidth_bytes_per_second_by_link_class",
        ),
    )
    if str(hardware_data["profile_digest"]) != hardware.profile_digest:
        raise ValueError("hardware profile digest mismatch in transport profile bundle")
    control = make_control_plane_profile(
        profile_id=str(control_data["profile_id"]),
        profile_provenance=str(control_data["profile_provenance"]),
        performance_eligible=_require_json_bool(control_data["performance_eligible"], name="control_profile.performance_eligible"),
        fixed_latency_ns=int(control_data["fixed_latency_ns"]),
        bandwidth_bytes_per_second=int(control_data["bandwidth_bytes_per_second"]),
    )
    if str(control_data["profile_digest"]) != control.profile_digest:
        raise ValueError("control profile digest mismatch in transport profile bundle")
    contention_data = payload["bandwidth_contention"]
    if not isinstance(contention_data, dict) or set(contention_data) != {"mode", "group_source", "share_policy"}:
        raise ValueError("bandwidth_contention fields do not match the strict schema")
    contention = BandwidthContentionSensitivity(
        mode=str(contention_data["mode"]),
        group_source=str(contention_data["group_source"]),
        share_policy=str(contention_data["share_policy"]),
    )
    bundle = make_transport_profile_bundle(
        bundle_id=str(payload["bundle_id"]),
        profile_kind=str(payload["profile_kind"]),
        profile_provenance=str(payload["profile_provenance"]),
        local_assembly_latency_ns=int(payload["local_assembly_latency_ns"]),
        hardware_profile=hardware,
        control_profile=control,
        bandwidth_contention=contention,
        source_digests=tuple(str(value) for value in payload["source_digests"]),
        assumptions=tuple(str(value) for value in payload["assumptions"]),
    )
    if str(payload["bundle_digest"]) != bundle.bundle_digest:
        raise ValueError("transport profile bundle JSON digest mismatch")
    if _require_json_bool(
        payload["performance_eligible"], name="performance_eligible"
    ) != bundle.performance_eligible:
        raise ValueError("transport profile bundle JSON eligibility mismatch")
    return bundle


def load_transport_profile_bundle_json(path: str | Path) -> TransportProfileBundle:
    """Load a resolved synthetic or calibrated profile bundle from JSON."""

    source_path = Path(path)
    return transport_profile_bundle_from_json_dict(
        json.loads(source_path.read_text(encoding="utf-8"))
    )


def write_transport_profile_bundle_json(
    path: str | Path, bundle: TransportProfileBundle
) -> None:
    if not isinstance(bundle, TransportProfileBundle):
        raise TypeError("bundle must be TransportProfileBundle")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            bundle.to_json_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class SyntheticTransportProfileSet:
    """Synthetic sensitivity inputs for local, network and descriptor paths.

    ``local_assembly_latency_ns`` is an integration-facing reference only. Local
    diagonal payload remains Backend-owned and never enters the transport DataPlane.
    """

    profile_set_id: str
    profile_provenance: str
    local_assembly_latency_ns: int
    hardware_profile: HardwareProfile
    control_profile: ControlPlaneProfile
    profile_set_digest: str
    performance_eligible: bool = False
    bandwidth_contention: BandwidthContentionSensitivity = field(
        default_factory=fixed_per_lane_bandwidth_sensitivity
    )

    def __post_init__(self) -> None:
        if not self.profile_set_id:
            raise ValueError("profile_set_id must be non-empty")
        if not self.profile_provenance:
            raise ValueError("profile_provenance must be non-empty")
        if (
            not isinstance(self.local_assembly_latency_ns, int)
            or isinstance(self.local_assembly_latency_ns, bool)
            or self.local_assembly_latency_ns < 0
        ):
            raise ValueError("local_assembly_latency_ns must be a non-negative int")
        if self.performance_eligible:
            raise ValueError("synthetic profile sets cannot be performance eligible")
        if self.hardware_profile.performance_eligible or self.control_profile.performance_eligible:
            raise ValueError("synthetic member profiles cannot be performance eligible")
        if self.hardware_profile.profile_provenance != self.profile_provenance:
            raise ValueError("hardware profile provenance mismatch")
        if self.control_profile.profile_provenance != self.profile_provenance:
            raise ValueError("control profile provenance mismatch")
        if not isinstance(
            self.bandwidth_contention, BandwidthContentionSensitivity
        ):
            raise TypeError(
                "bandwidth_contention must be BandwidthContentionSensitivity"
            )

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "synthetic_profile_set_id": self.profile_set_id,
            "synthetic_profile_set_digest": self.profile_set_digest,
            "synthetic_profile_provenance": self.profile_provenance,
            "synthetic_profile_performance_eligible": False,
            "local_assembly_latency_ns": self.local_assembly_latency_ns,
            "local_assembly_enters_data_plane": False,
            "hardware_profile_id": self.hardware_profile.profile_id,
            "hardware_profile_digest": self.hardware_profile.profile_digest,
            "control_profile_id": self.control_profile.profile_id,
            "control_profile_digest": self.control_profile.profile_digest,
            **self.bandwidth_contention.manifest_fragment(),
        }

    def as_profile_bundle(self) -> TransportProfileBundle:
        return make_transport_profile_bundle(
            bundle_id=self.profile_set_id,
            profile_kind=PROFILE_KIND_SYNTHETIC,
            profile_provenance=self.profile_provenance,
            local_assembly_latency_ns=self.local_assembly_latency_ns,
            hardware_profile=self.hardware_profile,
            control_profile=self.control_profile,
            bandwidth_contention=self.bandwidth_contention,
            assumptions=tuple(sorted({
                "CONTROL_PLANE_GLOBAL_FIFO_SERIALIZATION_SERVER",
                "CONTROL_PLANE_PROPAGATION_PIPELINED_AFTER_SERIALIZATION",
                "DATA_PLANE_RESOURCE_HELD_COMMIT_THROUGH_COMPLETION",
                "DATA_PLANE_TASKS_NON_PIPELINED_PER_ENDPOINT_NIC_LANE",
                "INTRA_INTER_SHARE_RANK_ENDPOINT_RESOURCES",
                "LOCAL_ASSEMBLY_FIXED_LATENCY_PER_EDGE",
                "LOCAL_ASSEMBLY_NO_SHARED_RESOURCE_CONTENTION",
                "SYNTHETIC_CORRECTNESS_AND_SENSITIVITY_ONLY",
            })),
        )


def make_synthetic_profile_sensitivity_set(
    *,
    profile_set_id: str = "rs-sim-synthetic-sensitivity",
    profile_provenance: str = "SYNTHETIC_TEST_ONLY",
    max_batch_tasks: int,
    local_assembly_latency_ns: int = 100,
    intra_launch_delay_ns: int = 100,
    intra_fixed_latency_ns: int = 1_000,
    intra_bandwidth_bytes_per_second: int = 100_000_000_000,
    inter_launch_delay_ns: int = 1_000,
    inter_fixed_latency_ns: int = 5_000,
    inter_bandwidth_bytes_per_second: int = 25_000_000_000,
    control_fixed_latency_ns: int = 1_000,
    control_bandwidth_bytes_per_second: int = 10_000_000_000,
    bandwidth_contention: BandwidthContentionSensitivity | None = None,
) -> SyntheticTransportProfileSet:
    """Build explicit synthetic local/intra/inter/ControlPlane sensitivity inputs."""

    profile_set_id = str(profile_set_id)
    provenance = str(profile_provenance)
    contention = bandwidth_contention or fixed_per_lane_bandwidth_sensitivity()
    hardware_profile = make_hardware_profile(
        profile_id=f"{profile_set_id}:data",
        profile_provenance=provenance,
        performance_eligible=False,
        max_batch_tasks=int(max_batch_tasks),
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, int(intra_launch_delay_ns)),
            (LinkClass.INTER_NODE, int(inter_launch_delay_ns)),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, int(intra_fixed_latency_ns)),
            (LinkClass.INTER_NODE, int(inter_fixed_latency_ns)),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, int(intra_bandwidth_bytes_per_second)),
            (LinkClass.INTER_NODE, int(inter_bandwidth_bytes_per_second)),
        ),
    )
    control_profile = make_control_plane_profile(
        profile_id=f"{profile_set_id}:control",
        profile_provenance=provenance,
        performance_eligible=False,
        fixed_latency_ns=int(control_fixed_latency_ns),
        bandwidth_bytes_per_second=int(control_bandwidth_bytes_per_second),
    )
    semantic = {
        "profile_set_id": profile_set_id,
        "profile_provenance": provenance,
        "performance_eligible": False,
        "local_assembly_latency_ns": int(local_assembly_latency_ns),
        "hardware_profile_digest": hardware_profile.profile_digest,
        "control_profile_digest": control_profile.profile_digest,
        "bandwidth_contention_digest": contention.config_digest,
    }
    return SyntheticTransportProfileSet(
        profile_set_id=profile_set_id,
        profile_provenance=provenance,
        local_assembly_latency_ns=int(local_assembly_latency_ns),
        hardware_profile=hardware_profile,
        control_profile=control_profile,
        profile_set_digest=stable_digest(
            semantic, domain="TRANSPORT_SYNTHETIC_PROFILE_SET"
        ),
        bandwidth_contention=contention,
    )


__all__ = [
    "BANDWIDTH_MODE_FIXED_PER_LANE",
    "BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE",
    "BandwidthContentionSensitivity",
    "TRANSPORT_PROFILE_SCHEMA",
    "TransportProfileBundle",
    "TransportProfileProvider",
    "PROFILE_KIND_CALIBRATED",
    "PROFILE_KIND_SYNTHETIC",
    "StaticTransportProfileProvider",
    "SyntheticTransportProfileSet",
    "fixed_per_lane_bandwidth_sensitivity",
    "load_transport_profile_bundle_json",
    "transport_profile_bundle_from_json_dict",
    "make_calibrated_transport_profile_bundle",
    "make_transport_profile_bundle",
    "make_synthetic_profile_sensitivity_set",
    "visible_fabric_equal_share_sensitivity",
    "write_transport_profile_bundle_json",
]
