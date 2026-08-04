"""Trace-owned immutable schema for the formal RS-SIM runtime.

No type in this file replaces shared types such as PhaseKey, TaskId, or
SimulationEvent.  The Trace Provider stops at workload truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .canonical import stable_digest
from .constants import (
    CONTROL_PLANE_ACCOUNTING,
    DATA_PLANE_ACCOUNTING,
    DESCRIPTOR_METADATA_SCHEMA_VERSION,
    DESCRIPTOR_PAYLOAD_RELATION,
    FIXTURE_SCHEMA_VERSION,
    PHASE_COMBINE,
    PHASE_DISPATCH,
    RECEIVER_MODEL,
    REQUIRED_EXCLUDED_COMPUTE_COMPONENTS,
    PAYLOAD_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    VALID_DATASET_SPLITS,
    VALID_PADDING_RULES,
    VALID_PHASE_KINDS,
)


class TraceValidationError(ValueError):
    """Raised when trace truth violates the formal Trace contract."""


def _as_int_tuple(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _as_matrix(values: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(_as_int_tuple(row) for row in values)


def _require_nonnegative(name: str, values: Iterable[int]) -> None:
    for index, value in enumerate(values):
        if int(value) < 0:
            raise TraceValidationError(f"{name}[{index}] must be nonnegative, got {value}")


@dataclass(frozen=True)
class PayloadSpec:
    phase_kind: str
    token_payload_bytes_per_row: int
    auxiliary_payload_bytes_per_row: int
    metadata_bytes_per_edge: int
    alignment_bytes: int
    padding_rule: str
    dtype: str
    metadata_accounting_plane: str = DATA_PLANE_ACCOUNTING
    includes_control_plane_descriptor_bytes: bool = False
    schema_version: str = PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.phase_kind not in VALID_PHASE_KINDS:
            raise TraceValidationError(f"unsupported phase_kind={self.phase_kind!r}")
        for name in (
            "token_payload_bytes_per_row",
            "auxiliary_payload_bytes_per_row",
            "metadata_bytes_per_edge",
        ):
            if int(getattr(self, name)) < 0:
                raise TraceValidationError(f"{name} must be nonnegative")
        if int(self.alignment_bytes) <= 0:
            raise TraceValidationError("alignment_bytes must be positive")
        if self.padding_rule not in VALID_PADDING_RULES:
            raise TraceValidationError(f"unsupported padding_rule={self.padding_rule!r}")
        if not str(self.dtype).strip():
            raise TraceValidationError("dtype must be non-empty")
        if self.metadata_accounting_plane != DATA_PLANE_ACCOUNTING:
            raise TraceValidationError("PayloadSpec metadata must be accounted only on DATA_PLANE")
        if self.includes_control_plane_descriptor_bytes:
            raise TraceValidationError("DataPlane PayloadSpec must not include ControlPlane descriptor bytes")

    @property
    def row_payload_bytes(self) -> int:
        return int(self.token_payload_bytes_per_row) + int(self.auxiliary_payload_bytes_per_row)

    def edge_payload_layout(self, row_count: int) -> dict[str, int | str]:
        rows = int(row_count)
        if rows < 0:
            raise TraceValidationError("row_count must be nonnegative")
        if rows == 0:
            return {
                "row_count": 0,
                "row_bytes": 0,
                "metadata_bytes": 0,
                "unaligned_total_bytes": 0,
                "alignment_padding_bytes": 0,
                "total_payload_bytes": 0,
                "padding_rule": self.padding_rule,
                "metadata_accounting_plane": self.metadata_accounting_plane,
                "control_plane_descriptor_bytes": 0,
            }
        row_bytes = rows * self.row_payload_bytes
        unaligned = row_bytes + int(self.metadata_bytes_per_edge)
        if self.padding_rule == "NONE":
            total = unaligned
        else:
            alignment = int(self.alignment_bytes)
            total = ((unaligned + alignment - 1) // alignment) * alignment
        return {
            "row_count": rows,
            "row_bytes": row_bytes,
            "metadata_bytes": int(self.metadata_bytes_per_edge),
            "unaligned_total_bytes": unaligned,
            "alignment_padding_bytes": total - unaligned,
            "total_payload_bytes": total,
            "padding_rule": self.padding_rule,
            "metadata_accounting_plane": self.metadata_accounting_plane,
            "control_plane_descriptor_bytes": 0,
        }

    def edge_payload_bytes(self, row_count: int) -> int:
        return int(self.edge_payload_layout(row_count)["total_payload_bytes"])

    def digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class DescriptorMetadataSpec:
    """ControlPlane descriptor byte truth, separate from DataPlane payload headers."""

    fixed_header_bytes: int
    per_destination_entry_bytes: int
    accounting_plane: str = CONTROL_PLANE_ACCOUNTING
    included_in_data_plane_payload: bool = False
    schema_version: str = DESCRIPTOR_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.fixed_header_bytes) < 0 or int(self.per_destination_entry_bytes) < 0:
            raise TraceValidationError("descriptor metadata byte costs must be nonnegative")
        if self.accounting_plane != CONTROL_PLANE_ACCOUNTING:
            raise TraceValidationError("descriptor metadata belongs only to CONTROL_PLANE")
        if self.included_in_data_plane_payload:
            raise TraceValidationError("descriptor metadata must not be double-counted in DataPlane payload")

    def descriptor_payload_bytes(self, destination_count: int) -> int:
        count = int(destination_count)
        if count <= 0:
            raise TraceValidationError("destination_count must be positive")
        return int(self.fixed_header_bytes) + count * int(self.per_destination_entry_bytes)

    def digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class RankNodeExpertMapping:
    world_size: int
    rank_to_node: tuple[int, ...]
    expert_to_rank: tuple[int, ...]
    mapping_name: str = ""

    def __post_init__(self) -> None:
        if int(self.world_size) <= 0:
            raise TraceValidationError("world_size must be positive")
        if len(self.rank_to_node) != int(self.world_size):
            raise TraceValidationError("rank_to_node length must equal world_size")
        if not self.expert_to_rank:
            raise TraceValidationError("expert_to_rank must be non-empty")
        _require_nonnegative("rank_to_node", self.rank_to_node)
        for expert_id, rank in enumerate(self.expert_to_rank):
            if int(rank) < 0 or int(rank) >= int(self.world_size):
                raise TraceValidationError(f"expert_to_rank[{expert_id}]={rank} outside world_size")

    @property
    def num_experts(self) -> int:
        return len(self.expert_to_rank)

    def digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class RealizedRouting:
    """Post-policy routing truth with explicit clipping/dropping/padding closure.

    All matrices have shape [source_rank][expert_id].
    raw_selected = kept + dropped
    realized_transfer = kept + padding
    """

    raw_selected_rows: tuple[tuple[int, ...], ...]
    kept_rows: tuple[tuple[int, ...], ...]
    dropped_rows: tuple[tuple[int, ...], ...]
    padding_rows: tuple[tuple[int, ...], ...]
    realization_origin: str = "captured_post_policy"

    def __post_init__(self) -> None:
        matrices = {
            "raw_selected_rows": self.raw_selected_rows,
            "kept_rows": self.kept_rows,
            "dropped_rows": self.dropped_rows,
            "padding_rows": self.padding_rows,
        }
        shapes = {(len(matrix), len(matrix[0]) if matrix else 0) for matrix in matrices.values()}
        if len(shapes) != 1:
            raise TraceValidationError(f"routing matrices have inconsistent shapes: {shapes}")
        rows, cols = next(iter(shapes))
        if rows <= 0 or cols <= 0:
            raise TraceValidationError("routing matrices must be non-empty")
        for name, matrix in matrices.items():
            if any(len(row) != cols for row in matrix):
                raise TraceValidationError(f"{name} is ragged")
            for row_index, row in enumerate(matrix):
                _require_nonnegative(f"{name}[{row_index}]", row)
        for src in range(rows):
            for expert in range(cols):
                raw = int(self.raw_selected_rows[src][expert])
                kept = int(self.kept_rows[src][expert])
                dropped = int(self.dropped_rows[src][expert])
                if raw != kept + dropped:
                    raise TraceValidationError(
                        f"row closure failed at src={src}, expert={expert}: raw={raw}, kept+dropped={kept + dropped}"
                    )
        if not str(self.realization_origin).strip():
            raise TraceValidationError("realization_origin must be non-empty")

    @classmethod
    def from_lists(
        cls,
        *,
        raw_selected_rows: Iterable[Iterable[int]],
        kept_rows: Iterable[Iterable[int]],
        dropped_rows: Iterable[Iterable[int]],
        padding_rows: Iterable[Iterable[int]],
        realization_origin: str = "captured_post_policy",
    ) -> "RealizedRouting":
        return cls(
            raw_selected_rows=_as_matrix(raw_selected_rows),
            kept_rows=_as_matrix(kept_rows),
            dropped_rows=_as_matrix(dropped_rows),
            padding_rows=_as_matrix(padding_rows),
            realization_origin=realization_origin,
        )

    @property
    def world_size(self) -> int:
        return len(self.raw_selected_rows)

    @property
    def num_experts(self) -> int:
        return len(self.raw_selected_rows[0])

    @property
    def realized_transfer_rows(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(self.kept_rows[src][expert]) + int(self.padding_rows[src][expert]) for expert in range(self.num_experts))
            for src in range(self.world_size)
        )

    def totals(self) -> dict[str, int]:
        def total(matrix: tuple[tuple[int, ...], ...]) -> int:
            return sum(sum(int(v) for v in row) for row in matrix)

        realized = self.realized_transfer_rows
        return {
            "raw_selected_rows": total(self.raw_selected_rows),
            "kept_rows": total(self.kept_rows),
            "dropped_rows": total(self.dropped_rows),
            "padding_rows": total(self.padding_rows),
            "realized_transfer_rows": total(realized),
        }

    def dispatch_row_matrix(self, mapping: RankNodeExpertMapping) -> tuple[tuple[int, ...], ...]:
        if mapping.world_size != self.world_size:
            raise TraceValidationError("routing world_size does not match mapping")
        if mapping.num_experts != self.num_experts:
            raise TraceValidationError("routing num_experts does not match mapping")
        matrix = [[0 for _ in range(self.world_size)] for _ in range(self.world_size)]
        realized = self.realized_transfer_rows
        for src in range(self.world_size):
            for expert, count in enumerate(realized[src]):
                dst = int(mapping.expert_to_rank[expert])
                matrix[src][dst] += int(count)
        return tuple(tuple(row) for row in matrix)

    def combine_row_matrix(self, mapping: RankNodeExpertMapping) -> tuple[tuple[int, ...], ...]:
        dispatch = self.dispatch_row_matrix(mapping)
        return tuple(tuple(int(dispatch[dst][src]) for dst in range(self.world_size)) for src in range(self.world_size))

    def digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class PureComputeProvenance:
    measurement_method: str
    source_artifact_digest: str
    included_components: tuple[str, ...]
    excluded_components: tuple[str, ...] = REQUIRED_EXCLUDED_COMPUTE_COMPONENTS
    units: str = "ns"
    absolute_hook_timestamps_replayed: bool = False

    def __post_init__(self) -> None:
        if self.units != "ns":
            raise TraceValidationError("pure compute units must be ns")
        if self.absolute_hook_timestamps_replayed:
            raise TraceValidationError("old absolute hook timestamps must not be replayed")
        included = {str(v).strip() for v in self.included_components}
        forbidden = set(REQUIRED_EXCLUDED_COMPUTE_COMPONENTS)
        overlap = sorted(included & forbidden)
        if overlap:
            raise TraceValidationError(f"pure compute includes forbidden waiting components: {overlap}")
        missing_exclusions = sorted(forbidden - {str(v).strip() for v in self.excluded_components})
        if missing_exclusions:
            raise TraceValidationError(f"pure compute provenance does not exclude: {missing_exclusions}")
        if not self.measurement_method.strip() or not self.source_artifact_digest.strip():
            raise TraceValidationError("measurement_method and source_artifact_digest are required")


@dataclass(frozen=True)
class LocalComputeProfile:
    combine_release_to_router_ready_ns: tuple[int, ...]
    router_and_pack_ns: tuple[int, ...]
    dispatch_local_postprocess_ns: tuple[int, ...]
    dispatch_release_to_combine_source_ready_ns: tuple[int, ...]
    bootstrap_router_and_pack_ns: tuple[int, ...]
    provenance: PureComputeProvenance

    def __post_init__(self) -> None:
        fields = (
            "combine_release_to_router_ready_ns",
            "router_and_pack_ns",
            "dispatch_local_postprocess_ns",
            "dispatch_release_to_combine_source_ready_ns",
            "bootstrap_router_and_pack_ns",
        )
        lengths = {len(getattr(self, name)) for name in fields}
        if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
            raise TraceValidationError("all local compute vectors must share a positive world_size")
        for name in fields:
            _require_nonnegative(name, getattr(self, name))

    @property
    def world_size(self) -> int:
        return len(self.router_and_pack_ns)

    def local_path_duration_ns(self, rank: int) -> int:
        return int(self.combine_release_to_router_ready_ns[rank]) + int(self.router_and_pack_ns[rank])

    def source_local_path_times(self, *, rank: int, p1_local_complete_ns: int) -> dict[str, int]:
        """Apply the frozen source-local formula without creating events."""
        if int(p1_local_complete_ns) < 0:
            raise TraceValidationError("p1_local_complete_ns must be nonnegative")
        post_combine = int(p1_local_complete_ns) + int(self.combine_release_to_router_ready_ns[rank])
        local_complete = post_combine + int(self.router_and_pack_ns[rank])
        return {
            "p1_local_complete_ns": int(p1_local_complete_ns),
            "post_combine_local_path_complete_ns": post_combine,
            "local_path_complete_ns": local_complete,
            "source_descriptor_ready_ns": local_complete,
            "source_payload_ready_ns": local_complete,
            "destination_dispatch_thread_ready_ns": local_complete,
        }

    def bootstrap_source_ready_times(self, *, rank: int, bootstrap_start_ns: int) -> dict[str, int]:
        """Return bootstrap P0 readiness without inventing a predecessor P1."""
        if int(bootstrap_start_ns) < 0:
            raise TraceValidationError("bootstrap_start_ns must be nonnegative")
        ready = int(bootstrap_start_ns) + int(self.bootstrap_router_and_pack_ns[rank])
        return {
            "bootstrap_start_ns": int(bootstrap_start_ns),
            "bootstrap_local_path_complete_ns": ready,
            "source_descriptor_ready_ns": ready,
            "source_payload_ready_ns": ready,
            "destination_dispatch_thread_ready_ns": ready,
            "predecessor_p1_exists": False,
        }

    def combine_source_ready_at_ns(self, *, rank: int, dispatch_release_ns: int) -> int:
        if int(dispatch_release_ns) < 0:
            raise TraceValidationError("dispatch_release_ns must be nonnegative")
        return int(dispatch_release_ns) + int(self.dispatch_release_to_combine_source_ready_ns[rank])

    def adapter_payload(self) -> dict[str, Any]:
        """Plain values only; the runtime adapter owns construction of shared objects."""
        return {
            "combine_release_to_router_ready_ns": list(self.combine_release_to_router_ready_ns),
            "router_and_pack_ns": list(self.router_and_pack_ns),
            "dispatch_local_postprocess_ns": list(self.dispatch_local_postprocess_ns),
            "dispatch_release_to_combine_source_ready_ns": list(self.dispatch_release_to_combine_source_ready_ns),
            "bootstrap_router_and_pack_ns": list(self.bootstrap_router_and_pack_ns),
        }


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    split: str
    source_digest: str
    transform_digest: str
    capture_id: str
    collector_version: str
    source_kind: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.split not in VALID_DATASET_SPLITS:
            raise TraceValidationError(f"split must be one of {sorted(VALID_DATASET_SPLITS)}")
        for name in ("dataset_id", "source_digest", "transform_digest", "capture_id", "collector_version", "source_kind"):
            if not str(getattr(self, name)).strip():
                raise TraceValidationError(f"{name} must be non-empty")

    def identity_digest(self) -> str:
        return stable_digest(asdict(self))


@dataclass(frozen=True)
class TraceWindow:
    window_id: str
    layer_id: int
    request_id: str
    decode_step: int
    is_bootstrap_p0: bool
    mapping: RankNodeExpertMapping
    routing: RealizedRouting
    local_compute: LocalComputeProfile
    dispatch_payload_spec: PayloadSpec
    combine_payload_spec: PayloadSpec
    descriptor_metadata_spec: DescriptorMetadataSpec
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.window_id or not self.request_id:
            raise TraceValidationError("window_id and request_id are required")
        if int(self.layer_id) < 0 or int(self.decode_step) < 0:
            raise TraceValidationError("layer_id and decode_step must be nonnegative")
        if self.mapping.world_size != self.routing.world_size:
            raise TraceValidationError("mapping/routing world_size mismatch")
        if self.mapping.num_experts != self.routing.num_experts:
            raise TraceValidationError("mapping/routing num_experts mismatch")
        if self.mapping.world_size != self.local_compute.world_size:
            raise TraceValidationError("mapping/local_compute world_size mismatch")
        if self.dispatch_payload_spec.phase_kind != PHASE_DISPATCH:
            raise TraceValidationError("dispatch_payload_spec must have phase_kind=DISPATCH")
        if self.combine_payload_spec.phase_kind != PHASE_COMBINE:
            raise TraceValidationError("combine_payload_spec must have phase_kind=COMBINE")
        if self.dispatch_payload_spec.includes_control_plane_descriptor_bytes or self.combine_payload_spec.includes_control_plane_descriptor_bytes:
            raise TraceValidationError("phase payload specs must exclude ControlPlane descriptor bytes")

    @property
    def dispatch_rows(self) -> tuple[tuple[int, ...], ...]:
        return self.routing.dispatch_row_matrix(self.mapping)

    @property
    def combine_rows(self) -> tuple[tuple[int, ...], ...]:
        return self.routing.combine_row_matrix(self.mapping)

    def payload_matrix(self, phase_kind: str) -> tuple[tuple[int, ...], ...]:
        if phase_kind == PHASE_DISPATCH:
            rows = self.dispatch_rows
            spec = self.dispatch_payload_spec
        elif phase_kind == PHASE_COMBINE:
            rows = self.combine_rows
            spec = self.combine_payload_spec
        else:
            raise TraceValidationError(f"unsupported phase_kind={phase_kind!r}")
        return tuple(tuple(spec.edge_payload_bytes(value) for value in row) for row in rows)

    def payload_accounting(self, phase_kind: str) -> dict[str, int]:
        matrix = self.payload_matrix(phase_kind)
        rows = self.dispatch_rows if phase_kind == PHASE_DISPATCH else self.combine_rows
        total_payload = sum(sum(int(value) for value in row) for row in matrix)
        local_payload = sum(int(matrix[rank][rank]) for rank in range(self.mapping.world_size))
        total_rows = sum(sum(int(value) for value in row) for row in rows)
        local_rows = sum(int(rows[rank][rank]) for rank in range(self.mapping.world_size))
        return {
            "total_payload_bytes": total_payload,
            "local_payload_bytes": local_payload,
            "local_workload_payload_bytes": local_payload,
            "local_data_plane_payload_bytes": 0,
            "remote_network_payload_bytes": total_payload - local_payload,
            "data_plane_payload_bytes": total_payload - local_payload,
            "total_rows": total_rows,
            "local_rows": local_rows,
            "remote_rows": total_rows - local_rows,
        }

    def descriptor_payload_bytes(self) -> int:
        """ControlPlane descriptor bytes for one source-rank row descriptor."""
        return self.descriptor_metadata_spec.descriptor_payload_bytes(self.mapping.world_size)

    def descriptor_payload_bytes_all_sources(self) -> int:
        return self.mapping.world_size * self.descriptor_payload_bytes()

    def complete_payload_accounting(self) -> dict[str, Any]:
        """Close workload and transport accounting without double counting.

        Local diagonal payload remains workload truth but is not charged to the
        network DataPlane. Descriptor bytes are charged only to ControlPlane.
        """
        dispatch = self.payload_accounting(PHASE_DISPATCH)
        combine = self.payload_accounting(PHASE_COMBINE)
        descriptor_total = self.descriptor_payload_bytes_all_sources()
        return {
            "dispatch": dispatch,
            "combine": combine,
            "descriptor": {
                "per_source_descriptor_bytes": self.descriptor_payload_bytes(),
                "source_descriptor_count": self.mapping.world_size,
                "control_plane_descriptor_bytes": descriptor_total,
                "data_plane_descriptor_bytes": 0,
                "accounting_plane": self.descriptor_metadata_spec.accounting_plane,
            },
            "workload_payload_bytes_total": dispatch["total_payload_bytes"] + combine["total_payload_bytes"],
            "workload_payload_bytes_local": dispatch["local_payload_bytes"] + combine["local_payload_bytes"],
            "network_data_plane_payload_bytes": dispatch["remote_network_payload_bytes"] + combine["remote_network_payload_bytes"],
            "control_plane_descriptor_bytes": descriptor_total,
            "transport_accounted_bytes": dispatch["remote_network_payload_bytes"] + combine["remote_network_payload_bytes"] + descriptor_total,
            "descriptor_double_counted_in_data_plane": False,
        }

    def workload_identity_digest(self) -> str:
        """Time-independent workload identity for paired experiments.

        Pure local timing and publication/arrival time are intentionally absent.
        The trace objects are immutable, so cache the digest after its first use.
        """
        cached = getattr(self, "_cached_workload_identity_digest", None)
        if cached is not None:
            return str(cached)
        body = {
            "window_id": self.window_id,
            "layer_id": self.layer_id,
            "request_id": self.request_id,
            "decode_step": self.decode_step,
            "is_bootstrap_p0": self.is_bootstrap_p0,
            "mapping": asdict(self.mapping),
            "routing": asdict(self.routing),
            "dispatch_payload_spec": asdict(self.dispatch_payload_spec),
            "combine_payload_spec": asdict(self.combine_payload_spec),
            "descriptor_metadata_spec": asdict(self.descriptor_metadata_spec),
            "schema_version": self.schema_version,
        }
        digest = stable_digest(body, prefix="workload")
        object.__setattr__(self, "_cached_workload_identity_digest", digest)
        return digest

    def timing_profile_digest(self) -> str:
        cached = getattr(self, "_cached_timing_profile_digest", None)
        if cached is not None:
            return str(cached)
        digest = stable_digest(asdict(self.local_compute), prefix="timing")
        object.__setattr__(self, "_cached_timing_profile_digest", digest)
        return digest

    def truth_digest(self) -> str:
        cached = getattr(self, "_cached_truth_digest", None)
        if cached is not None:
            return str(cached)
        digest = stable_digest(asdict(self))
        object.__setattr__(self, "_cached_truth_digest", digest)
        return digest

    def adapter_payload(self) -> dict[str, Any]:
        """Export trace facts without constructing shared-schema shared objects."""
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "layer_id": self.layer_id,
            "request_id": self.request_id,
            "decode_step": self.decode_step,
            "is_bootstrap_p0": self.is_bootstrap_p0,
            "world_size": self.mapping.world_size,
            "rank_to_node": list(self.mapping.rank_to_node),
            "expert_to_rank": list(self.mapping.expert_to_rank),
            "dispatch_realized_rows": [list(row) for row in self.dispatch_rows],
            "combine_realized_rows": [list(row) for row in self.combine_rows],
            "dispatch_payload_bytes": [list(row) for row in self.payload_matrix(PHASE_DISPATCH)],
            "combine_payload_bytes": [list(row) for row in self.payload_matrix(PHASE_COMBINE)],
            "dispatch_payload_accounting": self.payload_accounting(PHASE_DISPATCH),
            "combine_payload_accounting": self.payload_accounting(PHASE_COMBINE),
            "descriptor_metadata_spec": asdict(self.descriptor_metadata_spec),
            "descriptor_payload_bytes": self.descriptor_payload_bytes(),
            "complete_payload_accounting": self.complete_payload_accounting(),
            "workload_identity_digest": self.workload_identity_digest(),
            "timing_profile_digest": self.timing_profile_digest(),
            "local_compute": self.local_compute.adapter_payload(),
            "descriptor_payload_ready_relation": DESCRIPTOR_PAYLOAD_RELATION,
        }


@dataclass(frozen=True)
class FixtureInitialState:
    initial_time_ns: int
    bootstrap_window_id: str
    bootstrap_source_ranks: tuple[int, ...]
    predecessor_p1_exists: bool = False
    receiver_model: str = RECEIVER_MODEL

    def __post_init__(self) -> None:
        if int(self.initial_time_ns) < 0:
            raise TraceValidationError("initial_time_ns must be nonnegative")
        if not self.bootstrap_window_id:
            raise TraceValidationError("bootstrap_window_id is required")
        _require_nonnegative("bootstrap_source_ranks", self.bootstrap_source_ranks)
        if self.predecessor_p1_exists:
            raise TraceValidationError("bootstrap P0 must not fabricate a predecessor P1")
        if self.receiver_model != RECEIVER_MODEL:
            raise TraceValidationError(f"receiver_model must be {RECEIVER_MODEL}")


@dataclass(frozen=True)
class FixtureInput:
    fixture_id: str
    provenance: DatasetProvenance
    initial_state: FixtureInitialState
    windows: tuple[TraceWindow, ...]
    expected_invariants: dict[str, Any]
    schema_version: str = FIXTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise TraceValidationError("fixture_id is required")
        if not self.windows:
            raise TraceValidationError("fixture must contain at least one window")
        if self.windows[0].window_id != self.initial_state.bootstrap_window_id:
            raise TraceValidationError("initial_state bootstrap_window_id must identify first window")
        if not self.windows[0].is_bootstrap_p0:
            raise TraceValidationError("first window must be bootstrap P0")
        if any(window.is_bootstrap_p0 for window in self.windows[1:]):
            raise TraceValidationError("only first window may be bootstrap P0")
        keys = [(w.request_id, w.decode_step, w.layer_id) for w in self.windows]
        if len(keys) != len(set(keys)):
            raise TraceValidationError("duplicate request/decode_step/layer windows")
        world_sizes = {w.mapping.world_size for w in self.windows}
        if len(world_sizes) != 1:
            raise TraceValidationError("fixture windows must share world_size")

    @property
    def world_size(self) -> int:
        return self.windows[0].mapping.world_size

    def workload_identity_digest(self) -> str:
        cached = getattr(self, "_cached_workload_identity_digest", None)
        if cached is not None:
            return str(cached)
        digest = stable_digest(
            {
                "fixture_id": self.fixture_id,
                "windows": [window.workload_identity_digest() for window in self.windows],
                "schema_version": self.schema_version,
            },
            prefix="fixture-workload",
        )
        object.__setattr__(self, "_cached_workload_identity_digest", digest)
        return digest

    def truth_digest(self) -> str:
        cached = getattr(self, "_cached_truth_digest", None)
        if cached is not None:
            return str(cached)
        body = {
            "fixture_id": self.fixture_id,
            "provenance": asdict(self.provenance),
            "initial_state": asdict(self.initial_state),
            "windows": [asdict(window) for window in self.windows],
            "expected_invariants": self.expected_invariants,
            "schema_version": self.schema_version,
        }
        digest = stable_digest(body)
        object.__setattr__(self, "_cached_truth_digest", digest)
        return digest
