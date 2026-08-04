"""JSON serialization for Trace Provider artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..schema.canonical import read_json, write_json
from ..schema.constants import (
    DESCRIPTOR_METADATA_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    RECEIVER_MODEL,
    TRACE_SCHEMA_VERSION,
)
from ..schema.canonical import stable_digest
from .canonicalization import canonicalize_fixture_serialization
from ..schema.model import (
    DatasetProvenance,
    DescriptorMetadataSpec,
    FixtureInitialState,
    FixtureInput,
    LocalComputeProfile,
    PayloadSpec,
    PureComputeProvenance,
    RankNodeExpertMapping,
    RealizedRouting,
    TraceWindow,
)


def _require_canonical_serialization(data: dict[str, Any]) -> None:
    """Reject trace files that are not already in the canonical schema."""

    fixture_schema = str(data.get("schema_version", FIXTURE_SCHEMA_VERSION))
    receiver_model = str(data.get("initial_state", {}).get("receiver_model", RECEIVER_MODEL))
    windows = data.get("windows", ())
    labels = {
        "fixture": fixture_schema,
        "receiver": receiver_model,
        "trace": {str(row.get("schema_version", TRACE_SCHEMA_VERSION)) for row in windows},
        "payload": {
            str(row[key].get("schema_version", PAYLOAD_SCHEMA_VERSION))
            for row in windows
            for key in ("dispatch_payload_spec", "combine_payload_spec")
        },
        "descriptor": {
            str(row["descriptor_metadata_spec"].get("schema_version", DESCRIPTOR_METADATA_SCHEMA_VERSION))
            for row in windows
        },
        "padding": {
            str(row[key].get("padding_rule", ""))
            for row in windows
            for key in ("dispatch_payload_spec", "combine_payload_spec")
        },
    }
    if not (
        labels["fixture"] == FIXTURE_SCHEMA_VERSION
        and labels["receiver"] == RECEIVER_MODEL
        and labels["trace"] <= {TRACE_SCHEMA_VERSION}
        and labels["payload"] <= {PAYLOAD_SCHEMA_VERSION}
        and labels["descriptor"] <= {DESCRIPTOR_METADATA_SCHEMA_VERSION}
        and labels["padding"] <= {"NONE", "EDGE_TOTAL_ALIGN_UP"}
    ):
        raise ValueError(f"unsupported or mixed trace serialization labels: {labels}")


def _tuple_int(values: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _matrix_int(values: list[list[Any]] | tuple[tuple[Any, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(_tuple_int(row) for row in values)


def payload_spec_from_dict(data: dict[str, Any]) -> PayloadSpec:
    normalized = dict(data)
    normalized["schema_version"] = PAYLOAD_SCHEMA_VERSION
    return PayloadSpec(**normalized)


def descriptor_metadata_spec_from_dict(data: dict[str, Any]) -> DescriptorMetadataSpec:
    normalized = dict(data)
    normalized["schema_version"] = DESCRIPTOR_METADATA_SCHEMA_VERSION
    return DescriptorMetadataSpec(**normalized)


def mapping_from_dict(data: dict[str, Any]) -> RankNodeExpertMapping:
    return RankNodeExpertMapping(
        world_size=int(data["world_size"]),
        rank_to_node=_tuple_int(data["rank_to_node"]),
        expert_to_rank=_tuple_int(data["expert_to_rank"]),
        mapping_name=str(data.get("mapping_name", "")),
    )


def routing_from_dict(data: dict[str, Any]) -> RealizedRouting:
    return RealizedRouting(
        raw_selected_rows=_matrix_int(data["raw_selected_rows"]),
        kept_rows=_matrix_int(data["kept_rows"]),
        dropped_rows=_matrix_int(data["dropped_rows"]),
        padding_rows=_matrix_int(data["padding_rows"]),
        realization_origin=str(data.get("realization_origin", "captured_post_policy")),
    )


def compute_provenance_from_dict(data: dict[str, Any]) -> PureComputeProvenance:
    return PureComputeProvenance(
        measurement_method=str(data["measurement_method"]),
        source_artifact_digest=str(data["source_artifact_digest"]),
        included_components=tuple(str(v) for v in data["included_components"]),
        excluded_components=tuple(str(v) for v in data["excluded_components"]),
        units=str(data.get("units", "ns")),
        absolute_hook_timestamps_replayed=bool(data.get("absolute_hook_timestamps_replayed", False)),
    )


def local_compute_from_dict(data: dict[str, Any]) -> LocalComputeProfile:
    return LocalComputeProfile(
        combine_release_to_router_ready_ns=_tuple_int(data["combine_release_to_router_ready_ns"]),
        router_and_pack_ns=_tuple_int(data["router_and_pack_ns"]),
        dispatch_local_postprocess_ns=_tuple_int(data["dispatch_local_postprocess_ns"]),
        dispatch_release_to_combine_source_ready_ns=_tuple_int(data["dispatch_release_to_combine_source_ready_ns"]),
        bootstrap_router_and_pack_ns=_tuple_int(data["bootstrap_router_and_pack_ns"]),
        provenance=compute_provenance_from_dict(data["provenance"]),
    )


def trace_window_from_dict(data: dict[str, Any]) -> TraceWindow:
    return TraceWindow(
        window_id=str(data["window_id"]),
        layer_id=int(data["layer_id"]),
        request_id=str(data["request_id"]),
        decode_step=int(data["decode_step"]),
        is_bootstrap_p0=bool(data["is_bootstrap_p0"]),
        mapping=mapping_from_dict(data["mapping"]),
        routing=routing_from_dict(data["routing"]),
        local_compute=local_compute_from_dict(data["local_compute"]),
        dispatch_payload_spec=payload_spec_from_dict(data["dispatch_payload_spec"]),
        combine_payload_spec=payload_spec_from_dict(data["combine_payload_spec"]),
        descriptor_metadata_spec=descriptor_metadata_spec_from_dict(data["descriptor_metadata_spec"]),
        metadata=dict(data.get("metadata", {})),
        schema_version=TRACE_SCHEMA_VERSION,
    )


def fixture_from_dict(
    data: dict[str, Any], *, regenerate_expected_invariants: bool = False
) -> FixtureInput:
    provenance = DatasetProvenance(**data["provenance"])
    initial_data = data["initial_state"]
    initial = FixtureInitialState(
        initial_time_ns=int(initial_data["initial_time_ns"]),
        bootstrap_window_id=str(initial_data["bootstrap_window_id"]),
        bootstrap_source_ranks=_tuple_int(initial_data["bootstrap_source_ranks"]),
        predecessor_p1_exists=bool(initial_data.get("predecessor_p1_exists", False)),
        receiver_model=RECEIVER_MODEL,
    )
    windows = tuple(trace_window_from_dict(row) for row in data["windows"])
    expected_invariants = dict(data["expected_invariants"])
    if regenerate_expected_invariants:
        from ..schema.invariants import fixture_invariants

        expected_invariants = fixture_invariants(windows)
    return FixtureInput(
        fixture_id=str(data["fixture_id"]),
        provenance=provenance,
        initial_state=initial,
        windows=windows,
        expected_invariants=expected_invariants,
        schema_version=FIXTURE_SCHEMA_VERSION,
    )


def write_fixture(path: Path, fixture: FixtureInput) -> None:
    payload = asdict(fixture)
    payload["fixture_truth_digest"] = fixture.truth_digest()
    write_json(path, payload)


def load_fixture(path: Path, *, verify_declared_digest: bool = True) -> FixtureInput:
    """Load one canonical fixture.

    ``verify_declared_digest=False`` is reserved for an isolated worker whose
    parent process already verified the immutable source artifact.  The worker
    must still compare the constructed fixture truth digest with the trusted
    digest supplied by that parent.
    """
    data = read_json(path)
    declared_digest = data.pop("fixture_truth_digest", None)
    # Verify the immutable source artifact before constructing runtime objects.
    if declared_digest is not None and verify_declared_digest:
        source_digest = stable_digest(data)
        if str(declared_digest) != source_digest:
            raise ValueError(
                f"fixture_truth_digest mismatch in {path}: "
                f"declared={declared_digest}, source={source_digest}"
            )
    data, labels_changed = canonicalize_fixture_serialization(data)
    _require_canonical_serialization(data)
    fixture = fixture_from_dict(data, regenerate_expected_invariants=labels_changed)
    if declared_digest is not None and verify_declared_digest and not labels_changed:
        # The verified serialized digest is exactly the immutable fixture truth
        # digest for canonical files.  Prime the object cache so downstream
        # runtime components do not repeatedly hash the same large fixture.
        object.__setattr__(fixture, "_cached_truth_digest", str(declared_digest))
    return fixture
