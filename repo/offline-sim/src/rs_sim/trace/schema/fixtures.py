"""Deterministic EP4 contract fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .canonical import stable_digest
from ..build.collector import TraceCollector
from .constants import PHASE_COMBINE, PHASE_DISPATCH, PADDING_EDGE_TOTAL_ALIGN_UP
from .model import (
    DatasetProvenance,
    DescriptorMetadataSpec,
    LocalComputeProfile,
    PayloadSpec,
    PureComputeProvenance,
    RankNodeExpertMapping,
    RealizedRouting,
)
from ..io.serialization import write_fixture


def _mapping() -> RankNodeExpertMapping:
    return RankNodeExpertMapping(
        world_size=4,
        rank_to_node=(0, 0, 1, 1),
        expert_to_rank=(0, 0, 1, 1, 2, 2, 3, 3),
        mapping_name="ep4_2nodes_8experts",
    )


def _payloads() -> tuple[PayloadSpec, PayloadSpec]:
    dispatch = PayloadSpec(
        phase_kind=PHASE_DISPATCH,
        token_payload_bytes_per_row=4096,
        auxiliary_payload_bytes_per_row=16,
        metadata_bytes_per_edge=64,
        alignment_bytes=256,
        padding_rule=PADDING_EDGE_TOTAL_ALIGN_UP,
        dtype="bf16+route_meta_v1",
    )
    combine = PayloadSpec(
        phase_kind=PHASE_COMBINE,
        token_payload_bytes_per_row=4096,
        auxiliary_payload_bytes_per_row=8,
        metadata_bytes_per_edge=48,
        alignment_bytes=128,
        padding_rule=PADDING_EDGE_TOTAL_ALIGN_UP,
        dtype="bf16+combine_meta_v1",
    )
    return dispatch, combine


def _descriptor_metadata() -> DescriptorMetadataSpec:
    return DescriptorMetadataSpec(fixed_header_bytes=16, per_destination_entry_bytes=8)


def _compute(tag: str, offset: int = 0) -> LocalComputeProfile:
    provenance = PureComputeProvenance(
        measurement_method="synthetic_contract_fixture_segment_sum_v1",
        source_artifact_digest=stable_digest({"fixture_compute_tag": tag}),
        included_components=(
            "combine_postprocess",
            "residual_and_norm",
            "local_attention_path",
            "router",
            "pack",
            "dispatch_postprocess",
            "expert_compute",
        ),
    )
    return LocalComputeProfile(
        combine_release_to_router_ready_ns=(1100 + offset, 1300 + offset, 1200 + offset, 1400 + offset),
        router_and_pack_ns=(700 + offset, 800 + offset, 750 + offset, 850 + offset),
        dispatch_local_postprocess_ns=(300 + offset, 350 + offset, 325 + offset, 375 + offset),
        dispatch_release_to_combine_source_ready_ns=(2100 + offset, 2200 + offset, 2300 + offset, 2400 + offset),
        bootstrap_router_and_pack_ns=(900 + offset, 1000 + offset, 950 + offset, 1050 + offset),
        provenance=provenance,
    )


def _provenance(split: str, fixture_id: str) -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="rs_sim_contract_ep4",
        split=split,
        source_digest=stable_digest({"split": split, "fixture": fixture_id, "source": "synthetic_contract_fixture"}),
        transform_digest=stable_digest({"adapter": "builtin_fixture_v1", "split": split, "fixture": fixture_id}),
        capture_id=f"capture-{fixture_id}",
        collector_version="rs-sim-trace-provider",
        source_kind="synthetic_contract_fixture",
        notes="Contract-only fixture; not a performance result or real model trace.",
    )


def _routing(raw: list[list[int]], kept: list[list[int]] | None = None, padding: list[list[int]] | None = None, origin: str = "synthetic") -> RealizedRouting:
    kept = raw if kept is None else kept
    padding = [[0 for _ in row] for row in raw] if padding is None else padding
    dropped = [[int(raw[r][e]) - int(kept[r][e]) for e in range(len(raw[r]))] for r in range(len(raw))]
    return RealizedRouting.from_lists(
        raw_selected_rows=raw,
        kept_rows=kept,
        dropped_rows=dropped,
        padding_rows=padding,
        realization_origin=origin,
    )


def build_builtin_fixtures() -> tuple:
    mapping = _mapping()
    dispatch_spec, combine_spec = _payloads()
    descriptor_metadata_spec = _descriptor_metadata()

    definitions = [
        (
            "ep4_train_balanced",
            "train",
            [
                _routing(
                    [[4, 4, 2, 2, 1, 1, 1, 1], [1, 1, 4, 4, 2, 2, 1, 1], [1, 1, 1, 1, 4, 4, 2, 2], [2, 2, 1, 1, 1, 1, 4, 4]],
                    origin="synthetic_balanced_no_drop_no_padding",
                ),
                _routing(
                    [[3, 3, 3, 3, 1, 1, 1, 1], [1, 1, 3, 3, 3, 3, 1, 1], [1, 1, 1, 1, 3, 3, 3, 3], [3, 3, 1, 1, 1, 1, 3, 3]],
                    origin="synthetic_rotated_no_drop_no_padding",
                ),
            ],
        ),
        (
            "ep4_validation_skew_clipped",
            "validation",
            [
                _routing(
                    raw=[[12, 8, 2, 1, 1, 1, 1, 0], [10, 6, 2, 2, 1, 1, 0, 0], [8, 7, 3, 2, 2, 1, 1, 0], [9, 5, 2, 2, 1, 1, 1, 1]],
                    kept=[[8, 6, 2, 1, 1, 1, 1, 0], [7, 5, 2, 2, 1, 1, 0, 0], [6, 5, 3, 2, 2, 1, 1, 0], [6, 4, 2, 2, 1, 1, 1, 1]],
                    origin="synthetic_explicit_capacity_clipping",
                ),
                _routing(
                    raw=[[2, 2, 8, 6, 2, 1, 1, 0], [2, 2, 10, 8, 1, 1, 0, 0], [1, 1, 9, 7, 2, 2, 1, 1], [1, 1, 8, 6, 2, 2, 1, 1]],
                    kept=[[2, 2, 6, 5, 2, 1, 1, 0], [2, 2, 7, 6, 1, 1, 0, 0], [1, 1, 6, 5, 2, 2, 1, 1], [1, 1, 6, 5, 2, 2, 1, 1]],
                    origin="synthetic_explicit_capacity_clipping",
                ),
            ],
        ),
        (
            "ep4_test_padding",
            "test",
            [
                _routing(
                    raw=[[3, 2, 1, 0, 0, 0, 1, 0], [0, 1, 3, 2, 1, 0, 0, 0], [0, 0, 1, 0, 3, 2, 0, 1], [1, 0, 0, 0, 0, 1, 3, 2]],
                    padding=[[1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]],
                    origin="synthetic_explicit_transfer_padding",
                ),
                _routing(
                    raw=[[2, 1, 0, 1, 0, 1, 0, 1], [1, 0, 2, 1, 1, 0, 1, 0], [0, 1, 1, 0, 2, 1, 0, 1], [1, 0, 0, 1, 1, 0, 2, 1]],
                    padding=[[0, 1, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]],
                    origin="synthetic_explicit_transfer_padding",
                ),
            ],
        ),
    ]

    fixtures = []
    for fixture_id, split, routings in definitions:
        collector = TraceCollector(fixture_id=fixture_id, provenance=_provenance(split, fixture_id))
        for layer_id, routing in enumerate(routings):
            collector.record_window(
                window_id=f"{fixture_id}-request0-step0-layer{layer_id}",
                layer_id=layer_id,
                request_id=f"{fixture_id}-request0",
                decode_step=0,
                is_bootstrap_p0=layer_id == 0,
                mapping=mapping,
                routing=routing,
                local_compute=_compute(f"{fixture_id}-layer{layer_id}", offset=layer_id * 100),
                dispatch_payload_spec=dispatch_spec,
                combine_payload_spec=combine_spec,
                descriptor_metadata_spec=descriptor_metadata_spec,
                metadata={
                    "fixture_role": "contract_acceptance",
                    "realized_after_clipping_dropping_padding": True,
                    "performance_claim_allowed": False,
                },
            )
        fixtures.append(collector.freeze())
    return tuple(fixtures)


def write_builtin_fixtures(output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fixture in build_builtin_fixtures():
        path = output_dir / f"{fixture.fixture_id}.json"
        write_fixture(path, fixture)
        paths.append(path)
    return tuple(paths)


def build_golden_fixture():
    """Return the deterministic two-window fixture used by focused tests."""
    return build_builtin_fixtures()[0]
