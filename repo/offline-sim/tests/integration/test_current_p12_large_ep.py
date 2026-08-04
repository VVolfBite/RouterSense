from __future__ import annotations

import pytest

from rs_sim.runtime import build_current_p12_integration_runtime
from rs_sim.scheduler.prediction.fate_p2 import FATE_METADATA_KEY, canonical_fate_metadata
from rs_sim.trace.schema.canonical import stable_digest
from tests.support.runtime_profiles import synthetic_runtime_profile
from rs_sim.trace.build.collector import TraceCollector
from rs_sim.trace.schema.constants import PHASE_COMBINE, PHASE_DISPATCH
from rs_sim.trace.schema.model import (
    DatasetProvenance,
    DescriptorMetadataSpec,
    LocalComputeProfile,
    PayloadSpec,
    PureComputeProvenance,
    RankNodeExpertMapping,
    RealizedRouting,
)


def _sparse_fixture(world_size: int):
    mapping = RankNodeExpertMapping(
        world_size=world_size,
        rank_to_node=tuple(rank // 8 for rank in range(world_size)),
        expert_to_rank=tuple(range(world_size)),
        mapping_name=f"ep{world_size}-sparse-smoke",
    )
    dispatch = PayloadSpec(
        phase_kind=PHASE_DISPATCH,
        token_payload_bytes_per_row=128,
        auxiliary_payload_bytes_per_row=0,
        metadata_bytes_per_edge=0,
        alignment_bytes=1,
        padding_rule="NONE",
        dtype="bf16",
    )
    combine = PayloadSpec(
        phase_kind=PHASE_COMBINE,
        token_payload_bytes_per_row=128,
        auxiliary_payload_bytes_per_row=0,
        metadata_bytes_per_edge=0,
        alignment_bytes=1,
        padding_rule="NONE",
        dtype="bf16",
    )
    compute = LocalComputeProfile(
        combine_release_to_router_ready_ns=(10,) * world_size,
        router_and_pack_ns=(10,) * world_size,
        dispatch_local_postprocess_ns=(10,) * world_size,
        dispatch_release_to_combine_source_ready_ns=(10,) * world_size,
        bootstrap_router_and_pack_ns=(10,) * world_size,
        provenance=PureComputeProvenance(
            measurement_method="synthetic_large_ep_smoke",
            source_artifact_digest=stable_digest({"world_size": world_size}),
            included_components=("router", "expert_compute"),
        ),
    )
    collector = TraceCollector(
        fixture_id=f"ep{world_size}-sparse-smoke",
        provenance=DatasetProvenance(
            dataset_id=f"ep{world_size}-sparse-smoke",
            split="test",
            source_digest=stable_digest({"source": world_size}),
            transform_digest=stable_digest({"transform": world_size}),
            capture_id=f"ep{world_size}-sparse-smoke",
            collector_version="test",
            source_kind="synthetic",
        ),
    )
    for layer in range(2):
        rows = []
        for src in range(world_size):
            row = [0] * world_size
            row[(src + 1 + layer) % world_size] = 1
            rows.append(row)
        routing = RealizedRouting.from_lists(
            raw_selected_rows=rows,
            kept_rows=rows,
            dropped_rows=[[0] * world_size for _ in range(world_size)],
            padding_rows=[[0] * world_size for _ in range(world_size)],
        )
        metadata = {}
        if layer == 0:
            next_rows = []
            for next_src in range(world_size):
                next_row = [0] * world_size
                next_row[(next_src + 2) % world_size] = 1
                next_rows.append(tuple(next_row))
            metadata[FATE_METADATA_KEY] = canonical_fate_metadata(
                predictor_id="test-fate",
                source_layer_id=0,
                target_layer_id=1,
                confidence_ppm=1_000_000,
                routing_rows=tuple(next_rows),
                estimator_kind="TEST_GATE_PREDICTION",
                source_artifact_digest=stable_digest({"world_size": world_size, "kind": "test-fate"}),
            )
        collector.record_window(
            window_id=f"w{layer}",
            layer_id=layer,
            request_id="request",
            decode_step=0,
            is_bootstrap_p0=(layer == 0),
            mapping=mapping,
            routing=routing,
            local_compute=compute,
            dispatch_payload_spec=dispatch,
            combine_payload_spec=combine,
            descriptor_metadata_spec=DescriptorMetadataSpec(
                fixed_header_bytes=1,
                per_destination_entry_bytes=1,
            ),
            metadata=metadata,
        )
    return collector.freeze()


@pytest.mark.parametrize("world_size", [8, 16])
def test_formal_current_p12_rscf_joint_scales_beyond_ep4(world_size: int):
    runtime = build_current_p12_integration_runtime(
        fixture_input=_sparse_fixture(world_size),
        run_id=f"ep{world_size}-formal-smoke",
        paired_instance_id=f"ep{world_size}-formal-smoke",
        staging_sensitivity="1.0X",
        release_mode="RANK_LOCAL",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
        max_task_bytes=1 << 20,
        runtime_profile=synthetic_runtime_profile(
            max_batch_tasks=world_size, local_assembly_latency_ns=1
        ),
    )
    try:
        runtime.run_to_completion(max_timestamps=100_000)
        runtime.assert_terminal()
        assert len(runtime.data_plane.completed_task_ids) == 4 * world_size
    finally:
        runtime.dispose()
