from __future__ import annotations

from rs.core.hashing import stable_hash_dict

from .contracts import RecordMetadata, TrafficInstance, TraceSample


def _identity_mapping(world_size: int) -> tuple[int, ...]:
    return tuple(range(int(world_size)))


def build_traffic_instance(
    *,
    trace_sample: TraceSample,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
    virtual_ep_size: int,
    metadata: RecordMetadata,
) -> TrafficInstance:
    mapping = _identity_mapping(int(virtual_ep_size))
    digest = stable_hash_dict(
        {
            "trace_sample_id": trace_sample.trace_sample_id,
            "virtual_ep_size": int(virtual_ep_size),
            "expert_to_rank_mapping": list(mapping),
            "P0_matrix": [list(row) for row in p0_matrix],
            "P1_matrix": [list(row) for row in p1_matrix],
            "P2_truth_matrix": [list(row) for row in p2_matrix],
        }
    )
    return TrafficInstance(
        instance_id=f"{trace_sample.trace_sample_id}:vep{virtual_ep_size}",
        trace_sample_id=trace_sample.trace_sample_id,
        virtual_ep_size=int(virtual_ep_size),
        expert_to_rank_mapping=mapping,
        P0_matrix=p0_matrix,
        P1_matrix=p1_matrix,
        P2_truth_matrix=p2_matrix,
        flow_granularity="compact_matrix_rows",
        bucketization="canonical_bucket_rows",
        cost_model_id="formal_replay_makespan",
        traffic_digest=digest,
        metadata=metadata,
        physical_world_size=1,
    )
