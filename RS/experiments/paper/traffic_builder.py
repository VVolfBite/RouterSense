from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rs.core.hashing import stable_hash_dict

from .contracts import RecordMetadata, TrafficInstance, TraceSample
from .trace_dataset import load_trace_bundle, trace_bundle_to_trace_samples


def deterministic_expert_to_rank_mapping(*, model_id: str, layer_id: str, num_experts: int, virtual_ep_size: int) -> tuple[int, ...]:
    return tuple(
        int(stable_hash_dict({"model_id": model_id, "layer_id": layer_id, "expert_id": expert_id, "virtual_ep_size": virtual_ep_size}), 16) % int(virtual_ep_size)
        for expert_id in range(int(num_experts))
    )


def _source_rank_for_token(*, sample_id: str, layer_id: str, token_position: int, virtual_ep_size: int) -> int:
    return int(stable_hash_dict({"sample_id": sample_id, "layer_id": layer_id, "token_position": token_position, "virtual_ep_size": virtual_ep_size}), 16) % int(virtual_ep_size)


def _zero_matrix(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(int(size))) for _ in range(int(size)))


def _matrix_from_records(
    *,
    records: list[dict[str, Any]],
    sample_id: str,
    layer_id: str,
    mapping: tuple[int, ...],
    virtual_ep_size: int,
) -> tuple[tuple[int, ...], ...]:
    rows = [[0 for _ in range(int(virtual_ep_size))] for _ in range(int(virtual_ep_size))]
    for record in records:
        src = _source_rank_for_token(
            sample_id=str(sample_id),
            layer_id=str(layer_id),
            token_position=int(record["token_position"]),
            virtual_ep_size=int(virtual_ep_size),
        )
        dst = int(mapping[int(record["expert_id"])])
        rows[src][dst] += 1
    return tuple(tuple(int(v) for v in row) for row in rows)


def _transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(matrix[src][dst]) for src in range(len(matrix))) for dst in range(len(matrix)))


def build_traffic_instance(
    *,
    trace_sample: TraceSample,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
    virtual_ep_size: int,
    metadata: RecordMetadata,
    mapping: tuple[int, ...] | None = None,
    cost_model_id: str = "formal_replay_makespan",
) -> TrafficInstance:
    mapping = mapping or deterministic_expert_to_rank_mapping(
        model_id=trace_sample.model_id,
        layer_id=trace_sample.layer_id,
        num_experts=trace_sample.num_experts,
        virtual_ep_size=int(virtual_ep_size),
    )
    mapping_digest = stable_hash_dict({"mapping": list(mapping)})
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
        expert_to_rank_mapping=tuple(int(item) for item in mapping),
        mapping_digest=mapping_digest,
        P0_matrix=p0_matrix,
        P1_matrix=p1_matrix,
        P2_truth_matrix=p2_matrix,
        flow_granularity="compact_matrix_rows",
        bucketization="canonical_bucket_rows",
        cost_model_id=str(cost_model_id),
        traffic_digest=digest,
        metadata=metadata,
        physical_world_size=1,
        source_trace_bundle=str(trace_sample.trace_bundle_path),
    )


def build_traffic_instances_from_trace_bundle(
    *,
    bundle_dir: Path,
    virtual_ep_sizes: tuple[int, ...],
    selected_layers: set[str] | None,
    metadata: RecordMetadata,
    cost_model_id: str,
) -> tuple[list[TraceSample], list[TrafficInstance]]:
    payload = load_trace_bundle(bundle_dir)
    records_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in payload["records"]:
        records_by_key[(str(row["sample_id"]), str(row["layer_id"]))].append(row)
    trace_samples = trace_bundle_to_trace_samples(bundle_dir, metadata=metadata)
    sample_lookup = {(sample.prompt_id, sample.layer_id): sample for sample in trace_samples}
    traffic_instances: list[TrafficInstance] = []
    for (sample_id, layer_id), sample in sorted(sample_lookup.items()):
        if selected_layers is not None and str(layer_id) not in selected_layers:
            continue
        current_records = records_by_key[(sample_id, layer_id)]
        next_records = records_by_key.get((sample_id, str(int(layer_id) + 1)), [])
        for virtual_ep_size in virtual_ep_sizes:
            mapping = deterministic_expert_to_rank_mapping(
                model_id=sample.model_id,
                layer_id=sample.layer_id,
                num_experts=sample.num_experts,
                virtual_ep_size=int(virtual_ep_size),
            )
            p0 = _matrix_from_records(
                records=current_records,
                sample_id=sample_id,
                layer_id=layer_id,
                mapping=mapping,
                virtual_ep_size=int(virtual_ep_size),
            )
            p1 = _transpose(p0)
            p2 = _zero_matrix(int(virtual_ep_size))
            if next_records:
                p2 = _matrix_from_records(
                    records=next_records,
                    sample_id=sample_id,
                    layer_id=str(int(layer_id) + 1),
                    mapping=mapping,
                    virtual_ep_size=int(virtual_ep_size),
                )
            traffic_instances.append(
                build_traffic_instance(
                    trace_sample=sample,
                    p0_matrix=p0,
                    p1_matrix=p1,
                    p2_matrix=p2,
                    virtual_ep_size=int(virtual_ep_size),
                    metadata=metadata,
                    mapping=mapping,
                    cost_model_id=cost_model_id,
                )
            )
    return trace_samples, traffic_instances
