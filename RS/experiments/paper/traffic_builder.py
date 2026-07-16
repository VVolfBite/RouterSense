from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rs.core.hashing import stable_hash_dict

from .contracts import RecordMetadata, TrafficInstance, TraceSample
from .trace_dataset import load_trace_bundle, trace_bundle_to_trace_samples


DEFAULT_SOURCE_OWNERSHIP_POLICY_ID = "stable_token_owner_v1"
DEFAULT_PLACEMENT_POLICY_ID = "contiguous_balanced"


def stable_token_owner_v1(
    *,
    model_id: str,
    model_revision: str,
    prompt_id: str,
    batch_id: str,
    token_position: int,
    virtual_ep_size: int,
) -> int:
    return int(
        stable_hash_dict(
            {
                "owner_policy": DEFAULT_SOURCE_OWNERSHIP_POLICY_ID,
                "model_id": model_id,
                "model_revision": model_revision,
                "prompt_id": prompt_id,
                "batch_id": batch_id,
                "token_position": int(token_position),
                "virtual_ep_size": int(virtual_ep_size),
            }
        ),
        16,
    ) % int(virtual_ep_size)


def contiguous_balanced_expert_to_rank_mapping(*, num_experts: int, virtual_ep_size: int) -> tuple[int, ...]:
    experts = int(num_experts)
    vep = int(virtual_ep_size)
    return tuple(int(expert_id * vep / experts) for expert_id in range(experts))


def round_robin_expert_to_rank_mapping(*, num_experts: int, virtual_ep_size: int) -> tuple[int, ...]:
    return tuple(int(expert_id % int(virtual_ep_size)) for expert_id in range(int(num_experts)))


def _actual_exported_mapping(*, payload: dict[str, Any], layer_id: str) -> tuple[int, ...] | None:
    for candidate in (
        payload.get("summary", {}).get("expert_to_rank_mapping_by_layer"),
        payload.get("architecture_probe", {}).get("expert_to_rank_mapping_by_layer"),
        payload.get("run_manifest", {}).get("expert_to_rank_mapping_by_layer"),
    ):
        if not isinstance(candidate, dict):
            continue
        row = candidate.get(str(layer_id))
        if isinstance(row, list):
            return tuple(int(item) for item in row)
    return None


def resolve_expert_to_rank_mapping(
    *,
    payload: dict[str, Any],
    layer_id: str,
    num_experts: int,
    virtual_ep_size: int,
    placement_policy_id: str | None = None,
) -> tuple[str, tuple[int, ...]]:
    actual = _actual_exported_mapping(payload=payload, layer_id=layer_id)
    if actual is not None:
        return ("actual_exported", actual)
    policy = str(placement_policy_id or DEFAULT_PLACEMENT_POLICY_ID)
    if policy == "contiguous_balanced":
        return (policy, contiguous_balanced_expert_to_rank_mapping(num_experts=num_experts, virtual_ep_size=virtual_ep_size))
    if policy == "round_robin":
        return (policy, round_robin_expert_to_rank_mapping(num_experts=num_experts, virtual_ep_size=virtual_ep_size))
    raise ValueError(f"unsupported placement_policy_id {placement_policy_id!r}")


def _zero_matrix(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(int(size))) for _ in range(int(size)))


def _transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(matrix[src][dst]) for src in range(len(matrix))) for dst in range(len(matrix)))


def _matrix_from_records(
    *,
    records: list[dict[str, Any]],
    trace_sample: TraceSample,
    mapping: tuple[int, ...],
    virtual_ep_size: int,
) -> tuple[tuple[int, ...], ...]:
    rows = [[0 for _ in range(int(virtual_ep_size))] for _ in range(int(virtual_ep_size))]
    for record in records:
        src = stable_token_owner_v1(
            model_id=trace_sample.model_id,
            model_revision=trace_sample.model_revision,
            prompt_id=trace_sample.prompt_id,
            batch_id=trace_sample.batch_id,
            token_position=int(record["token_position"]),
            virtual_ep_size=int(virtual_ep_size),
        )
        dst = int(mapping[int(record["expert_id"])])
        rows[src][dst] += 1
    return tuple(tuple(int(v) for v in row) for row in rows)


def _mapping_digest(*, mapping: tuple[int, ...], policy_id: str, layer_id: str) -> str:
    return stable_hash_dict(
        {
            "placement_policy_id": policy_id,
            "layer_id": str(layer_id),
            "expert_to_rank_mapping": list(mapping),
        }
    )


def build_traffic_instance(
    *,
    trace_sample: TraceSample,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
    virtual_ep_size: int,
    metadata: RecordMetadata,
    mapping: tuple[int, ...] | None = None,
    placement_policy_id: str | None = None,
    current_layer_mapping_digest: str | None = None,
    target_layer_mapping_digest: str | None = None,
    cost_model_id: str = "formal_replay_makespan",
) -> TrafficInstance:
    resolved_mapping = tuple(mapping or contiguous_balanced_expert_to_rank_mapping(num_experts=trace_sample.num_experts, virtual_ep_size=virtual_ep_size))
    resolved_policy_id = str(placement_policy_id or DEFAULT_PLACEMENT_POLICY_ID)
    resolved_current_digest = str(
        current_layer_mapping_digest
        or _mapping_digest(mapping=resolved_mapping, policy_id=resolved_policy_id, layer_id=trace_sample.layer_id)
    )
    resolved_target_digest = str(target_layer_mapping_digest or resolved_current_digest)
    digest = stable_hash_dict(
        {
            "trace_sample_id": trace_sample.trace_sample_id,
            "virtual_ep_size": int(virtual_ep_size),
            "placement_policy_id": resolved_policy_id,
            "source_ownership_policy_id": DEFAULT_SOURCE_OWNERSHIP_POLICY_ID,
            "current_layer_mapping_digest": resolved_current_digest,
            "target_layer_mapping_digest": resolved_target_digest,
            "expert_to_rank_mapping": list(resolved_mapping),
            "P0_matrix": [list(row) for row in p0_matrix],
            "P1_matrix": [list(row) for row in p1_matrix],
            "P2_truth_matrix": [list(row) for row in p2_matrix],
        }
    )
    return TrafficInstance(
        instance_id=f"{trace_sample.trace_sample_id}:vep{virtual_ep_size}",
        trace_sample_id=trace_sample.trace_sample_id,
        virtual_ep_size=int(virtual_ep_size),
        expert_to_rank_mapping=tuple(int(item) for item in resolved_mapping),
        mapping_digest=resolved_current_digest,
        placement_policy_id=resolved_policy_id,
        source_ownership_policy_id=DEFAULT_SOURCE_OWNERSHIP_POLICY_ID,
        current_layer_mapping_digest=resolved_current_digest,
        target_layer_mapping_digest=resolved_target_digest,
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


def summarize_ownership_and_placement(*, traffic_instances: list[TrafficInstance]) -> dict[str, Any]:
    return {
        "source_ownership_policy_ids": sorted({item.source_ownership_policy_id for item in traffic_instances}),
        "placement_policy_ids": sorted({item.placement_policy_id for item in traffic_instances}),
        "virtual_ep_sizes": sorted({int(item.virtual_ep_size) for item in traffic_instances}),
        "current_layer_mapping_digests": sorted({item.current_layer_mapping_digest for item in traffic_instances}),
        "target_layer_mapping_digests": sorted({item.target_layer_mapping_digest for item in traffic_instances}),
    }


def build_traffic_instances_from_trace_bundle(
    *,
    bundle_dir: Path,
    virtual_ep_sizes: tuple[int, ...],
    selected_layers: set[str] | None,
    metadata: RecordMetadata,
    cost_model_id: str,
    placement_policy_id: str | None = None,
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
        target_num_experts = sample.num_experts
        next_sample = sample_lookup.get((sample_id, str(int(layer_id) + 1)))
        if next_sample is not None:
            target_num_experts = next_sample.num_experts
        for virtual_ep_size in virtual_ep_sizes:
            current_policy_id, current_mapping = resolve_expert_to_rank_mapping(
                payload=payload,
                layer_id=sample.layer_id,
                num_experts=sample.num_experts,
                virtual_ep_size=int(virtual_ep_size),
                placement_policy_id=placement_policy_id,
            )
            target_policy_id, target_mapping = resolve_expert_to_rank_mapping(
                payload=payload,
                layer_id=str(int(layer_id) + 1),
                num_experts=target_num_experts,
                virtual_ep_size=int(virtual_ep_size),
                placement_policy_id=placement_policy_id,
            )
            if current_policy_id != target_policy_id:
                raise ValueError("current and target placement policies must match within one TrafficInstance")
            current_digest = _mapping_digest(mapping=current_mapping, policy_id=current_policy_id, layer_id=sample.layer_id)
            target_digest = _mapping_digest(mapping=target_mapping, policy_id=target_policy_id, layer_id=str(int(layer_id) + 1))
            p0 = _matrix_from_records(
                records=current_records,
                trace_sample=sample,
                mapping=current_mapping,
                virtual_ep_size=int(virtual_ep_size),
            )
            p1 = _transpose(p0)
            p2 = _zero_matrix(int(virtual_ep_size))
            if next_records:
                p2 = _matrix_from_records(
                    records=next_records,
                    trace_sample=sample,
                    mapping=target_mapping,
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
                    mapping=current_mapping,
                    placement_policy_id=current_policy_id,
                    current_layer_mapping_digest=current_digest,
                    target_layer_mapping_digest=target_digest,
                    cost_model_id=cost_model_id,
                )
            )
    return trace_samples, traffic_instances
