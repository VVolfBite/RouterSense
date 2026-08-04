#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from rs_sim.scheduler.prediction.fate_p2 import canonical_fate_metadata
from rs_sim.scheduler.stable import stable_digest
from rs_sim.trace.io.serialization import fixture_from_dict, write_fixture
from rs_sim.trace.io.canonicalization import canonicalize_fixture_serialization
from rs_sim.trace.schema.validation import validate_fixture


def _split_count(count: int, parts: int, rotation: int) -> list[int]:
    q, r = divmod(int(count), int(parts))
    out = [q] * int(parts)
    for index in range(r):
        out[(int(rotation) + index) % int(parts)] += 1
    return out


def _split_source_matrix(matrix: list[list[int]], factor: int, salt: int) -> list[list[int]]:
    old_world = len(matrix)
    experts = len(matrix[0])
    out = [[0] * experts for _ in range(old_world * factor)]
    for src, row in enumerate(matrix):
        for expert, value in enumerate(row):
            pieces = _split_count(value, factor, (salt + src * 17 + expert * 31) % factor)
            for child, piece in enumerate(pieces):
                out[src * factor + child][expert] = piece
    return out


def _split_square_matrix(matrix: list[list[int]], factor: int, salt: int) -> list[list[int]]:
    old_world = len(matrix)
    new_world = old_world * factor
    out = [[0] * new_world for _ in range(new_world)]
    block = factor * factor
    for src in range(old_world):
        for dst in range(old_world):
            pieces = _split_count(matrix[src][dst], block, (salt + src * 19 + dst * 37) % block)
            for index, piece in enumerate(pieces):
                src_child = index // factor
                dst_child = index % factor
                out[src * factor + src_child][dst * factor + dst_child] = piece
    return out


def _balanced_expert_mapping(num_experts: int, world_size: int) -> list[int]:
    if int(num_experts) < int(world_size):
        raise ValueError("target EP cannot exceed expert count")
    return [(expert * int(world_size)) // int(num_experts) for expert in range(int(num_experts))]


def _project_compute(
    original_compute: dict[str, Any],
    parent_raw_rows: list[list[int]],
    child_raw_rows: list[list[int]],
    factor: int,
    target_world_size: int,
) -> dict[str, Any]:
    fields = (
        "combine_release_to_router_ready_ns",
        "router_and_pack_ns",
        "dispatch_local_postprocess_ns",
        "dispatch_release_to_combine_source_ready_ns",
        "bootstrap_router_and_pack_ns",
    )
    result = copy.deepcopy(original_compute)
    for field in fields:
        values: list[int] = []
        for child_rank in range(target_world_size):
            parent_rank = child_rank // factor
            parent_total = sum(int(v) for v in parent_raw_rows[parent_rank])
            child_total = sum(int(v) for v in child_raw_rows[child_rank])
            parent_time = int(original_compute[field][parent_rank])
            projected = parent_time if parent_total <= 0 else int(round(parent_time * child_total / parent_total))
            if parent_time > 0 and child_total > 0:
                projected = max(1, projected)
            values.append(projected)
        result[field] = values
    provenance = dict(result.get("provenance", {}))
    provenance["measurement_method"] = "PROJECTED_EP_BALANCED_SOURCE_SPLIT_PROPORTIONAL_COMPUTE_V1"
    provenance["source_artifact_digest"] = stable_digest({
        "source": provenance.get("source_artifact_digest", ""),
        "target_world_size": target_world_size,
        "factor": factor,
    })
    result["provenance"] = provenance
    return result


def project_fixture(source_path: Path, output_path: Path, target_world_size: int, ranks_per_node: int) -> dict[str, Any]:
    source_serialized = json.loads(source_path.read_text(encoding="utf-8"))
    source_truth = str(source_serialized.get("fixture_truth_digest", ""))
    payload = copy.deepcopy(source_serialized)
    payload.pop("fixture_truth_digest", None)
    payload, _ = canonicalize_fixture_serialization(payload)
    old_world = int(payload["windows"][0]["mapping"]["world_size"])
    if target_world_size % old_world:
        raise ValueError("target EP must be an integer multiple of source EP")
    factor = target_world_size // old_world
    if factor <= 1:
        raise ValueError("target EP must be larger than source EP")

    transform_spec = {
        "schema_version": "RS_SIM_CROSS_EP_PROJECTION_V1",
        "source_path": source_path.name,
        "source_truth_digest": source_truth,
        "source_world_size": old_world,
        "target_world_size": target_world_size,
        "factor": factor,
        "source_split": "DETERMINISTIC_BALANCED_INTEGER_SPLIT",
        "expert_mapping": "CONTIGUOUS_BALANCED_EXPERT_TO_RANK",
        "compute_projection": "PROPORTIONAL_TO_PROJECTED_SOURCE_ROUTING_ROWS",
        "fate_projection": "DETERMINISTIC_BALANCED_BLOCK_SPLIT",
        "ranks_per_node": ranks_per_node,
        "evidence_class": "PROJECTED_NOT_MEASURED",
    }
    transform_digest = stable_digest(transform_spec)

    provenance = payload["provenance"]
    provenance["dataset_id"] = f"{provenance['dataset_id']}:projected-ep{target_world_size}"
    provenance["source_digest"] = source_truth or str(provenance.get("source_digest", ""))
    provenance["transform_digest"] = transform_digest
    provenance["capture_id"] = f"{provenance['capture_id']}:projected-ep{target_world_size}"
    provenance["source_kind"] = "projected_cross_ep_from_measured_fate_trace"
    provenance["notes"] = (
        f"PROJECTED_NOT_MEASURED; source EP{old_world}; target EP{target_world_size}; "
        "routing and FATE counts are conserved under deterministic balanced splitting; "
        "pure-compute vectors are proportionally projected and are not hardware measurements."
    )
    payload["fixture_id"] = f"{payload['fixture_id']}:projected-ep{target_world_size}"
    payload["initial_state"]["bootstrap_source_ranks"] = list(range(target_world_size))

    for window_index, window in enumerate(payload["windows"]):
        routing = window["routing"]
        parent_raw = copy.deepcopy(routing["raw_selected_rows"])
        salt = int(window.get("layer_id", window_index)) * 1009 + window_index * 9176
        kept = _split_source_matrix(routing["kept_rows"], factor, salt + 1)
        dropped = _split_source_matrix(routing["dropped_rows"], factor, salt + 2)
        padding = _split_source_matrix(routing["padding_rows"], factor, salt + 3)
        raw = [[kept[r][e] + dropped[r][e] for e in range(len(kept[0]))] for r in range(target_world_size)]
        routing.update({
            "raw_selected_rows": raw,
            "kept_rows": kept,
            "dropped_rows": dropped,
            "padding_rows": padding,
            "realization_origin": "projected_cross_ep_balanced_source_split_v1",
        })

        num_experts = len(kept[0])
        window["mapping"].update({
            "world_size": target_world_size,
            "rank_to_node": [rank // int(ranks_per_node) for rank in range(target_world_size)],
            "expert_to_rank": _balanced_expert_mapping(num_experts, target_world_size),
            "mapping_name": f"projected_ep{target_world_size}_contiguous_expert_mapping_v1",
        })
        window["local_compute"] = _project_compute(
            window["local_compute"], parent_raw, raw, factor, target_world_size
        )
        original_window_id = str(window["window_id"])
        window["window_id"] = f"{original_window_id}:projected-ep{target_world_size}"

        metadata = window.setdefault("metadata", {})
        fate = metadata.get("fate_p2_prediction")
        if isinstance(fate, dict):
            common = {
                "predictor_id": f"{fate.get('predictor_id', 'fate')}:projected-ep{target_world_size}",
                "source_layer_id": int(fate["source_layer_id"]),
                "target_layer_id": int(fate["target_layer_id"]),
                "confidence_ppm": int(fate.get("confidence_ppm", 0)),
                "estimator_kind": f"{fate.get('estimator_kind', 'FATE')}:PROJECTED_BLOCK_SPLIT_V1",
                "source_artifact_digest": stable_digest({
                    "source_fate_digest": fate.get("artifact_digest", ""),
                    "transform_digest": transform_digest,
                    "window_index": window_index,
                }),
            }
            if fate.get("routing_rows") is not None:
                matrix = _split_square_matrix(fate["routing_rows"], factor, salt + 4)
                metadata["fate_p2_prediction"] = canonical_fate_metadata(
                    **common,
                    routing_rows=tuple(tuple(int(v) for v in row) for row in matrix),
                )
            elif fate.get("payload_matrix") is not None:
                matrix = _split_square_matrix(fate["payload_matrix"], factor, salt + 4)
                metadata["fate_p2_prediction"] = canonical_fate_metadata(
                    **common,
                    payload_matrix=tuple(tuple(int(v) for v in row) for row in matrix),
                )
        metadata["performance_eligible"] = False
        metadata["cross_ep_projection"] = {
            **transform_spec,
            "window_index": window_index,
            "original_window_id": original_window_id,
        }

    payload["initial_state"]["bootstrap_window_id"] = payload["windows"][0]["window_id"]
    payload["expected_invariants"] = {}
    fixture = fixture_from_dict(payload, regenerate_expected_invariants=True)
    validate_fixture(fixture)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_fixture(output_path, fixture)

    def total_raw(serialized_windows: list[dict[str, Any]]) -> int:
        return sum(
            sum(sum(int(value) for value in row) for row in window["routing"]["raw_selected_rows"])
            for window in serialized_windows
        )

    source_rows = total_raw(source_serialized["windows"])
    projected_rows = sum(
        sum(sum(int(value) for value in row) for row in window.routing.raw_selected_rows)
        for window in fixture.windows
    )
    return {
        **transform_spec,
        "output_path": str(output_path),
        "output_truth_digest": fixture.truth_digest(),
        "window_count": len(fixture.windows),
        "num_experts": fixture.windows[0].mapping.num_experts,
        "source_total_raw_rows": source_rows,
        "projected_total_raw_rows": projected_rows,
        "routing_rows_conserved": source_rows == projected_rows,
        "validated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Project a measured EP fixture to a larger synthetic EP size")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-ep", required=True, type=int)
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = project_fixture(args.source, args.output, args.target_ep, args.ranks_per_node)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
