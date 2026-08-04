"""Convert rank-local capture artifacts into RS-SIM FixtureInput files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from rs_sim.trace.adapters.routesense_source_expert_counts import (
    ImportedRoutingRecord,
    merge_source_expert_count_records,
)
from rs_sim.trace.schema.canonical import stable_digest
from rs_sim.trace.build.collector import TraceCollector
from rs_sim.trace.schema.constants import PHASE_COMBINE, PHASE_DISPATCH
from rs_sim.trace.schema.model import (
    DatasetProvenance,
    DescriptorMetadataSpec,
    LocalComputeProfile,
    PayloadSpec,
    PureComputeProvenance,
    TraceValidationError,
)
from rs_sim.trace.io.serialization import write_fixture
from rs_sim.trace.schema.validation import validate_fixture

from .fate_artifacts import load_fate_bundle, validate_fate_record_digest
from rs_sim.scheduler.prediction.fate_p2 import canonical_fate_metadata


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"invalid capture JSONL {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise TraceValidationError(f"capture JSONL row must be object: {path}:{line_no}")
        rows.append(value)
    return rows


def _artifact_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "fixture"


def _compute_rows(raw_dir: Path) -> dict[tuple[str, int, int, int], dict[str, list[int]]]:
    grouped: dict[tuple[str, int, int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(raw_dir.glob("*_local_compute.jsonl")):
        for row in _read_jsonl(path):
            fields = dict(row.get("field_values_ns", {}))
            key = (
                str(row.get("request_id", "")),
                int(row.get("decode_step", 0)),
                int(row["layer_id"]),
                int(row.get("source_rank", row.get("rank", -1))),
            )
            for name, value in fields.items():
                grouped[key][str(name)].append(int(value))
    return grouped


def _captured_fate_predictions(raw_dir: Path) -> tuple[dict[tuple[str, int, int], dict[str, Any]], str]:
    paths = tuple(sorted(raw_dir.glob("*_fate_p2_rows.jsonl")))
    if not paths:
        return {}, ""
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for line_index, row in enumerate(_read_jsonl(path), 1):
            if str(row.get("schema_version", "")) != "RS_SIM_CAPTURE_FATE_P2_ROW":
                raise TraceValidationError(f"unsupported captured FATE row schema in {path}")
            validate_fate_record_digest(
                row, context=f"captured FATE row {path}:{line_index}"
            )
            key = (
                str(row.get("sample_id", "")), int(row.get("source_layer_id", -1)),
                int(row.get("target_layer_id", -1)),
            )
            if not key[0] or key[1] < 0 or key[2] != key[1] + 1:
                raise TraceValidationError(f"invalid captured FATE identity {key}")
            grouped[key].append(row)
    source_digest = _artifact_digest(paths)
    output: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        world_sizes = {int(row.get("world_size", 0)) for row in rows}
        if len(world_sizes) != 1 or next(iter(world_sizes)) <= 0:
            raise TraceValidationError(f"captured FATE world_size mismatch for {key}")
        world_size = next(iter(world_sizes))
        by_source: dict[int, tuple[int, ...]] = {}
        identity_fields = (
            "schema_version", "capture_id", "request_id", "sample_id",
            "decode_step", "model_id", "world_size", "source_layer_id",
            "target_layer_id", "sample_token_count", "original_token_count",
            "top_k", "num_experts", "sampling_method", "estimator_kind",
            "predictor_id", "confidence_ppm",
        )
        identities = {
            field: {str(row.get(field, "")) for row in rows}
            for field in identity_fields
        }
        inconsistent = sorted(field for field, values in identities.items() if len(values) != 1)
        if inconsistent:
            raise TraceValidationError(
                f"captured FATE metadata differs across ranks for {key}: {inconsistent}"
            )
        predictor_ids = {str(row.get("predictor_id", "")) for row in rows}
        confidence_values = {int(row.get("confidence_ppm", -1)) for row in rows}
        estimator_values = {str(row.get("estimator_kind", "")) for row in rows}
        top_k = int(rows[0].get("top_k", 0))
        original_token_count = int(rows[0].get("original_token_count", 0))
        sample_token_count = int(rows[0].get("sample_token_count", 0))
        num_experts = int(rows[0].get("num_experts", 0))
        if top_k <= 0 or original_token_count <= 0 or sample_token_count <= 0 or num_experts <= 0:
            raise TraceValidationError(f"captured FATE dimensions must be positive for {key}")
        if sample_token_count > original_token_count or top_k > num_experts:
            raise TraceValidationError(f"captured FATE dimensions are inconsistent for {key}")
        for row in rows:
            src = int(row.get("source_rank", -1))
            values = tuple(int(item) for item in row.get("routing_rows_by_destination", ()))
            if not 0 <= src < world_size or len(values) != world_size or any(item < 0 for item in values):
                raise TraceValidationError(f"invalid captured FATE row for {key}, src={src}")
            expected_mass = original_token_count * top_k
            if sum(values) != expected_mass:
                raise TraceValidationError(
                    f"captured FATE row mass mismatch for {key}, src={src}: "
                    f"sum={sum(values)} expected={expected_mass}"
                )
            if src in by_source:
                raise TraceValidationError(f"duplicate captured FATE source row for {key}, src={src}")
            by_source[src] = values
        if set(by_source) != set(range(world_size)):
            raise TraceValidationError(f"captured FATE rows incomplete for {key}: {sorted(by_source)}")
        matrix = tuple(by_source[src] for src in range(world_size))
        output[key] = canonical_fate_metadata(
            predictor_id=next(iter(predictor_ids)), source_layer_id=key[1], target_layer_id=key[2],
            confidence_ppm=next(iter(confidence_values)), routing_rows=matrix,
            estimator_kind=next(iter(estimator_values)), source_artifact_digest=source_digest,
        )
    return output, source_digest


def _value_for(
    compute: dict[tuple[str, int, int, int], dict[str, list[int]]],
    *, request_id: str,
    decode_step: int,
    layer_id: int,
    rank: int,
    field: str,
    fallback: int,
    allow_fallback: bool,
) -> tuple[int, str]:
    rows = compute.get((request_id, decode_step, layer_id, rank), {}).get(field, [])
    if rows:
        return int(median(rows)), "MEASURED_AUTO"
    if not allow_fallback:
        raise TraceValidationError(
            f"missing compute field={field} request={request_id} step={decode_step} layer={layer_id} rank={rank}"
        )
    return int(fallback), "CONFIG_FALLBACK"


def _payload_spec(config: dict[str, Any], phase: str) -> PayloadSpec:
    value = config["payload"][phase]
    return PayloadSpec(
        phase_kind=PHASE_DISPATCH if phase == "dispatch" else PHASE_COMBINE,
        token_payload_bytes_per_row=int(value["token_payload_bytes_per_row"]),
        auxiliary_payload_bytes_per_row=int(value["auxiliary_payload_bytes_per_row"]),
        metadata_bytes_per_edge=int(value["metadata_bytes_per_edge"]),
        alignment_bytes=int(value["alignment_bytes"]),
        padding_rule=str(value["padding_rule"]),
        dtype=str(value["dtype"]),
    )


def build_fixtures_from_capture(config: dict[str, Any]) -> tuple[Path, ...]:
    output_dir = Path(config["output_dir"])
    raw_dir = output_dir / "raw"
    route_paths = tuple(sorted(raw_dir.glob("*_source_expert_counts.jsonl")))
    if not route_paths:
        raise TraceValidationError(f"no source-expert-count capture files found in {raw_dir}")
    explicit_nodes = config["capture"].get("rank_to_node")
    records = merge_source_expert_count_records(route_paths, rank_to_node=explicit_nodes)
    compute = _compute_rows(raw_dir)
    source_digest = _artifact_digest(
        tuple(route_paths)
        + tuple(sorted(raw_dir.glob("*_local_compute.jsonl")))
        + tuple(sorted(raw_dir.glob("*_fate_p2_rows.jsonl")))
    )
    fate_predictions: dict[tuple[str, int, int], dict[str, Any]] = {}
    fate_source_digest = ""
    prediction_config = config.get("prediction", {})
    fate_provider = str(prediction_config.get("provider", "EXTERNAL_ARTIFACT")).upper()
    fate_path = prediction_config.get("fate_artifact_path")
    if str(prediction_config.get("mode", "")).upper() == "FATE_P2":
        if fate_provider == "EXTERNAL_ARTIFACT":
            fate_predictions, fate_source_digest = load_fate_bundle(fate_path)
        elif fate_provider == "MEGATRON_SAMPLED_FATE":
            fate_predictions, fate_source_digest = _captured_fate_predictions(raw_dir)
        else:
            raise TraceValidationError(f"unsupported FATE provider {fate_provider!r}")
    transform_digest = stable_digest(
        {
            "payload": config["payload"],
            "fixture": config["fixture"],
            "prediction": config.get("prediction", {}),
            "capture_id": config["capture"]["capture_id"],
        },
        prefix="capture-transform",
    )
    provenance = DatasetProvenance(
        dataset_id=str(config["dataset"]["dataset_id"]),
        split=str(config["dataset"]["split"]),
        source_digest=source_digest,
        transform_digest=transform_digest,
        capture_id=str(config["capture"]["capture_id"]),
        collector_version=str(config["capture"]["collector_version"]),
        source_kind=str(config["dataset"]["source_kind"]),
        notes=str(config["dataset"].get("notes", "")),
    )
    dispatch_spec = _payload_spec(config, "dispatch")
    combine_spec = _payload_spec(config, "combine")
    descriptor_value = config["payload"]["descriptor"]
    descriptor_spec = DescriptorMetadataSpec(
        fixed_header_bytes=int(descriptor_value["fixed_header_bytes"]),
        per_destination_entry_bytes=int(descriptor_value["per_destination_entry_bytes"]),
    )
    by_sample: dict[str, list[ImportedRoutingRecord]] = defaultdict(list)
    for record in records:
        by_sample[record.sample_id].append(record)

    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    fallback = config["fixture"]["compute_fallback_ns"]
    allow_fallback = bool(config["fixture"].get("allow_compute_fallback", True))
    for sample_id, sample_records in sorted(by_sample.items()):
        sample_records.sort(key=lambda item: item.layer_id)
        if len(sample_records) < 2:
            raise TraceValidationError(
                f"sample={sample_id} contains {len(sample_records)} layer; Current P12 requires at least two consecutive layers"
            )
        if any(
            sample_records[index + 1].layer_id != sample_records[index].layer_id + 1
            for index in range(len(sample_records) - 1)
        ):
            raise TraceValidationError(f"sample={sample_id} layers must be consecutive for Current P12")
        collector = TraceCollector(
            fixture_id=f"{config['fixture']['fixture_id_prefix']}:{sample_id}",
            provenance=provenance,
        )
        for index, record in enumerate(sample_records):
            vectors: dict[str, list[int]] = {
                name: []
                for name in (
                    "combine_release_to_router_ready_ns",
                    "router_and_pack_ns",
                    "dispatch_local_postprocess_ns",
                    "dispatch_release_to_combine_source_ready_ns",
                    "bootstrap_router_and_pack_ns",
                )
            }
            quality: dict[str, list[str]] = {name: [] for name in vectors}
            for rank in range(record.mapping.world_size):
                for field in vectors:
                    lookup = compute.get(
                        (record.request_id, record.decode_step, record.layer_id, rank), {}
                    )
                    if field == "bootstrap_router_and_pack_ns" and not lookup.get(field):
                        router_rows = lookup.get("router_and_pack_ns", [])
                        if router_rows:
                            value, origin = int(median(router_rows)), "DERIVED_FROM_ROUTER_AND_PACK"
                        else:
                            value, origin = _value_for(
                                compute,
                                request_id=record.request_id,
                                decode_step=record.decode_step,
                                layer_id=record.layer_id,
                                rank=rank,
                                field=field,
                                fallback=int(fallback[field]),
                                allow_fallback=allow_fallback,
                            )
                    elif (
                        field == "combine_release_to_router_ready_ns"
                        and index == len(sample_records) - 1
                        and not lookup.get(field)
                    ):
                        # The final captured MoE layer has no next router.  This
                        # duration is outside every Current-P12 window and must
                        # not force a synthetic timing fallback.
                        value, origin = 0, "TRUNCATED_TAIL_NOT_CONSUMED"
                    else:
                        value, origin = _value_for(
                            compute,
                            request_id=record.request_id,
                            decode_step=record.decode_step,
                            layer_id=record.layer_id,
                            rank=rank,
                            field=field,
                            fallback=int(fallback[field]),
                            allow_fallback=allow_fallback,
                        )
                    vectors[field].append(value)
                    quality[field].append(origin)
            local_compute = LocalComputeProfile(
                combine_release_to_router_ready_ns=tuple(vectors["combine_release_to_router_ready_ns"]),
                router_and_pack_ns=tuple(vectors["router_and_pack_ns"]),
                dispatch_local_postprocess_ns=tuple(vectors["dispatch_local_postprocess_ns"]),
                dispatch_release_to_combine_source_ready_ns=tuple(vectors["dispatch_release_to_combine_source_ready_ns"]),
                bootstrap_router_and_pack_ns=tuple(vectors["bootstrap_router_and_pack_ns"]),
                provenance=PureComputeProvenance(
                    measurement_method="independent_capture_hybrid_cuda_event_and_explicit_fallback",
                    source_artifact_digest=source_digest,
                    included_components=(
                        "combine_postprocess_residual_norm_attention",
                        "router_and_pack",
                        "dispatch_local_postprocess",
                        "expert_compute_and_combine_pack",
                    ),
                ),
            )
            collector.record_window(
                window_id=f"{sample_id}:layer{record.layer_id}",
                layer_id=record.layer_id,
                request_id=record.request_id,
                decode_step=record.decode_step,
                is_bootstrap_p0=index == 0,
                mapping=record.mapping,
                routing=record.routing,
                local_compute=local_compute,
                dispatch_payload_spec=dispatch_spec,
                combine_payload_spec=combine_spec,
                descriptor_metadata_spec=descriptor_spec,
                metadata={
                    "capture_id": config["capture"]["capture_id"],
                    "model_id": config["capture"]["model_id"],
                    "model_path": config["capture"].get("model_path"),
                    "source_paths": list(record.source_paths),
                    "source_artifact_digest": record.source_artifact_digest,
                    "compute_field_quality": quality,
                    "performance_eligible": bool(config["capture"].get("performance_eligible", False)),
                    "performance_qualification": dict(config["capture"].get("performance_qualification", {})),
                    **(
                        {"fate_p2_prediction": fate_predictions[(sample_id, int(record.layer_id), int(record.layer_id) + 1)]}
                        if (sample_id, int(record.layer_id), int(record.layer_id) + 1) in fate_predictions
                        else {}
                    ),
                },
            )
        if config.get("prediction", {}).get("mode") == "FATE_P2" and bool(
            config.get("prediction", {}).get("require_complete_fate_coverage", True)
        ):
            missing = [
                (sample_id, int(item.layer_id), int(item.layer_id) + 1)
                for item in sample_records[:-1]
                if (sample_id, int(item.layer_id), int(item.layer_id) + 1) not in fate_predictions
            ]
            if missing:
                raise TraceValidationError(f"FATE artifact bundle missing Current-P12 windows: {missing}")
        fixture = collector.freeze()
        validate_fixture(fixture)
        path = fixture_dir / f"{_safe_name(sample_id)}.json"
        write_fixture(path, fixture)
        outputs.append(path)
    summary = {
        "schema_version": "RS_SIM_CAPTURE_FINALIZE_SUMMARY",
        "status": "PASS",
        "capture_id": config["capture"]["capture_id"],
        "source_digest": source_digest,
        "fixture_paths": [str(path) for path in outputs],
        "fixture_count": len(outputs),
        "performance_eligible": bool(config["capture"].get("performance_eligible", False)),
        "performance_qualification": dict(config["capture"].get("performance_qualification", {})),
        "fate_artifact_source_digest": fate_source_digest,
        "fate_prediction_count": len(fate_predictions),
    }
    (output_dir / "finalize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(outputs)
