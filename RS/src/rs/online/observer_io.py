from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import EpExecutionTrace


def write_online_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    jsonl_path = base / f"{run_id}.jsonl"
    metadata_path = base / f"{run_id}_metadata.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for route_trace in trace.route_traces:
            for route_record in route_trace.route_records:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "route_record",
                            "layer_id": route_trace.layer_id,
                            "trace_origin": route_trace.trace_origin,
                            "future_information_mode": route_trace.future_information_mode,
                            "payload": route_record.to_dict(),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        for stage_timing in trace.stage_timings:
            handle.write(
                json.dumps(
                    {
                        "record_type": "rank_stage_timing",
                        "payload": stage_timing.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for expert_bucket in trace.expert_buckets:
            handle.write(
                json.dumps(
                    {
                        "record_type": "expert_bucket_record",
                        "payload": expert_bucket.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for route_trace in trace.online_route_traces:
            for route_record in route_trace.all_routes:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "online_route_record",
                            "layer_id": route_trace.layer_id,
                            "trace_origin": route_trace.trace_origin,
                            "future_information_mode": route_trace.future_information_mode,
                            "payload": route_record.to_dict(),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        for rank_manifest in trace.rank_manifests:
            handle.write(
                json.dumps(
                    {
                        "record_type": "rank_manifest",
                        "payload": rank_manifest.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for expert_placement in trace.expert_placements:
            handle.write(
                json.dumps(
                    {
                        "record_type": "expert_placement",
                        "payload": expert_placement.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for transport_operation in trace.transport_operations:
            handle.write(
                json.dumps(
                    {
                        "record_type": "transport_operation_record",
                        "payload": transport_operation.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for validation_result in trace.validation_results:
            handle.write(
                json.dumps(
                    {
                        "record_type": "metadata_validation_result",
                        "payload": validation_result.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    local_route_rows = 0
    remote_route_rows = 0
    for route_trace in trace.route_traces:
        for route_record in route_trace.route_records:
            if route_record.is_local_route:
                local_route_rows += int(route_record.payload_rows)
            if route_record.is_remote_route:
                remote_route_rows += int(route_record.payload_rows)
    for route_trace in trace.online_route_traces:
        for route_record in route_trace.all_routes:
            if route_record.is_local_route:
                local_route_rows += int(route_record.payload_rows)
            if route_record.is_remote_route:
                remote_route_rows += int(route_record.payload_rows)

    metadata_path.write_text(
        json.dumps(
            {
                **metadata,
                "trace_origin": trace.trace_origin,
                "future_information_mode": trace.future_information_mode,
                "trace_artifact_schema_version": 2,
                "route_trace_count": len(trace.route_traces),
                "stage_timing_count": len(trace.stage_timings),
                "expert_bucket_count": len(trace.expert_buckets),
                "online_route_trace_count": len(trace.online_route_traces),
                "rank_manifest_count": len(trace.rank_manifests),
                "expert_placement_count": len(trace.expert_placements),
                "transport_operation_count": len(trace.transport_operations),
                "metadata_validation_count": len(trace.validation_results),
                "local_route_rows": local_route_rows,
                "remote_route_rows": remote_route_rows,
                "jsonl_path": str(jsonl_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return jsonl_path, metadata_path
