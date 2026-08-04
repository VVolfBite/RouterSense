"""Import legacy RouterSense source-expert-count artifacts.

The source-expert-count importer supports multiple files, samples, layers, and ranks. It uses
explicit raw/kept/dropped/padding fields when present and never guesses missing
clipping or padding. Legacy rows that only contain already-realized counts are
represented as raw=kept and dropped=padding=0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..schema.canonical import stable_digest
from ..schema.model import RankNodeExpertMapping, RealizedRouting, TraceValidationError


@dataclass(frozen=True)
class ImportedRoutingRecord:
    sample_id: str
    request_id: str
    decode_step: int
    layer_id: int
    mapping: RankNodeExpertMapping
    routing: RealizedRouting
    source_artifact_digest: str
    source_paths: tuple[str, ...]

    def digest(self) -> str:
        return stable_digest(asdict(self), prefix="imported-routing")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TraceValidationError(f"invalid JSONL object {path}:{line_no}: expected mapping")
        payload = dict(payload)
        payload["__source_path"] = str(path)
        payload["__source_line"] = line_no
        rows.append(payload)
    return rows


def _sample_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    request_id = str(row.get("request_id", "")).strip()
    decode_step = int(row.get("decode_step", row.get("step", 0)) or 0)
    explicit = str(row.get("sample_id", "")).strip()
    if explicit:
        return explicit, request_id or explicit, decode_step
    if request_id:
        return f"{request_id}:step{decode_step}", request_id, decode_step
    run_digest = str(row.get("run_id_digest", "")).strip()
    if run_digest:
        return f"run:{run_digest}:step{decode_step}", f"run:{run_digest}", decode_step
    return "legacy-default:step0", "legacy-default", 0


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _source_vector(
    value: Any,
    *,
    field_name: str,
    source_rank: int,
    world_size: int,
    num_experts: int,
) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise TraceValidationError(f"{field_name} must be a non-empty vector or matrix")
    if all(not isinstance(item, (list, tuple)) for item in value):
        vector = [int(item) for item in value]
    else:
        rows = list(value)
        if len(rows) == world_size:
            vector = [int(item) for item in rows[source_rank]]
        elif len(rows) == 1:
            vector = [int(item) for item in rows[0]]
        else:
            raise TraceValidationError(
                f"{field_name} matrix rows={len(rows)} must be world_size={world_size} or 1"
            )
    if len(vector) != num_experts:
        raise TraceValidationError(
            f"{field_name} source row length {len(vector)} != num_experts {num_experts}"
        )
    if any(value < 0 for value in vector):
        raise TraceValidationError(f"{field_name} contains negative values")
    return vector


def _merge_vector(target: list[list[int] | None], source_rank: int, vector: list[int], *, label: str) -> None:
    existing = target[source_rank]
    if existing is not None and existing != vector:
        raise TraceValidationError(f"conflicting duplicate {label} source row {source_rank}")
    target[source_rank] = vector


def merge_source_expert_count_records(
    paths: Iterable[Path],
    *,
    rank_to_node: Iterable[int] | None = None,
) -> tuple[ImportedRoutingRecord, ...]:
    """Return one explicit realized-routing record per sample/layer."""

    all_rows: list[dict[str, Any]] = []
    for path in paths:
        all_rows.extend(_read_jsonl(Path(path)))
    if not all_rows:
        raise TraceValidationError("no RouterSense source-expert-count rows found")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    identity_fields: dict[tuple[str, int], tuple[str, int]] = {}
    for row in all_rows:
        sample_id, request_id, decode_step = _sample_identity(row)
        layer_id = int(row["layer_id"])
        key = (sample_id, layer_id)
        grouped[key].append(row)
        identity_fields[key] = (request_id, decode_step)

    explicit_rank_to_node = None if rank_to_node is None else tuple(int(v) for v in rank_to_node)
    output: list[ImportedRoutingRecord] = []
    for (sample_id, layer_id), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        world_sizes = {int(row["world_size"]) for row in rows}
        num_experts_set = {int(row["num_experts"]) for row in rows}
        if len(world_sizes) != 1 or len(num_experts_set) != 1:
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: inconsistent world_size/num_experts"
            )
        world_size = next(iter(world_sizes))
        num_experts = next(iter(num_experts_set))
        kept_rows: list[list[int] | None] = [None] * world_size
        raw_rows: list[list[int] | None] = [None] * world_size
        dropped_rows: list[list[int] | None] = [None] * world_size
        padding_rows: list[list[int] | None] = [None] * world_size
        seen_sources: set[int] = set()
        expert_maps: set[tuple[int, ...]] = set()
        embedded_node_maps: set[tuple[int, ...]] = set()

        for row in rows:
            source_rank = int(row.get("source_rank", row.get("rank", -1)))
            if source_rank < 0 or source_rank >= world_size:
                raise TraceValidationError(
                    f"sample={sample_id} layer={layer_id}: invalid source_rank={source_rank}"
                )
            kept = _source_vector(
                _first_present(
                    row,
                    (
                        "kept_rows",
                        "realized_source_expert_counts",
                        "source_expert_counts",
                        "counts",
                    ),
                ),
                field_name="kept/source_expert_counts",
                source_rank=source_rank,
                world_size=world_size,
                num_experts=num_experts,
            )
            if kept is None:
                raise TraceValidationError(
                    f"sample={sample_id} layer={layer_id}: missing kept/realized source-expert counts"
                )
            raw = _source_vector(
                _first_present(row, ("raw_selected_rows", "raw_source_expert_counts", "pre_capacity_source_expert_counts")),
                field_name="raw_selected_rows",
                source_rank=source_rank,
                world_size=world_size,
                num_experts=num_experts,
            )
            dropped = _source_vector(
                _first_present(row, ("dropped_rows", "dropped_source_expert_counts")),
                field_name="dropped_rows",
                source_rank=source_rank,
                world_size=world_size,
                num_experts=num_experts,
            )
            padding = _source_vector(
                _first_present(row, ("padding_rows", "padding_source_expert_counts")),
                field_name="padding_rows",
                source_rank=source_rank,
                world_size=world_size,
                num_experts=num_experts,
            )
            if raw is None and dropped is None:
                raw = list(kept)
                dropped = [0] * num_experts
            elif raw is None:
                assert dropped is not None
                raw = [kept[index] + dropped[index] for index in range(num_experts)]
            elif dropped is None:
                dropped = [raw[index] - kept[index] for index in range(num_experts)]
                if any(value < 0 for value in dropped):
                    raise TraceValidationError(
                        f"sample={sample_id} layer={layer_id}: raw_selected smaller than kept"
                    )
            padding = [0] * num_experts if padding is None else padding
            if any(raw[index] != kept[index] + dropped[index] for index in range(num_experts)):
                raise TraceValidationError(
                    f"sample={sample_id} layer={layer_id}: explicit raw/kept/dropped closure failed for source={source_rank}"
                )
            _merge_vector(kept_rows, source_rank, kept, label="kept")
            _merge_vector(raw_rows, source_rank, raw, label="raw")
            _merge_vector(dropped_rows, source_rank, dropped, label="dropped")
            _merge_vector(padding_rows, source_rank, padding, label="padding")
            seen_sources.add(source_rank)

            expert_map = row.get("expert_to_rank_map") or row.get("expert_to_ep_local_rank_map")
            if expert_map is not None:
                expert_maps.add(tuple(int(value) for value in expert_map))
            if row.get("rank_to_node") is not None:
                embedded_node_maps.add(tuple(int(value) for value in row["rank_to_node"]))

        if seen_sources != set(range(world_size)):
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: missing source ranks "
                f"{sorted(set(range(world_size)) - seen_sources)}"
            )
        if len(expert_maps) != 1:
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: expected exactly one expert_to_rank_map, found {len(expert_maps)}"
            )
        expert_to_rank = next(iter(expert_maps))
        if len(expert_to_rank) != num_experts:
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: expert map length {len(expert_to_rank)} != {num_experts}"
            )
        if explicit_rank_to_node is not None:
            node_map = explicit_rank_to_node
            if embedded_node_maps and embedded_node_maps != {node_map}:
                raise TraceValidationError(
                    f"sample={sample_id} layer={layer_id}: explicit rank_to_node conflicts with embedded mapping"
                )
        elif len(embedded_node_maps) == 1 and all(row.get("rank_to_node") is not None for row in rows):
            node_map = next(iter(embedded_node_maps))
        else:
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: rank_to_node is required and must not be guessed"
            )
        if len(node_map) != world_size:
            raise TraceValidationError(
                f"sample={sample_id} layer={layer_id}: rank_to_node length mismatch"
            )

        assert all(row is not None for row in kept_rows + raw_rows + dropped_rows + padding_rows)
        routing = RealizedRouting(
            raw_selected_rows=tuple(tuple(value for value in row or ()) for row in raw_rows),
            kept_rows=tuple(tuple(value for value in row or ()) for row in kept_rows),
            dropped_rows=tuple(tuple(value for value in row or ()) for row in dropped_rows),
            padding_rows=tuple(tuple(value for value in row or ()) for row in padding_rows),
            realization_origin=(
                "routesense_explicit_realization_truth"
                if any(sum(row or ()) for row in dropped_rows + padding_rows)
                else "routesense_already_realized_source_expert_counts"
            ),
        )
        mapping = RankNodeExpertMapping(
            world_size=world_size,
            rank_to_node=node_map,
            expert_to_rank=expert_to_rank,
            mapping_name="routesense_source_expert_counts_adapter",
        )
        source_paths = tuple(sorted({str(row["__source_path"]) for row in rows}))
        request_id, decode_step = identity_fields[(sample_id, layer_id)]
        output.append(
            ImportedRoutingRecord(
                sample_id=sample_id,
                request_id=request_id,
                decode_step=decode_step,
                layer_id=layer_id,
                mapping=mapping,
                routing=routing,
                source_artifact_digest=stable_digest(
                    sorted(
                        (
                            {key: value for key, value in row.items() if not key.startswith("__")}
                            for row in rows
                        ),
                        key=lambda item: (
                            int(item.get("source_rank", item.get("rank", -1))),
                            json.dumps(item, sort_keys=True, separators=(",", ":")),
                        ),
                    ),
                    prefix="routesense-source",
                ),
                source_paths=source_paths,
            )
        )
    return tuple(output)


def merge_source_expert_count_files(
    paths: Iterable[Path],
    *,
    rank_to_node: Iterable[int] | None = None,
) -> dict[int, tuple[RankNodeExpertMapping, RealizedRouting]]:
    """Backward-compatible single-sample layer map.

    Call :func:`merge_source_expert_count_records` when multiple samples are
    present. This wrapper fails closed rather than silently mixing samples.
    """

    records = merge_source_expert_count_records(paths, rank_to_node=rank_to_node)
    sample_ids = {record.sample_id for record in records}
    if len(sample_ids) != 1:
        raise TraceValidationError(
            f"multiple samples found {sorted(sample_ids)}; use merge_source_expert_count_records"
        )
    return {record.layer_id: (record.mapping, record.routing) for record in records}


__all__ = [
    "ImportedRoutingRecord",
    "merge_source_expert_count_files",
    "merge_source_expert_count_records",
]
