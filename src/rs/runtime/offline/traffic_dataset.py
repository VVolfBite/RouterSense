"""Portable loaders for RouterSense traffic packages.

The loader supports the current JSON traffic bundle and the compact NPZ/index
layout requested for multi-model vEP8/vEP12 trace packages. It validates the
scheduling semantics before returning an instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrafficInstanceRecord:
    traffic_instance_id: str
    sample_id: str
    split: str
    layer_id: int
    target_layer_id: int | None
    world_size: int
    p0: Matrix
    p1: Matrix
    p2: Matrix
    p2_available: bool
    model_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        matrices = {"p0": self.p0, "p1": self.p1, "p2": self.p2}
        for name, matrix in matrices.items():
            if len(matrix) != self.world_size or any(len(row) != self.world_size for row in matrix):
                raise ValueError(f"{name} shape does not match world_size")
            if any(int(value) < 0 for row in matrix for value in row):
                raise ValueError(f"{name} must be non-negative")
        expected_p1 = tuple(
            tuple(int(self.p0[destination][source]) for destination in range(self.world_size))
            for source in range(self.world_size)
        )
        if self.p1 != expected_p1:
            raise ValueError("P1 must equal transpose(P0)")
        if not self.p2_available and any(value for row in self.p2 for value in row):
            raise ValueError("unavailable P2 must be all zero")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "traffic_instance_id": self.traffic_instance_id,
            "sample_id": self.sample_id,
            "split": self.split,
            "layer_id": self.layer_id,
            "target_layer_id": self.target_layer_id,
            "world_size": self.world_size,
            "P0_dispatch_matrix": [list(row) for row in self.p0],
            "P1_return_matrix": [list(row) for row in self.p1],
            "P2_next_layer_dispatch_matrix": [list(row) for row in self.p2],
            "p2_available": self.p2_available,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }


def _matrix(value: Any) -> Matrix:
    matrix = tuple(tuple(int(cell) for cell in row) for row in value)
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    return matrix


def _from_json_row(row: dict[str, Any]) -> TrafficInstanceRecord:
    p0 = _matrix(row.get("P0_dispatch_matrix", row.get("p0")))
    p1 = _matrix(row.get("P1_return_matrix", row.get("p1")))
    p2 = _matrix(row.get("P2_next_layer_dispatch_matrix", row.get("p2")))
    metadata = dict(row.get("metadata", {}))
    record = TrafficInstanceRecord(
        traffic_instance_id=str(row.get("traffic_instance_id", row.get("instance_id", ""))),
        sample_id=str(row.get("sample_id", "")),
        split=str(row.get("split", "")),
        layer_id=int(row.get("layer_id", 0)),
        target_layer_id=(None if row.get("target_layer_id") is None else int(row["target_layer_id"])),
        world_size=int(row.get("virtual_ep_size", row.get("world_size", len(p0)))),
        p0=p0,
        p1=p1,
        p2=p2,
        p2_available=bool(row.get("p2_available", True)),
        model_id=str(row.get("model_id", metadata.get("model_id", ""))),
        metadata={key: value for key, value in row.items() if key not in {
            "P0_dispatch_matrix", "P1_return_matrix", "P2_next_layer_dispatch_matrix", "p0", "p1", "p2"
        }},
    )
    record.validate()
    return record


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("instances", "records", "traffic_instances"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key]]
    raise ValueError("unsupported traffic JSON schema")


def _load_npz_records(index_payload: Any, npz_bytes: bytes) -> list[TrafficInstanceRecord]:
    index_rows = _json_records(index_payload)
    with np.load(BytesIO(npz_bytes), allow_pickle=False) as arrays:
        keys = set(arrays.files)
        def pick(*names: str):
            for name in names:
                if name in keys:
                    return arrays[name]
            raise ValueError(f"NPZ missing one of {names}")
        p0_all = pick("P0", "p0", "P0_dispatch_matrix", "p0_matrices")
        p1_all = pick("P1", "p1", "P1_return_matrix", "p1_matrices")
        p2_all = pick("P2", "p2", "P2_next_layer_dispatch_matrix", "p2_matrices")
        if not (len(index_rows) == len(p0_all) == len(p1_all) == len(p2_all)):
            raise ValueError("NPZ matrix count does not match index")
        output = []
        for index, row in enumerate(index_rows):
            expanded = {
                **row,
                "P0_dispatch_matrix": p0_all[index].tolist(),
                "P1_return_matrix": p1_all[index].tolist(),
                "P2_next_layer_dispatch_matrix": p2_all[index].tolist(),
            }
            output.append(_from_json_row(expanded))
        return output


def _filter(records: Iterable[TrafficInstanceRecord], *, split: str | None, world_sizes: set[int] | None) -> list[TrafficInstanceRecord]:
    return [
        row for row in records
        if (split is None or row.split == split)
        and (world_sizes is None or row.world_size in world_sizes)
    ]


def load_traffic_instances(
    source: str | Path,
    *,
    split: str | None = None,
    world_sizes: Iterable[int] | None = None,
) -> list[TrafficInstanceRecord]:
    """Load validated traffic records from JSON, directory, or ZIP package."""

    path = Path(source)
    worlds = None if world_sizes is None else {int(value) for value in world_sizes}
    records: list[TrafficInstanceRecord] = []
    if path.is_dir():
        candidates = sorted(path.glob("**/*instances.json"))
        for candidate in candidates:
            records.extend(_from_json_row(row) for row in _json_records(json.loads(candidate.read_text(encoding="utf-8"))))
        for index_path in sorted(path.glob("**/world*_index.json")):
            npz_path = index_path.with_name(index_path.name.replace("_index.json", "_instances.npz"))
            if npz_path.exists():
                records.extend(_load_npz_records(json.loads(index_path.read_text(encoding="utf-8")), npz_path.read_bytes()))
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for name in sorted(names):
                if name.endswith("_instances.json") or name.endswith("development_instances.json") or name.endswith("validation_instances.json"):
                    payload = json.loads(archive.read(name).decode("utf-8"))
                    records.extend(_from_json_row(row) for row in _json_records(payload))
            for index_name in sorted(name for name in names if name.endswith("_index.json") and "/traffic/" in f"/{name}"):
                npz_name = index_name.replace("_index.json", "_instances.npz")
                if npz_name not in names:
                    continue
                records.extend(_load_npz_records(json.loads(archive.read(index_name).decode("utf-8")), archive.read(npz_name)))
    elif path.suffix.lower() == ".json":
        records.extend(_from_json_row(row) for row in _json_records(json.loads(path.read_text(encoding="utf-8"))))
    else:
        raise ValueError(f"unsupported traffic source {path}")
    # De-duplicate packages that expose the same records through more than one
    # index while failing closed on conflicting contents.
    dedup: dict[str, TrafficInstanceRecord] = {}
    for record in records:
        key = record.traffic_instance_id
        if key in dedup and dedup[key] != record:
            raise ValueError(f"conflicting duplicate traffic instance {key}")
        dedup[key] = record
    return _filter(dedup.values(), split=split, world_sizes=worlds)


__all__ = ["TrafficInstanceRecord", "load_traffic_instances"]
