from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .identity import canonical_instance_id

MODELS = ("olmoe", "qwen15moe", "deepseekv2lite")


@dataclass(frozen=True)
class TrafficInstance:
    model: str
    instance_id: str
    prompt_id: str
    layer: int
    split: str
    world_size: int
    p0_full: np.ndarray
    p1_full: np.ndarray
    p2_full: np.ndarray
    expert_to_rank: tuple[int, ...]
    placement_policy_id: str
    ownership_policy_id: str
    cost_model_id: str
    terminal_layer: bool | None = None

    def __post_init__(self) -> None:
        if not str(self.model) or not str(self.instance_id) or not str(self.prompt_id):
            raise ValueError("model, instance_id, and prompt_id must be non-empty")
        if int(self.layer) < 0:
            raise ValueError("layer must be non-negative")
        n = int(self.world_size)
        if n <= 0:
            raise ValueError("world_size must be positive")

        def freeze_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
            matrix = np.asarray(value)
            if matrix.shape != (n, n):
                raise ValueError(f"{name} shape {matrix.shape} != ({n}, {n})")
            if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must contain finite numeric values")
            if (matrix < 0).any():
                raise ValueError(f"{name} must be non-negative")
            rounded = np.rint(matrix)
            if not np.allclose(matrix, rounded, atol=0.0, rtol=0.0):
                raise ValueError(f"{name} must contain integral row counts")
            if rounded.size and float(rounded.max()) > float(np.iinfo(np.int32).max):
                raise ValueError(f"{name} exceeds int32 row-count range")
            output = np.ascontiguousarray(rounded.astype(np.int32, copy=False)).copy()
            output.setflags(write=False)
            return output

        object.__setattr__(self, "p0_full", freeze_matrix(self.p0_full, name="p0_full"))
        object.__setattr__(self, "p1_full", freeze_matrix(self.p1_full, name="p1_full"))
        object.__setattr__(self, "p2_full", freeze_matrix(self.p2_full, name="p2_full"))

        mapping = tuple(int(x) for x in self.expert_to_rank)
        if not mapping:
            raise ValueError("expert_to_rank must be non-empty")
        if min(mapping) < 0 or max(mapping) >= n:
            raise ValueError("expert_to_rank contains a rank outside world_size")
        object.__setattr__(self, "expert_to_rank", mapping)
        if self.terminal_layer is not None:
            object.__setattr__(self, "terminal_layer", bool(self.terminal_layer))

    @property
    def p0(self) -> np.ndarray:
        out = self.p0_full.copy(); np.fill_diagonal(out, 0); return out

    @property
    def p1(self) -> np.ndarray:
        out = self.p1_full.copy(); np.fill_diagonal(out, 0); return out

    @property
    def p2(self) -> np.ndarray:
        out = self.p2_full.copy(); np.fill_diagonal(out, 0); return out

    @property
    def is_last_layer(self) -> bool:
        """Whether the trace explicitly identifies this as the final MoE layer.

        Older hand-built fixtures did not carry terminal metadata, so they keep
        the legacy zero-P2 fallback. Loaded corpora use explicit per-prompt layer
        position and therefore do not confuse an all-local intermediate dispatch
        with model termination.
        """
        if self.terminal_layer is not None:
            return bool(self.terminal_layer)
        return int(self.p2_full.sum()) == 0


def load_instances(root: Path) -> list[TrafficInstance]:
    root = Path(root)
    out: list[TrafficInstance] = []
    for model in MODELS:
        path = root / model / "traffic" / "traffic_instances.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        parsed_identity: list[tuple[str, int]] = []
        max_layer_by_prompt: dict[str, int] = {}
        for row in rows:
            prompt_id, layer_text = str(row["trace_sample_id"]).rsplit(":", 1)
            layer = int(layer_text)
            parsed_identity.append((prompt_id, layer))
            max_layer_by_prompt[prompt_id] = max(max_layer_by_prompt.get(prompt_id, -1), layer)
        for row, (prompt_id, layer) in zip(rows, parsed_identity, strict=True):
            split = "validation" if prompt_id.startswith("val-") else "development"
            n = int(row["virtual_ep_size"])
            matrices = [np.asarray(row[key]) for key in ("P0_matrix", "P1_matrix", "P2_truth_matrix")]
            out.append(
                TrafficInstance(
                    model=model,
                    instance_id=canonical_instance_id(model, str(row["instance_id"])),
                    prompt_id=prompt_id,
                    layer=layer,
                    split=split,
                    world_size=n,
                    p0_full=matrices[0], p1_full=matrices[1], p2_full=matrices[2],
                    expert_to_rank=tuple(int(x) for x in row["expert_to_rank_mapping"]),
                    placement_policy_id=str(row.get("placement_policy_id", "unknown")),
                    ownership_policy_id=str(row.get("source_ownership_policy_id", "unknown")),
                    cost_model_id=str(row.get("cost_model_id", "unknown")),
                    terminal_layer=layer == max_layer_by_prompt[prompt_id],
                )
            )
    return out


def stratified(instances: list[TrafficInstance], split: str, per_cell: int | None = None) -> list[TrafficInstance]:
    rows = [x for x in instances if x.split == split]
    if per_cell is None:
        return rows
    output: list[TrafficInstance] = []
    cells = sorted({(x.model, x.world_size) for x in rows})
    for model, n in cells:
        cell = sorted((x for x in rows if x.model == model and x.world_size == n), key=lambda x: x.instance_id)
        output.extend(cell[: int(per_cell)])
    return output


def max_layer_by_model(instances: list[TrafficInstance]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in instances:
        result[item.model] = max(result.get(item.model, -1), int(item.layer))
    return result


__all__ = ["MODELS", "TrafficInstance", "load_instances", "stratified", "max_layer_by_model"]
