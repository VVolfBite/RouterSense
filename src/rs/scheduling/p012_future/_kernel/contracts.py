from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

MatrixKind = Literal["remote_rows", "full_assignments"]
HintKind = Literal["zero_hint", "copy_current_dispatch", "learned_prediction", "perfect_trace_hint", "expert_route"]


def _matrix(value: np.ndarray, *, name: str, world_size: int | None = None, zero_diagonal: bool = False) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if world_size is not None and a.shape != (int(world_size), int(world_size)):
        raise ValueError(f"{name} shape {a.shape} != ({world_size}, {world_size})")
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if (a < 0).any():
        raise ValueError(f"{name} must be non-negative")
    rounded = np.rint(a)
    if not np.allclose(a, rounded, atol=0.0, rtol=0.0):
        raise ValueError(f"{name} must contain integral row counts")
    if rounded.size and float(rounded.max()) > float(np.iinfo(np.int32).max):
        raise ValueError(f"{name} exceeds int32 row-count range")
    out = rounded.astype(np.int32, copy=True)
    if zero_diagonal:
        np.fill_diagonal(out, 0)
    return out


def _readonly_matrix(value: np.ndarray, *, name: str, world_size: int | None = None, zero_diagonal: bool = False) -> np.ndarray:
    out = _matrix(value, name=name, world_size=world_size, zero_diagonal=zero_diagonal)
    out.setflags(write=False)
    return out


def _freeze_metadata(value: Any) -> Any:
    """Recursively freeze metadata so cached semantic digests cannot become stale."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value).copy()
        arr.setflags(write=False)
        return arr
    if isinstance(value, list):
        return tuple(_freeze_metadata(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_metadata(v) for v in value), key=repr))
    return value


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def semantic_metadata(value: Any) -> Any:
    """Convert immutable metadata to JSON-safe values and remove observability fields."""
    if isinstance(value, Mapping):
        return {
            str(k): semantic_metadata(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if not (str(k).endswith("_ms") or str(k) in {"created_at", "wall_time", "hostname"})
        }
    if isinstance(value, (list, tuple)):
        return [semantic_metadata(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class HomogeneousTopology:
    """Homogeneous multi-node topology with a stable rank-to-node mapping."""

    world_size: int
    ranks_per_node: int
    rank_to_node: tuple[int, ...] | None = None

    def validate(self) -> None:
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if int(self.ranks_per_node) <= 0:
            raise ValueError("ranks_per_node must be > 0")
        mapping = self.mapping()
        if len(mapping) != int(self.world_size):
            raise ValueError("rank_to_node length must equal world_size")
        if min(mapping, default=0) < 0:
            raise ValueError("node ids must be non-negative")

    def mapping(self) -> tuple[int, ...]:
        if self.rank_to_node is not None:
            return tuple(int(x) for x in self.rank_to_node)
        return tuple(r // int(self.ranks_per_node) for r in range(int(self.world_size)))

    @property
    def node_count(self) -> int:
        mapping = self.mapping()
        return 0 if not mapping else max(mapping) + 1

    def to_dict(self) -> dict:
        self.validate()
        return {
            "world_size": int(self.world_size),
            "ranks_per_node": int(self.ranks_per_node),
            "rank_to_node": list(self.mapping()),
            "node_count": int(self.node_count),
            "topology_semantic_version": "homogeneous_nodes_v1",
        }

    @classmethod
    def contiguous(cls, world_size: int, ranks_per_node: int) -> "HomogeneousTopology":
        return cls(world_size=int(world_size), ranks_per_node=int(ranks_per_node))


@dataclass(frozen=True)
class AffineLinkCost:
    """Simple topology correction ``time = k * rows + b``."""

    intra_k: float = 1.0
    intra_b: float = 0.0
    inter_k: float = 1.0
    inter_b: float = 0.0
    wave_launch_b: float = 0.0

    def validate(self) -> None:
        for name, value in self.to_dict(include_version=False).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.intra_k <= 0 or self.inter_k <= 0:
            raise ValueError("link slopes must be > 0")

    def matrices(self, topology: HomogeneousTopology) -> tuple[np.ndarray, np.ndarray]:
        self.validate(); topology.validate()
        n = int(topology.world_size); nodes = topology.mapping()
        slope = np.empty((n, n), dtype=np.float64)
        intercept = np.empty((n, n), dtype=np.float64)
        for s in range(n):
            for d in range(n):
                same = nodes[s] == nodes[d]
                slope[s, d] = float(self.intra_k if same else self.inter_k)
                intercept[s, d] = float(self.intra_b if same else self.inter_b)
        slope.setflags(write=False); intercept.setflags(write=False)
        return slope, intercept

    def to_dict(self, *, include_version: bool = True) -> dict:
        out = {
            "intra_k": float(self.intra_k), "intra_b": float(self.intra_b),
            "inter_k": float(self.inter_k), "inter_b": float(self.inter_b),
            "wave_launch_b": float(self.wave_launch_b),
        }
        if include_version:
            out["cost_semantic_version"] = "affine_link_v1"
        return out


@dataclass(frozen=True)
class PlannerConstraints:
    expert_compute_delay: float = 0.0
    max_waves: int = 10000

    def validate(self) -> None:
        if not math.isfinite(float(self.expert_compute_delay)) or self.expert_compute_delay < 0:
            raise ValueError("expert_compute_delay must be finite and non-negative")
        if int(self.max_waves) <= 0:
            raise ValueError("max_waves must be > 0")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "expert_compute_delay": float(self.expert_compute_delay),
            "max_waves": int(self.max_waves),
            "release_semantic_version": "rank_barrier_plus_compute_v1",
        }


@dataclass(frozen=True)
class TrafficHint:
    predictor_id: str
    target_dispatch_rows: np.ndarray
    confidence: float
    hint_kind: HintKind = "learned_prediction"
    oracle: bool = False
    matrix_kind: MatrixKind = "remote_rows"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.predictor_id):
            raise ValueError("predictor_id must be non-empty")
        if self.hint_kind not in {"zero_hint", "copy_current_dispatch", "learned_prediction", "perfect_trace_hint", "expert_route"}:
            raise ValueError(f"unsupported hint_kind {self.hint_kind!r}")
        if self.matrix_kind not in {"remote_rows", "full_assignments"}:
            raise ValueError(f"unsupported matrix_kind {self.matrix_kind!r}")
        m = _readonly_matrix(self.target_dispatch_rows, name="target_dispatch_rows")
        if self.matrix_kind == "remote_rows" and np.diag(m).any():
            raise ValueError("remote_rows hints must have a zero diagonal")
        if not math.isfinite(float(self.confidence)) or not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.hint_kind == "zero_hint" and m.any():
            raise ValueError("zero_hint must be all zero")
        if self.hint_kind == "zero_hint" and float(self.confidence) != 0.0:
            raise ValueError("zero_hint confidence must equal 0")
        if self.hint_kind == "perfect_trace_hint" and (not self.oracle or float(self.confidence) != 1.0):
            raise ValueError("perfect trace requires oracle=True and confidence=1")
        if self.oracle and self.hint_kind != "perfect_trace_hint":
            raise ValueError("oracle=True is reserved for perfect_trace_hint")
        object.__setattr__(self, "target_dispatch_rows", m)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def validate(self, *, world_size: int | None = None) -> None:
        if world_size is not None and self.target_dispatch_rows.shape != (int(world_size), int(world_size)):
            raise ValueError(f"target_dispatch_rows shape {self.target_dispatch_rows.shape} != ({world_size}, {world_size})")

    def matrix(self, *, world_size: int | None = None) -> np.ndarray:
        self.validate(world_size=world_size)
        return self.target_dispatch_rows

    def semantic_payload(self) -> dict:
        return {
            "prediction_semantic_version": "traffic_hint_v4",
            "predictor_id": str(self.predictor_id),
            "hint_kind": str(self.hint_kind),
            "target_dispatch_rows": self.target_dispatch_rows.tolist(),
            "confidence": float(self.confidence),
            "oracle": bool(self.oracle),
            "matrix_kind": str(self.matrix_kind),
            "metadata": semantic_metadata(self.metadata),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload())
            object.__setattr__(self, "_digest_cache", cached)
        return cached


@dataclass(frozen=True)
class ForecastPlanningRequest:
    p0_dispatch_rows: np.ndarray
    p1_return_rows: np.ndarray
    prediction_hint: TrafficHint
    topology: HomogeneousTopology
    cost_model: AffineLinkCost = field(default_factory=AffineLinkCost)
    constraints: PlannerConstraints = field(default_factory=PlannerConstraints)
    request_id: str = "offline"
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)
    _cost_cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.topology.validate(); self.cost_model.validate(); self.constraints.validate()
        n = int(self.topology.world_size)
        p0 = _readonly_matrix(self.p0_dispatch_rows, name="p0_dispatch_rows", world_size=n, zero_diagonal=True)
        p1 = _readonly_matrix(self.p1_return_rows, name="p1_return_rows", world_size=n, zero_diagonal=True)
        self.prediction_hint.validate(world_size=n)
        if self.prediction_hint.matrix_kind != "remote_rows":
            raise ValueError("planning requests require remote_rows prediction hints")
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")
        object.__setattr__(self, "p0_dispatch_rows", p0)
        object.__setattr__(self, "p1_return_rows", p1)

    def validate(self) -> None:
        # Construction performs strict validation; keep this method for callers.
        self.topology.validate(); self.cost_model.validate(); self.constraints.validate()
        self.prediction_hint.validate(world_size=self.topology.world_size)

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.p0_dispatch_rows, self.p1_return_rows, self.prediction_hint.matrix(world_size=self.topology.world_size)

    def cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cost_cache
        if cached is None:
            cached = self.cost_model.matrices(self.topology)
            object.__setattr__(self, "_cost_cache", cached)
        return cached

    def semantic_payload(self) -> dict:
        return {
            "planning_request_semantic_version": "forecast_request_v3",
            "request_id": str(self.request_id),
            "p0_dispatch_rows": self.p0_dispatch_rows.tolist(),
            "p1_return_rows": self.p1_return_rows.tolist(),
            "prediction_hint_digest": self.prediction_hint.semantic_digest(),
            "topology": self.topology.to_dict(),
            "cost_model": self.cost_model.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload())
            object.__setattr__(self, "_digest_cache", cached)
        return cached


@dataclass(frozen=True)
class P2RevealRequest:
    forecast_request_digest: str
    p2_truth_rows: np.ndarray
    request_id: str = "offline"

    def __post_init__(self) -> None:
        if len(str(self.forecast_request_digest)) != 64:
            raise ValueError("forecast_request_digest must be a SHA-256 digest")
        m = _readonly_matrix(self.p2_truth_rows, name="p2_truth_rows", zero_diagonal=True)
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")
        object.__setattr__(self, "p2_truth_rows", m)

    def validate(self, *, world_size: int) -> None:
        if self.p2_truth_rows.shape != (int(world_size), int(world_size)):
            raise ValueError(f"p2_truth_rows shape {self.p2_truth_rows.shape} != ({world_size}, {world_size})")

    def matrix(self, *, world_size: int) -> np.ndarray:
        self.validate(world_size=world_size)
        return self.p2_truth_rows


@dataclass(frozen=True)
class P2RowReveal:
    forecast_artifact_digest: str
    source_rank: int
    row: np.ndarray
    request_id: str = "offline"

    def __post_init__(self) -> None:
        if len(str(self.forecast_artifact_digest)) != 64:
            raise ValueError("forecast_artifact_digest must be a SHA-256 digest")
        values = np.asarray(self.row)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("row must be a numeric vector")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("row must be finite and non-negative")
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=0.0, rtol=0.0):
            raise ValueError("row must contain integral row counts")
        if rounded.size and float(rounded.max()) > float(np.iinfo(np.int32).max):
            raise ValueError("row exceeds int32 row-count range")
        out = rounded.astype(np.int32, copy=True)
        if 0 <= int(self.source_rank) < len(out):
            if out[int(self.source_rank)] != 0:
                raise ValueError("remote P2 row must have zero local entry")
        out.setflags(write=False)
        object.__setattr__(self, "row", out)

    def validate(self, *, world_size: int) -> None:
        if not 0 <= int(self.source_rank) < int(world_size):
            raise ValueError("source_rank outside world")
        if self.row.shape != (int(world_size),):
            raise ValueError("row must have shape [world_size]")

    def vector(self, *, world_size: int) -> np.ndarray:
        self.validate(world_size=world_size)
        return self.row


class P2RevealAccumulator:
    """Collect immutable per-source P2 reveals before producing a bind request."""
    def __init__(self, *, forecast_artifact_digest: str, world_size: int, request_id: str = "offline") -> None:
        if len(str(forecast_artifact_digest)) != 64:
            raise ValueError("forecast_artifact_digest must be a SHA-256 digest")
        self.forecast_artifact_digest = str(forecast_artifact_digest)
        self.world_size = int(world_size)
        self.request_id = str(request_id)
        self._rows: dict[int, np.ndarray] = {}

    def add(self, reveal: P2RowReveal) -> None:
        reveal.validate(world_size=self.world_size)
        if reveal.forecast_artifact_digest != self.forecast_artifact_digest:
            raise ValueError("row reveal references another forecast artifact")
        if reveal.request_id != self.request_id:
            raise ValueError("row reveal request_id mismatch")
        source = int(reveal.source_rank)
        row = np.ascontiguousarray(reveal.vector(world_size=self.world_size)).copy()
        row.setflags(write=False)
        if source in self._rows and not np.array_equal(self._rows[source], row):
            raise ValueError("source row was already revealed with different content")
        self._rows[source] = row

    @property
    def complete(self) -> bool:
        return len(self._rows) == self.world_size

    def finalize(self, *, missing_as_zero: bool = False) -> P2RevealRequest:
        if not self.complete and not missing_as_zero:
            missing = sorted(set(range(self.world_size)) - set(self._rows))
            raise ValueError(f"missing P2 source rows: {missing}")
        matrix = np.zeros((self.world_size, self.world_size), dtype=np.int32)
        for source, row in self._rows.items():
            matrix[source] = row
        return P2RevealRequest(self.forecast_artifact_digest, matrix, self.request_id)


__all__ = [
    "AffineLinkCost", "ForecastPlanningRequest", "HomogeneousTopology", "P2RevealRequest", "P2RowReveal", "P2RevealAccumulator",
    "PlannerConstraints", "TrafficHint", "_digest", "_freeze_metadata", "_matrix", "semantic_metadata",
]
