from __future__ import annotations

"""Validated topology-aware planner cost profiles.

A deployment calibration emits pairwise affine transfer costs in planner row
units.  The runtime consumes the profile as immutable planner configuration;
no scheduler is allowed to silently reinterpret or partially apply it.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LINK_COST_PROFILE_SCHEMA = "routersense.link_cost_profile.v1"


def precision_bytes(precision: str) -> int:
    normalized = str(precision).strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "fp8": 1,
        "float8": 1,
        "fp16": 2,
        "float16": 2,
        "half": 2,
        "bf16": 2,
        "bfloat16": 2,
        "fp32": 4,
        "float32": 4,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported deployment precision {precision!r}")
    return int(mapping[normalized])


def infer_model_row_contract(
    model_path: str | Path,
    *,
    precision: str | None = None,
    element_bytes: int | None = None,
) -> dict[str, object]:
    """Resolve the token-row byte contract from a local model config."""
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json"
    raw = config_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    hidden_size = int(payload.get("hidden_size") or payload.get("d_model") or 0)
    if hidden_size <= 0:
        raise ValueError(f"cannot infer hidden size from {config_path}")
    resolved_element_bytes = int(element_bytes or precision_bytes(str(precision or "")))
    if resolved_element_bytes <= 0:
        raise ValueError("element bytes must be positive")
    return {
        "model_path": str(path),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "hidden_size": hidden_size,
        "element_bytes": resolved_element_bytes,
        "row_bytes": hidden_size * resolved_element_bytes,
    }


def _square_float_matrix(value: object, *, name: str, world_size: int, positive: bool) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != int(world_size):
        raise ValueError(f"{name} must contain {world_size} rows")
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != int(world_size):
            raise ValueError(f"{name} must be {world_size}x{world_size}")
        converted = tuple(float(item) for item in row)
        for item in converted:
            if item < 0.0 or (positive and item <= 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} entries must be {qualifier}")
        rows.append(converted)
    return tuple(rows)




def fit_affine_row_cost(
    samples: Sequence[tuple[int, float]],
    *,
    row_bytes: int,
) -> tuple[float, float]:
    """Fit ``duration_us = slope_us_per_row * rows + intercept_us``.

    Negative estimates caused by timer noise are clamped to a valid planner
    model instead of allowing a calibration artifact to create negative costs.
    """
    if int(row_bytes) <= 0:
        raise ValueError("row_bytes must be positive")
    if len(samples) < 2:
        raise ValueError("at least two transfer sizes are required")
    xs = [float(size_bytes) / float(row_bytes) for size_bytes, _ in samples]
    ys = [float(duration_us) for _, duration_us in samples]
    if any(size <= 0 for size, _ in samples) or any(duration < 0.0 for _, duration in samples):
        raise ValueError("calibration samples must use positive sizes and non-negative durations")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denominator <= 0.0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    slope = max(float(slope), 1e-9)
    intercept = max(0.0, mean_y - slope * mean_x)
    return slope, intercept


@dataclass(frozen=True)
class LinkCostProfile:
    schema_version: str
    profile_id: str
    world_size: int
    ranks_per_node: int
    rank_to_node: tuple[int, ...]
    row_bytes: int
    edge_slope_us_per_row: tuple[tuple[float, ...], ...]
    edge_intercept_us: tuple[tuple[float, ...], ...]
    wave_launch_us: float = 0.0
    source: str = "measured"
    metadata: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if str(self.schema_version) != LINK_COST_PROFILE_SCHEMA:
            raise ValueError(f"unsupported link cost profile schema {self.schema_version!r}")
        if not str(self.profile_id):
            raise ValueError("profile_id must be non-empty")
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be positive")
        if int(self.ranks_per_node) <= 0:
            raise ValueError("ranks_per_node must be positive")
        if len(self.rank_to_node) != int(self.world_size):
            raise ValueError("rank_to_node length must equal world_size")
        if min(self.rank_to_node, default=0) < 0:
            raise ValueError("rank_to_node entries must be non-negative")
        if int(self.row_bytes) <= 0:
            raise ValueError("row_bytes must be positive")
        if float(self.wave_launch_us) < 0.0:
            raise ValueError("wave_launch_us must be non-negative")
        _square_float_matrix(
            self.edge_slope_us_per_row,
            name="edge_slope_us_per_row",
            world_size=int(self.world_size),
            positive=True,
        )
        _square_float_matrix(
            self.edge_intercept_us,
            name="edge_intercept_us",
            world_size=int(self.world_size),
            positive=False,
        )

    def planner_config(self) -> dict[str, object]:
        self.validate()
        return {
            "ranks_per_node": int(self.ranks_per_node),
            "rank_to_node": tuple(int(item) for item in self.rank_to_node),
            "edge_slope": tuple(tuple(float(item) for item in row) for row in self.edge_slope_us_per_row),
            "edge_intercept": tuple(tuple(float(item) for item in row) for row in self.edge_intercept_us),
            "wave_launch_b": float(self.wave_launch_us),
            "cost_profile_id": str(self.profile_id),
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rank_to_node"] = list(self.rank_to_node)
        payload["edge_slope_us_per_row"] = [list(row) for row in self.edge_slope_us_per_row]
        payload["edge_intercept_us"] = [list(row) for row in self.edge_intercept_us]
        payload["metadata"] = dict(self.metadata or {})
        return payload


def profile_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("profile_id", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def link_cost_profile_from_dict(payload: Mapping[str, Any]) -> LinkCostProfile:
    world_size = int(payload.get("world_size", 0))
    profile = LinkCostProfile(
        schema_version=str(payload.get("schema_version", "")),
        profile_id=str(payload.get("profile_id", "")),
        world_size=world_size,
        ranks_per_node=int(payload.get("ranks_per_node", 0)),
        rank_to_node=tuple(int(item) for item in payload.get("rank_to_node", ())),
        row_bytes=int(payload.get("row_bytes", 0)),
        edge_slope_us_per_row=_square_float_matrix(
            payload.get("edge_slope_us_per_row"),
            name="edge_slope_us_per_row",
            world_size=world_size,
            positive=True,
        ),
        edge_intercept_us=_square_float_matrix(
            payload.get("edge_intercept_us"),
            name="edge_intercept_us",
            world_size=world_size,
            positive=False,
        ),
        wave_launch_us=float(payload.get("wave_launch_us", 0.0)),
        source=str(payload.get("source", "measured")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )
    profile.validate()
    expected = profile_digest(profile.to_dict())
    if str(profile.profile_id) not in {expected, f"sha256:{expected}"}:
        raise ValueError("link cost profile_id does not match profile contents")
    return profile


def load_link_cost_profile(path: str | Path) -> LinkCostProfile:
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("link cost profile must contain a JSON object")
    return link_cost_profile_from_dict(payload)



def resolve_runtime_link_cost_profile(
    *,
    configured_path: str | Path | None,
    source_config_path: str | Path,
    repository_root: str | Path,
    model_path: str | Path,
    precision: str,
    world_size: int,
    local_world_size: int,
    require_profile: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve one immutable planner cost configuration for online runtime.

    Multi-node execution fails closed without a measured profile.  A supplied
    profile is bound to the actual torchrun layout and model token-row byte
    contract before any planner is created.
    """
    actual_world_size = int(world_size)
    ranks_per_node = int(local_world_size)
    if actual_world_size <= 0 or ranks_per_node <= 0 or actual_world_size % ranks_per_node != 0:
        raise RuntimeError(
            f"invalid runtime layout WORLD_SIZE={actual_world_size} "
            f"LOCAL_WORLD_SIZE={ranks_per_node}"
        )
    configured = str(configured_path or "").strip()
    required = bool(require_profile or actual_world_size > ranks_per_node)
    if not configured:
        if required:
            raise RuntimeError(
                "topology-aware link cost profile is required for multi-node execution; "
                "run scripts/deploy/calibrate_cluster_links.py before launch"
            )
        rank_to_node = tuple(int(rank // ranks_per_node) for rank in range(actual_world_size))
        planner_config: dict[str, object] = {
            "ranks_per_node": ranks_per_node,
            "rank_to_node": rank_to_node,
            "cost_profile_id": "homogeneous-default",
        }
        return planner_config, {
            "mode": "homogeneous_default",
            "profile_id": "homogeneous-default",
            "world_size": actual_world_size,
            "ranks_per_node": ranks_per_node,
        }

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        source_parent = Path(source_config_path).expanduser().resolve().parent
        repo_candidate = Path(repository_root).expanduser().resolve() / candidate
        probes = (source_parent / candidate, repo_candidate)
        candidate = next((item for item in probes if item.exists()), repo_candidate)
    profile = load_link_cost_profile(candidate)
    if int(profile.world_size) != actual_world_size:
        raise RuntimeError(
            f"link cost profile world_size={profile.world_size} does not match "
            f"runtime WORLD_SIZE={actual_world_size}"
        )
    if int(profile.ranks_per_node) != ranks_per_node:
        raise RuntimeError(
            f"link cost profile ranks_per_node={profile.ranks_per_node} does not match "
            f"runtime LOCAL_WORLD_SIZE={ranks_per_node}"
        )
    expected_rank_to_node = tuple(
        int(rank // ranks_per_node) for rank in range(actual_world_size)
    )
    if tuple(int(item) for item in profile.rank_to_node) != expected_rank_to_node:
        raise RuntimeError(
            "link cost profile rank_to_node does not match contiguous torchrun rank layout"
        )
    model_contract = infer_model_row_contract(model_path, precision=str(precision))
    if int(profile.row_bytes) != int(model_contract["row_bytes"]):
        raise RuntimeError(
            f"link cost profile row_bytes={profile.row_bytes} does not match "
            f"model/runtime row_bytes={model_contract['row_bytes']}"
        )
    profile_model_contract = dict((profile.metadata or {}).get("model_contract", {}) or {})
    expected_config_sha256 = str(model_contract["config_sha256"])
    calibrated_config_sha256 = str(profile_model_contract.get("config_sha256", "") or "")
    if calibrated_config_sha256 and calibrated_config_sha256 != expected_config_sha256:
        raise RuntimeError("link cost profile model config digest does not match runtime model")
    metadata: dict[str, object] = {
        "mode": "measured_pairwise",
        "path": str(candidate.resolve()),
        "profile_id": str(profile.profile_id),
        "world_size": int(profile.world_size),
        "ranks_per_node": int(profile.ranks_per_node),
        "row_bytes": int(profile.row_bytes),
        "source": str(profile.source),
        "model_config_sha256": expected_config_sha256,
    }
    return profile.planner_config(), metadata

def write_link_cost_profile(path: str | Path, payload: Mapping[str, Any]) -> LinkCostProfile:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = dict(payload)
    values["schema_version"] = LINK_COST_PROFILE_SCHEMA
    values.setdefault("wave_launch_us", 0.0)
    values.setdefault("source", "measured")
    values.setdefault("metadata", {})
    values["profile_id"] = profile_digest(values)
    profile = link_cost_profile_from_dict(values)
    output.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return profile


__all__ = [
    "LINK_COST_PROFILE_SCHEMA",
    "LinkCostProfile",
    "infer_model_row_contract",
    "fit_affine_row_cost",
    "link_cost_profile_from_dict",
    "load_link_cost_profile",
    "precision_bytes",
    "profile_digest",
    "resolve_runtime_link_cost_profile",
    "write_link_cost_profile",
]
