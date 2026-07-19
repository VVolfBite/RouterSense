from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WireCostContract:
    phase: str
    tensor_role: str
    rows: int
    hidden_bytes_per_row: int
    routing_prob_bytes_per_row: int
    phase_wire_bytes_per_row: int
    estimated_wire_bytes: int
    matrix_unit: str = "rows"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype_size(dtype: str) -> int:
    name = str(dtype).lower().strip()
    if name in {"fp16", "float16", "half", "bf16", "bfloat16"}:
        return 2
    if name in {"fp32", "float32", "float"}:
        return 4
    if name in {"fp64", "float64", "double"}:
        return 8
    if name in {"int8", "uint8", "byte", "bool"}:
        return 1
    if name in {"int16", "short"}:
        return 2
    if name in {"int32", "int"}:
        return 4
    if name in {"int64", "long"}:
        return 8
    raise ValueError(f"unsupported dtype size lookup {dtype!r}")


def hidden_bytes_per_row(*, hidden_size: int, hidden_dtype: str) -> int:
    return int(hidden_size) * _dtype_size(hidden_dtype)


def routing_prob_bytes_per_row(*, top_k: int, routing_probability_dtype: str) -> int:
    return int(top_k) * _dtype_size(routing_probability_dtype)


def phase_wire_bytes_per_row(
    *,
    phase: str,
    tensor_role: str,
    hidden_size: int,
    hidden_dtype: str,
    routing_probability_dtype: str,
    top_k: int = 1,
) -> int:
    phase_name = str(phase).upper()
    role = str(tensor_role)
    hidden = hidden_bytes_per_row(hidden_size=int(hidden_size), hidden_dtype=str(hidden_dtype))
    routing = routing_prob_bytes_per_row(top_k=int(top_k), routing_probability_dtype=str(routing_probability_dtype))
    if phase_name == "P0":
        if role == "hidden_states":
            return int(hidden)
        if role == "routing_probs":
            return int(routing)
        if role == "dispatch_bundle":
            return int(hidden + routing)
    if phase_name == "P1":
        if role in {"hidden_states", "combine_bundle"}:
            return int(hidden)
    raise ValueError(f"unsupported phase/tensor_role wire cost pair: {phase!r} {tensor_role!r}")


def phase_wire_bytes(
    *,
    phase: str,
    tensor_role: str,
    rows: int,
    hidden_size: int,
    hidden_dtype: str,
    routing_probability_dtype: str,
    top_k: int = 1,
) -> int:
    return int(rows) * phase_wire_bytes_per_row(
        phase=phase,
        tensor_role=tensor_role,
        hidden_size=hidden_size,
        hidden_dtype=hidden_dtype,
        routing_probability_dtype=routing_probability_dtype,
        top_k=top_k,
    )


def describe_wire_cost(
    *,
    phase: str,
    tensor_role: str,
    rows: int,
    hidden_size: int,
    hidden_dtype: str,
    routing_probability_dtype: str,
    top_k: int = 1,
) -> WireCostContract:
    hidden = hidden_bytes_per_row(hidden_size=int(hidden_size), hidden_dtype=str(hidden_dtype))
    routing = routing_prob_bytes_per_row(top_k=int(top_k), routing_probability_dtype=str(routing_probability_dtype))
    per_row = phase_wire_bytes_per_row(
        phase=phase,
        tensor_role=tensor_role,
        hidden_size=hidden_size,
        hidden_dtype=hidden_dtype,
        routing_probability_dtype=routing_probability_dtype,
        top_k=top_k,
    )
    return WireCostContract(
        phase=str(phase),
        tensor_role=str(tensor_role),
        rows=int(rows),
        hidden_bytes_per_row=int(hidden),
        routing_prob_bytes_per_row=int(routing),
        phase_wire_bytes_per_row=int(per_row),
        estimated_wire_bytes=int(rows) * int(per_row),
    )


__all__ = [
    "WireCostContract",
    "describe_wire_cost",
    "hidden_bytes_per_row",
    "phase_wire_bytes",
    "phase_wire_bytes_per_row",
    "routing_prob_bytes_per_row",
]
