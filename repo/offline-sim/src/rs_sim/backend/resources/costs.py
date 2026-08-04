"""Simple deterministic receiver cost models for fixtures and bootstrap runs."""

from __future__ import annotations

from dataclasses import dataclass

from rs_sim.backend.core.errors import BackendContractError


@dataclass(frozen=True, slots=True)
class LinearReceiverCostModel:
    """Integer fixed-plus-byte cost model.

    Production calibration can replace this through ``CostModel``.  Integer
    ceiling division avoids floating-point authority in simulation time.
    """

    posting_fixed_ns: int = 0
    posting_bytes_per_ns: int = 1
    drain_fixed_ns: int = 0
    drain_bytes_per_ns: int = 1

    def __post_init__(self) -> None:
        for name in (
            "posting_fixed_ns",
            "posting_bytes_per_ns",
            "drain_fixed_ns",
            "drain_bytes_per_ns",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BackendContractError(f"{name} must be a non-negative int")
        if self.posting_bytes_per_ns == 0 or self.drain_bytes_per_ns == 0:
            raise BackendContractError("bytes_per_ns must be positive")

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    def receiver_service_cost_ns(self, task_bytes: int) -> int:
        if task_bytes <= 0:
            raise BackendContractError("receiver task bytes must be positive")
        return self.posting_fixed_ns + self._ceil_div(
            task_bytes, self.posting_bytes_per_ns
        )

    def receiver_drain_cost_ns(self, task_bytes: int) -> int:
        if task_bytes <= 0:
            raise BackendContractError("receiver task bytes must be positive")
        return self.drain_fixed_ns + self._ceil_div(
            task_bytes, self.drain_bytes_per_ns
        )
