"""Stateful Trace Provider collector.

The collector accepts already-realized post-policy routing facts.  It never
recreates clipping or padding heuristics from raw intent, and it never emits
scheduler or transport objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.invariants import fixture_invariants
from ..schema.model import (
    DatasetProvenance,
    DescriptorMetadataSpec,
    FixtureInitialState,
    FixtureInput,
    LocalComputeProfile,
    PayloadSpec,
    RankNodeExpertMapping,
    RealizedRouting,
    TraceValidationError,
    TraceWindow,
)


@dataclass
class TraceCollector:
    fixture_id: str
    provenance: DatasetProvenance
    initial_time_ns: int = 0
    _windows: list[TraceWindow] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.fixture_id).strip():
            raise TraceValidationError("fixture_id is required")
        if int(self.initial_time_ns) < 0:
            raise TraceValidationError("initial_time_ns must be nonnegative")

    def record_window(
        self,
        *,
        window_id: str,
        layer_id: int,
        request_id: str,
        decode_step: int,
        is_bootstrap_p0: bool,
        mapping: RankNodeExpertMapping,
        routing: RealizedRouting,
        local_compute: LocalComputeProfile,
        dispatch_payload_spec: PayloadSpec,
        combine_payload_spec: PayloadSpec,
        descriptor_metadata_spec: DescriptorMetadataSpec,
        metadata: dict[str, Any] | None = None,
    ) -> TraceWindow:
        if not self._windows and not bool(is_bootstrap_p0):
            raise TraceValidationError("first recorded window must be bootstrap P0")
        if self._windows and bool(is_bootstrap_p0):
            raise TraceValidationError("bootstrap P0 may only be first")
        window = TraceWindow(
            window_id=str(window_id),
            layer_id=int(layer_id),
            request_id=str(request_id),
            decode_step=int(decode_step),
            is_bootstrap_p0=bool(is_bootstrap_p0),
            mapping=mapping,
            routing=routing,
            local_compute=local_compute,
            dispatch_payload_spec=dispatch_payload_spec,
            combine_payload_spec=combine_payload_spec,
            descriptor_metadata_spec=descriptor_metadata_spec,
            metadata=dict(metadata or {}),
        )
        identity = (window.request_id, window.decode_step, window.layer_id)
        if any(
            (item.request_id, item.decode_step, item.layer_id) == identity
            for item in self._windows
        ):
            raise TraceValidationError(f"duplicate trace window identity {identity!r}")
        if any(item.window_id == window.window_id for item in self._windows):
            raise TraceValidationError(f"duplicate trace window_id {window.window_id!r}")
        if self._windows and window.mapping.world_size != self._windows[0].mapping.world_size:
            raise TraceValidationError("all recorded windows must share world_size")
        self._windows.append(window)
        return window

    def freeze(self) -> FixtureInput:
        if not self._windows:
            raise TraceValidationError("cannot freeze an empty collector")
        windows = tuple(self._windows)
        first = windows[0]
        initial = FixtureInitialState(
            initial_time_ns=int(self.initial_time_ns),
            bootstrap_window_id=first.window_id,
            bootstrap_source_ranks=tuple(range(first.mapping.world_size)),
            predecessor_p1_exists=False,
        )
        return FixtureInput(
            fixture_id=str(self.fixture_id),
            provenance=self.provenance,
            initial_state=initial,
            windows=windows,
            expected_invariants=fixture_invariants(windows),
        )
