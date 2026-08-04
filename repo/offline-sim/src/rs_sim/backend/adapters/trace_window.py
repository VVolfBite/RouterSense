"""TraceWindow-to-backend batch registration without owning Trace schemas.

The builder consumes Trace Provider objects by attribute only and delegates all
shared-object construction to the frozen trace bridge and SimulationBackend.
It never creates canonical tasks or advances the simulation clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_sim.backend.core.engine import SimulationBackend
from rs_sim.backend.core.errors import BackendContractError
from rs_sim.backend.core.util import require_time_ns


@dataclass(frozen=True, slots=True)
class RegisteredTraceWindow:
    """Backend-owned registration receipt, not a replacement Trace schema."""

    trace_window: Any
    keys: Any
    combine_expectations: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class TraceFixtureRegistration:
    run_id: str
    initial_time_ns: int
    windows: tuple[RegisteredTraceWindow, ...]
    bootstrap_ready_at_ns_by_rank: tuple[int, ...]

    @property
    def terminal_combine_phase_key(self) -> Any:
        return self.windows[-1].keys.combine_phase_key

    @property
    def all_phase_keys(self) -> tuple[Any, ...]:
        result: list[Any] = []
        for window in self.windows:
            result.extend((window.keys.dispatch_phase_key, window.keys.combine_phase_key))
        return tuple(result)


class BackendTraceFixtureBuilder:
    """Register a complete multi-layer Trace fixture into one backend backend."""

    def __init__(self, *, backend: SimulationBackend) -> None:
        self.backend = backend

    def register_fixture(
        self,
        *,
        fixture_input: Any,
        run_id: str,
        bootstrap_start_ns: int | None = None,
        activate_bootstrap: bool = True,
    ) -> TraceFixtureRegistration:
        if not str(run_id):
            raise BackendContractError("run_id must be non-empty")
        from rs_sim.runtime.adapters.trace import (
            dispatch_row_truth_for_trace_window,
            keys_for_trace_window,
        )

        windows = tuple(getattr(fixture_input, "windows"))
        if not windows:
            raise BackendContractError("Trace fixture must contain at least one window")
        if int(getattr(fixture_input, "world_size")) != self.backend.world_size:
            raise BackendContractError("Trace fixture/backend world_size mismatch")
        initial_state = getattr(fixture_input, "initial_state")
        initial_time_ns = require_time_ns(
            getattr(initial_state, "initial_time_ns")
            if bootstrap_start_ns is None
            else bootstrap_start_ns,
            field="bootstrap_start_ns",
        )
        if not bool(getattr(windows[0], "is_bootstrap_p0")):
            raise BackendContractError("first TraceWindow must be bootstrap P0")
        if any(bool(getattr(window, "is_bootstrap_p0")) for window in windows[1:]):
            raise BackendContractError("only the first TraceWindow may be bootstrap P0")

        registered: list[RegisteredTraceWindow] = []
        keys = tuple(
            keys_for_trace_window(run_id=str(run_id), trace_window=window)
            for window in windows
        )
        for index, (window, window_keys) in enumerate(zip(windows, keys, strict=True)):
            local_compute = getattr(window, "local_compute")
            combine_expectations: list[Any] = []
            for rank in range(self.backend.world_size):
                truth = dispatch_row_truth_for_trace_window(
                    trace_window=window,
                    phase_key=window_keys.dispatch_phase_key,
                    src_rank=rank,
                )
                self.backend.register_exact_dispatch_row_truth(**truth)
                self.backend.register_dispatch_compute_spec(
                    dispatch_phase_key=window_keys.dispatch_phase_key,
                    next_combine_phase_key=window_keys.combine_phase_key,
                    rank_id=rank,
                    dispatch_local_postprocess_ns=int(
                        local_compute.dispatch_local_postprocess_ns[rank]
                    ),
                    dispatch_release_to_combine_source_ready_ns=int(
                        local_compute.dispatch_release_to_combine_source_ready_ns[rank]
                    ),
                )
                combine_expectations.extend(
                    self.backend.register_combine_expectations_from_dispatch_truth(
                        dispatch_phase_key=window_keys.dispatch_phase_key,
                        combine_phase_key=window_keys.combine_phase_key,
                        original_rank=rank,
                        created_at_ns=initial_time_ns,
                    )
                )
                if index + 1 < len(windows):
                    self.backend.register_local_path_spec(
                        combine_phase_key=window_keys.combine_phase_key,
                        next_dispatch_phase_key=keys[index + 1].dispatch_phase_key,
                        rank_id=rank,
                        combine_release_to_router_ready_ns=int(
                            local_compute.combine_release_to_router_ready_ns[rank]
                        ),
                        router_and_pack_ns=int(local_compute.router_and_pack_ns[rank]),
                    )
                else:
                    self.backend.register_terminal_local_path_spec(
                        combine_phase_key=window_keys.combine_phase_key,
                        rank_id=rank,
                        combine_release_to_router_ready_ns=int(
                            local_compute.combine_release_to_router_ready_ns[rank]
                        ),
                        terminal_local_compute_ns=int(
                            local_compute.router_and_pack_ns[rank]
                        ),
                    )
            registered.append(
                RegisteredTraceWindow(
                    trace_window=window,
                    keys=window_keys,
                    combine_expectations=tuple(combine_expectations),
                )
            )

        bootstrap_ranks = tuple(int(v) for v in initial_state.bootstrap_source_ranks)
        expected_ranks = tuple(range(self.backend.world_size))
        if tuple(sorted(bootstrap_ranks)) != expected_ranks:
            raise BackendContractError(
                "complete multi-rank bootstrap requires every logical rank exactly once"
            )
        bootstrap_profile = windows[0].local_compute
        ready_times = tuple(
            initial_time_ns + int(bootstrap_profile.bootstrap_router_and_pack_ns[rank])
            for rank in range(self.backend.world_size)
        )
        for rank in bootstrap_ranks:
            self.backend.register_bootstrap_source_local_path_start(
                phase_key=keys[0].dispatch_phase_key,
                rank_id=rank,
                at_ns=initial_time_ns,
            )
        if activate_bootstrap:
            for rank in bootstrap_ranks:
                self.backend.on_bootstrap_dispatch_local_path_complete(
                    phase_key=keys[0].dispatch_phase_key,
                    rank_id=rank,
                    at_ns=ready_times[rank],
                )
        return TraceFixtureRegistration(
            run_id=str(run_id),
            initial_time_ns=initial_time_ns,
            windows=tuple(registered),
            bootstrap_ready_at_ns_by_rank=ready_times,
        )


__all__ = [
    "BackendTraceFixtureBuilder",
    "RegisteredTraceWindow",
    "TraceFixtureRegistration",
]
