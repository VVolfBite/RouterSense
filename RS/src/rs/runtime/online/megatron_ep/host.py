"""Host/runtime bootstrap surface for the formal Megatron EP path."""

from __future__ import annotations

from ._host_impl import (
    StageStatus,
    attach_dispatch_facade,
    attach_dispatch_observer,
    build_position_ids,
    dataclass_to_dict,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    get_process_group_ranks_safe,
    get_process_group_root_safe,
    init_distributed,
    load_prompts,
    model_is_local_path,
    stage_barrier,
    summarize_native_dispatchers,
    summarize_observer_rows,
    summarize_rank_environment,
    validate_observer_mode,
)

__all__ = [
    "StageStatus",
    "attach_dispatch_facade",
    "attach_dispatch_observer",
    "build_position_ids",
    "dataclass_to_dict",
    "destroy_distributed",
    "dtype_from_name",
    "gather_rank_payloads",
    "get_process_group_ranks_safe",
    "get_process_group_root_safe",
    "init_distributed",
    "load_prompts",
    "model_is_local_path",
    "stage_barrier",
    "summarize_native_dispatchers",
    "summarize_observer_rows",
    "summarize_rank_environment",
    "validate_observer_mode",
]
