"""Megatron EP 在线运行时的外部入口与 bootstrap。

这个文件主要负责：
- 初始化/销毁分布式环境
- 安装原生 observer
- 安装正式 runtime facade
- 暴露 attach_formal_online_runtime 等外部入口
这里不承载调度算法本身，更多是“把运行时各模块装起来”。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.config import resolve_online_policy_config
from rs.runtime.online.megatron_ep.contracts import OnlineRuntimeConfig, RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep.observation import RouterSenseObserver
from rs.runtime.online.megatron_ep.public_types import (
    CombineFailedEvent,
    CombineCompleteEvent,
    CombineReadyEvent,
    ControlGroupHandle,
    DispatchFailedEvent,
    DispatchCompleteEvent,
    DispatchReadyEvent,
    ForwardBeginEvent,
    ForwardEndEvent,
    ForwardFailedEvent,
    LegacyObserverConflictError,
    RuntimeAlreadyAttachedError,
    RuntimeHandle,
)
from rs.runtime.online.megatron_ep.runtime import RouterSenseDispatcherFacade


@dataclass
class StageStatus:
    stage: str
    rank: int
    ok: bool
    detail: str = ""


@dataclass
class DedicatedP2PGroupRegistry:
    ordered_group_ranks: tuple[tuple[int, ...], ...]
    groups: dict[tuple[int, ...], dist.ProcessGroup]
    local_group_ranks: tuple[int, ...]
    local_group: dist.ProcessGroup | None
    warmup_passed: bool
    new_group_call_order: tuple[tuple[int, ...], ...]


_DEDICATED_P2P_GROUP_REGISTRY: dict[tuple[tuple[int, ...], ...], DedicatedP2PGroupRegistry] = {}
_CONTROL_GROUP_REGISTRY: dict[tuple[tuple[int, ...], ...], "ControlGroupRegistry"] = {}


# Basic runtime/bootstrap helpers


def get_process_group_ranks_safe(group: dist.ProcessGroup | None) -> tuple[int, ...]:
    if group is None:
        return tuple(range(dist.get_world_size())) if dist.is_initialized() else (0,)
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    return tuple(range(dist.get_world_size(group)))


def get_process_group_root_safe(group: dist.ProcessGroup | None) -> int:
    ranks = get_process_group_ranks_safe(group)
    return int(ranks[0]) if ranks else 0


def _discover_all_ep_group_tuples(local_group_ranks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    normalized_local = tuple(int(rank) for rank in local_group_ranks)
    if not dist.is_available() or not dist.is_initialized():
        return (normalized_local,)
    gathered: list[tuple[int, ...] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, normalized_local)
    unique = sorted({tuple(int(rank) for rank in (item or ())) for item in gathered if item is not None})
    return tuple(group for group in unique if group)


@dataclass
class ControlGroupRegistry:
    ordered_group_ranks: tuple[tuple[int, ...], ...]
    groups: dict[tuple[int, ...], dist.ProcessGroup]
    ref_counts: dict[tuple[int, ...], int]
    new_group_call_order: tuple[tuple[int, ...], ...]

    @classmethod
    def initialize(
        cls,
        *,
        local_ep_group_ranks: tuple[int, ...],
        world_group: dist.ProcessGroup | None = None,
    ) -> tuple["ControlGroupRegistry", ControlGroupHandle]:
        normalized_local = tuple(int(rank) for rank in local_ep_group_ranks)
        ordered = _discover_all_ep_group_tuples(normalized_local)
        cached = _CONTROL_GROUP_REGISTRY.get(ordered)
        if cached is None:
            groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
            call_order: list[tuple[int, ...]] = []
            for group_ranks in ordered:
                group = dist.new_group(ranks=list(group_ranks), backend="gloo")
                groups[group_ranks] = group
                call_order.append(tuple(int(rank) for rank in group_ranks))
            cached = cls(
                ordered_group_ranks=ordered,
                groups=groups,
                ref_counts={tuple(int(rank) for rank in item): 0 for item in ordered},
                new_group_call_order=tuple(call_order),
            )
            _CONTROL_GROUP_REGISTRY[ordered] = cached
        cached.ref_counts[normalized_local] = int(cached.ref_counts.get(normalized_local, 0)) + 1
        root_global_rank = int(normalized_local[0]) if normalized_local else 0
        root_group_rank = 0
        return cached, ControlGroupHandle(
            process_group=cached.groups.get(normalized_local),
            group_ranks=normalized_local,
            root_global_rank=root_global_rank,
            root_group_rank=root_group_rank,
            owned=True,
        )

    def close(self, *, local_ep_group_ranks: tuple[int, ...]) -> None:
        normalized_local = tuple(int(rank) for rank in local_ep_group_ranks)
        if normalized_local not in self.ref_counts:
            return
        self.ref_counts[normalized_local] = max(0, int(self.ref_counts[normalized_local]) - 1)
        if any(int(value) > 0 for value in self.ref_counts.values()):
            return
        for group in self.groups.values():
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group(group)
            except Exception:
                pass
        _CONTROL_GROUP_REGISTRY.pop(self.ordered_group_ranks, None)


def _get_or_create_dedicated_p2p_group_registry(
    *,
    ep_group_ranks: tuple[int, ...],
    local_rank: int,
) -> DedicatedP2PGroupRegistry | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    ordered = _discover_all_ep_group_tuples(tuple(int(rank) for rank in ep_group_ranks))
    cached = _DEDICATED_P2P_GROUP_REGISTRY.get(ordered)
    if cached is not None:
        return cached
    if torch.cuda.is_available() and int(local_rank) < int(torch.cuda.device_count()):
        torch.cuda.set_device(int(local_rank))
    groups: dict[tuple[int, ...], dist.ProcessGroup] = {}
    call_order: list[tuple[int, ...]] = []
    warmup_passed = False
    local_group: dist.ProcessGroup | None = None
    for group_ranks in ordered:
        group = dist.new_group(ranks=list(group_ranks))
        groups[group_ranks] = group
        call_order.append(group_ranks)
        if tuple(group_ranks) == tuple(int(rank) for rank in ep_group_ranks):
            local_group = group
    try:
        for group_ranks, group in groups.items():
            if int(dist.get_rank()) not in set(group_ranks):
                continue
            backend = str(dist.get_backend(group))
            warmup_device = "cuda" if backend == "nccl" and torch.cuda.is_available() else "cpu"
            tensor = torch.zeros(1, dtype=torch.int64, device=warmup_device)
            dist.all_reduce(tensor, group=group)
        warmup_passed = True
    except Exception:
        warmup_passed = False
    registry = DedicatedP2PGroupRegistry(
        ordered_group_ranks=ordered,
        groups=groups,
        local_group_ranks=tuple(int(rank) for rank in ep_group_ranks),
        local_group=local_group,
        warmup_passed=bool(warmup_passed),
        new_group_call_order=tuple(call_order),
    )
    _DEDICATED_P2P_GROUP_REGISTRY[ordered] = registry
    return registry


def _maybe_create_dedicated_p2p_group(
    *,
    ep_group_ranks: tuple[int, ...],
    local_rank: int,
) -> tuple[dist.ProcessGroup | None, dict[str, Any]]:
    registry = _get_or_create_dedicated_p2p_group_registry(
        ep_group_ranks=ep_group_ranks,
        local_rank=local_rank,
    )
    if registry is None:
        return None, {
            "dedicated_p2p_group_initialized": False,
            "p2p_group_ranks": list(ep_group_ranks),
            "p2p_group_warmup_passed": False,
            "hotpath_new_group_count": 0,
            "dedicated_p2p_groups_created": [],
            "local_dedicated_group_ranks": list(ep_group_ranks),
        }
    return registry.local_group, {
        "dedicated_p2p_group_initialized": True,
        "p2p_group_ranks": list(ep_group_ranks),
        "p2p_group_warmup_passed": bool(registry.warmup_passed),
        "hotpath_new_group_count": 0,
        "dedicated_p2p_groups_created": [list(item) for item in registry.ordered_group_ranks],
        "local_dedicated_group_ranks": list(registry.local_group_ranks),
        "new_group_call_order": [list(item) for item in registry.new_group_call_order],
    }


def _create_control_group_handle(
    *,
    ep_process_group: dist.ProcessGroup | None,
    ep_group_ranks: tuple[int, ...],
    root_global_rank: int,
) -> ControlGroupHandle:
    if not dist.is_available() or not dist.is_initialized():
        return ControlGroupHandle(
            process_group=None,
            group_ranks=tuple(int(rank) for rank in ep_group_ranks),
            root_global_rank=int(root_global_rank),
            root_group_rank=0,
            owned=False,
        )
    registry, handle = ControlGroupRegistry.initialize(
        local_ep_group_ranks=tuple(int(rank) for rank in ep_group_ranks),
        world_group=None,
    )
    handle.root_global_rank = int(root_global_rank)
    handle.root_group_rank = int(handle.group_ranks.index(int(root_global_rank))) if int(root_global_rank) in handle.group_ranks else 0
    setattr(handle, "_registry_key", registry.ordered_group_ranks)
    return handle


def model_is_local_path(model: str) -> bool:
    return Path(model).expanduser().exists()


def load_prompts(prompt_file: str | Path) -> list[str]:
    payload = json.loads(Path(prompt_file).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        prompts = [str(item) for item in payload]
    elif isinstance(payload, dict):
        prompts = [str(item) for item in payload.get("prompts", [])]
    else:
        raise ValueError(f"Unsupported prompts payload in {prompt_file}: expected list or mapping")
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def init_distributed(backend: str = "nccl", timeout_seconds: int = 300) -> dict[str, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def stage_barrier(stage: str, *, ok: bool, detail: str = "") -> None:
    if not dist.is_initialized():
        if not ok:
            raise RuntimeError(f"{stage}: {detail}")
        return
    payload = StageStatus(stage=stage, rank=int(dist.get_rank()), ok=ok, detail=detail)
    gathered: list[StageStatus | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    failures = [item for item in gathered if item is not None and not item.ok]
    if failures:
        detail_text = "; ".join(f"rank={item.rank} stage={item.stage} detail={item.detail}" for item in failures)
        raise RuntimeError(f"distributed preflight failed: {detail_text}")


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision {name!r}")


def build_position_ids(tokens: torch.Tensor) -> torch.Tensor:
    return torch.arange(tokens.size(1), device=tokens.device, dtype=torch.long).unsqueeze(0).expand_as(tokens)


def summarize_rank_environment(rank: int, local_rank: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "device": f"cuda:{local_rank}",
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
    }


# Snapshot helpers


def _maybe_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flattened.extend(int(x) for x in item)
            else:
                flattened.append(int(item))
        return flattened
    return []


def _snapshot_int_sequence(value: Any, *, max_items: int = 256) -> list[int] | None:
    if value is None:
        return None
    payload = _maybe_list(value)
    if len(payload) <= max_items:
        return payload
    return payload[:max_items]


def _snapshot_value(value: Any, *, max_items: int = 256, include_tensor_values: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        payload: dict[str, Any] = {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
        }
        if include_tensor_values and tensor.numel() <= max_items:
            payload["values"] = tensor.cpu().tolist()
        return payload
    if isinstance(value, (list, tuple)):
        sequence = list(value)
        payload = sequence[:max_items]
        if len(sequence) > max_items:
            payload.append(f"... truncated {len(sequence) - max_items} items")
        return [_snapshot_value(item, max_items=max_items, include_tensor_values=include_tensor_values) for item in payload]
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        payload = {
            str(key): _snapshot_value(item, max_items=max_items, include_tensor_values=include_tensor_values)
            for key, item in items
        }
        if len(value) > max_items:
            payload["__truncated__"] = len(value) - max_items
        return payload
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _observer_safe_record(observer: RouterSenseObserver, **payload: Any) -> None:
    try:
        payload.setdefault("ts_us", int(time.time() * 1e6))
        observer.record(**payload)
    except Exception:
        pass


def _snapshot_shape(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    return []


def _extract_int_list(value: Any) -> list[int]:
    if isinstance(value, dict):
        maybe_values = value.get("values")
        if isinstance(maybe_values, list):
            try:
                return [int(item) for item in maybe_values]
            except Exception:
                return []
    if isinstance(value, str):
        matches = re.findall(r"-?\d+", value)
        if matches:
            return [int(item) for item in matches]
    return _maybe_list(value)


# Observer helpers and summaries


def validate_observer_mode(mode: str) -> str:
    if mode not in {"off", "lightweight"}:
        raise ValueError(f"Unsupported observer_mode={mode!r}; expected 'off' or 'lightweight'")
    return mode


def summarize_observer_rows(rows: list[dict[str, Any]], *, rank: int) -> dict[str, Any]:
    remote_dispatch_rows = 0
    remote_combine_rows = 0
    local_dispatch_rows = 0
    local_combine_rows = 0
    warning_count = 0
    phase_counts: dict[str, int] = {}
    saw_dispatch_comm = False
    fallback_remote_dispatch_rows = 0
    fallback_local_dispatch_rows = 0
    for row in rows:
        phase = str(row.get("phase", "unknown"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if phase == "observer_warning":
            warning_count += 1
            continue
        if phase == "P0":
            expert_totals = _extract_int_list(row.get("routing_expert_totals_raw"))
            local_expert_indices = {int(item) for item in _extract_int_list(row.get("local_expert_indices_raw"))}
            if expert_totals and local_expert_indices:
                fallback_local_dispatch_rows += sum(
                    int(count) for expert_idx, count in enumerate(expert_totals) if expert_idx in local_expert_indices
                )
                fallback_remote_dispatch_rows += sum(
                    int(count) for expert_idx, count in enumerate(expert_totals) if expert_idx not in local_expert_indices
                )
        elif phase == "P0_comm":
            splits = _extract_int_list(row.get("input_splits_raw"))
            if splits:
                saw_dispatch_comm = True
                local_dispatch_rows += int(splits[rank]) if rank < len(splits) else 0
                remote_dispatch_rows += sum(int(value) for idx, value in enumerate(splits) if idx != rank)
        elif phase == "P1_comm":
            splits = _extract_int_list(row.get("output_splits_raw"))
            if splits:
                local_combine_rows += int(splits[rank]) if rank < len(splits) else 0
                remote_combine_rows += sum(int(value) for idx, value in enumerate(splits) if idx != rank)
    if not saw_dispatch_comm:
        local_dispatch_rows = fallback_local_dispatch_rows
        remote_dispatch_rows = fallback_remote_dispatch_rows
    return {
        "remote_dispatch_rows": remote_dispatch_rows,
        "remote_combine_rows": remote_combine_rows,
        "local_dispatch_rows": local_dispatch_rows,
        "local_combine_rows": local_combine_rows,
        "observer_warning_count": warning_count,
        "observer_phase_counts": phase_counts,
    }


def summarize_native_dispatchers(model: torch.nn.Module, *, rank: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    remote_dispatch_rows = 0
    remote_combine_rows = 0
    local_dispatch_rows = 0
    local_combine_rows = 0
    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None:
            continue
        input_splits = _extract_int_list(getattr(dispatcher, "input_splits", None))
        output_splits = _extract_int_list(getattr(dispatcher, "output_splits", None))
        local_expert_indices = _extract_int_list(getattr(dispatcher, "local_expert_indices", None))
        dispatch_remote = sum(int(value) for idx, value in enumerate(input_splits) if idx != rank)
        dispatch_local = int(input_splits[rank]) if rank < len(input_splits) else 0
        combine_remote = sum(int(value) for idx, value in enumerate(output_splits) if idx != rank)
        combine_local = int(output_splits[rank]) if rank < len(output_splits) else 0
        remote_dispatch_rows += dispatch_remote
        remote_combine_rows += combine_remote
        local_dispatch_rows += dispatch_local
        local_combine_rows += combine_local
        rows.append(
            {
                "layer": name,
                "dispatcher_class": type(dispatcher).__name__,
                "local_expert_indices": local_expert_indices,
                "input_splits": input_splits,
                "output_splits": output_splits,
                "remote_dispatch_rows": dispatch_remote,
                "remote_combine_rows": combine_remote,
                "local_dispatch_rows": dispatch_local,
                "local_combine_rows": combine_local,
            }
        )
    return {
        "remote_dispatch_rows": remote_dispatch_rows,
        "remote_combine_rows": remote_combine_rows,
        "local_dispatch_rows": local_dispatch_rows,
        "local_combine_rows": local_combine_rows,
        "dispatcher_rows": rows,
    }


def attach_dispatch_observer(
    observer: RouterSenseObserver,
    *,
    rank: int,
    local_rank: int,
    include_tensor_values: bool = False,
) -> Callable[[torch.nn.Module], None]:
    def _instrument(model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is None or getattr(dispatcher, "_routersense_wrapped", False):
                continue

            orig_pre = getattr(dispatcher, "dispatch_preprocess", None)
            orig_dispatch = dispatcher.token_dispatch
            orig_combine = dispatcher.token_combine
            orig_combine_post = getattr(dispatcher, "combine_postprocess", None)
            local_expert_indices = [int(idx) for idx in (getattr(dispatcher, "local_expert_indices", None) or [])]

            if orig_pre is not None:

                def wrapped_pre(hidden_states, routing_map, probs, _orig=orig_pre, _dispatcher=dispatcher, _name=name):
                    try:
                        _observer_safe_record(
                            observer,
                            phase="P0",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            hidden_shape=_snapshot_shape(hidden_states),
                            routing_map_shape=_snapshot_shape(routing_map),
                            probs_shape=_snapshot_shape(probs),
                            routing_expert_totals_raw=(
                                _snapshot_value(
                                    routing_map.sum(dim=0),
                                    include_tensor_values=True,
                                )
                                if include_tensor_values and isinstance(routing_map, torch.Tensor)
                                else None
                            ),
                            local_expert_indices_raw=_snapshot_int_sequence(local_expert_indices),
                            dispatcher_class=type(_dispatcher).__name__,
                        )
                    except Exception as exc:
                        _observer_safe_record(
                            observer,
                            phase="observer_warning",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            where="dispatch_preprocess_pre",
                            warning=f"{type(exc).__name__}: {exc}",
                        )
                    out = _orig(hidden_states, routing_map, probs)
                    try:
                        _observer_safe_record(
                            observer,
                            phase="dispatch_preprocess_after",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            input_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "input_splits", None)),
                            output_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "output_splits", None)),
                            packed_hidden_shape=_snapshot_shape(out[0] if isinstance(out, tuple) and len(out) >= 1 else None),
                            packed_probs_shape=_snapshot_shape(out[1] if isinstance(out, tuple) and len(out) >= 2 else None),
                        )
                    except Exception as exc:
                        _observer_safe_record(
                            observer,
                            phase="observer_warning",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            where="dispatch_preprocess_after",
                            warning=f"{type(exc).__name__}: {exc}",
                        )
                    return out

                dispatcher.dispatch_preprocess = wrapped_pre

            def wrapped_dispatch(hidden_states, probs, _orig=orig_dispatch, _dispatcher=dispatcher, _name=name):
                try:
                    _observer_safe_record(
                        observer,
                        phase="token_dispatch_enter",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        hidden_shape=_snapshot_shape(hidden_states),
                        probs_shape=_snapshot_shape(probs),
                        input_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "input_splits", None)),
                        output_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "output_splits", None)),
                    )
                except Exception as exc:
                    _observer_safe_record(
                        observer,
                        phase="observer_warning",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        where="token_dispatch_enter",
                        warning=f"{type(exc).__name__}: {exc}",
                    )
                result = _orig(hidden_states, probs)
                try:
                    _observer_safe_record(
                        observer,
                        phase="P0_comm",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        hidden_shape=_snapshot_shape(hidden_states),
                        probs_shape=_snapshot_shape(probs),
                        input_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "input_splits", None)),
                        num_global_tokens_per_local_expert_raw=_snapshot_int_sequence(
                            getattr(_dispatcher, "num_global_tokens_per_local_expert", None)
                        ),
                        dispatcher_class=type(_dispatcher).__name__,
                    )
                except Exception as exc:
                    _observer_safe_record(
                        observer,
                        phase="observer_warning",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        where="token_dispatch_post",
                        warning=f"{type(exc).__name__}: {exc}",
                    )
                return result

            def wrapped_combine(hidden_states, _orig=orig_combine, _dispatcher=dispatcher, _name=name):
                try:
                    _observer_safe_record(
                        observer,
                        phase="token_combine_enter",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        hidden_shape=_snapshot_shape(hidden_states),
                        output_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "output_splits", None)),
                    )
                except Exception as exc:
                    _observer_safe_record(
                        observer,
                        phase="observer_warning",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        where="token_combine_enter",
                        warning=f"{type(exc).__name__}: {exc}",
                    )
                result = _orig(hidden_states)
                try:
                    _observer_safe_record(
                        observer,
                        phase="P1_comm",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        hidden_shape=_snapshot_shape(hidden_states),
                        output_splits_raw=_snapshot_int_sequence(getattr(_dispatcher, "output_splits", None)),
                        dispatcher_class=type(_dispatcher).__name__,
                    )
                except Exception as exc:
                    _observer_safe_record(
                        observer,
                        phase="observer_warning",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        where="token_combine_post",
                        warning=f"{type(exc).__name__}: {exc}",
                    )
                return result

            if orig_combine_post is not None:

                def wrapped_combine_post(hidden_states, _orig=orig_combine_post, _dispatcher=dispatcher, _name=name):
                    try:
                        _observer_safe_record(
                            observer,
                            phase="P1",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            hidden_shape=_snapshot_shape(hidden_states),
                            dispatcher_class=type(_dispatcher).__name__,
                        )
                    except Exception as exc:
                        _observer_safe_record(
                            observer,
                            phase="observer_warning",
                            layer=_name,
                            rank=rank,
                            local_rank=local_rank,
                            where="combine_postprocess_post",
                            warning=f"{type(exc).__name__}: {exc}",
                        )
                    return _orig(hidden_states)

                dispatcher.combine_postprocess = wrapped_combine_post

            dispatcher.token_dispatch = wrapped_dispatch
            dispatcher.token_combine = wrapped_combine
            dispatcher._routersense_wrapped = True

    return _instrument


# Runtime attachment entrypoints


def attach_dispatch_facade(
    *,
    model: torch.nn.Module,
    config: RouterSenseInjectionConfig,
    rank: int,
    local_rank: int,
    run_id: str,
    step_id: str = "unknown",
    microbatch_id: str = "unknown",
    model_revision: str,
    request_table_hash: str,
    hostname: str,
    observer: RouterSenseObserver | None = None,
) -> RuntimeHandle:
    if getattr(model, "_routersense_runtime_owner", None) is not None:
        raise RuntimeAlreadyAttachedError("formal runtime already attached to model")
    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "_routersense_wrapped", False):
            raise LegacyObserverConflictError(f"legacy observer wrapper already attached at {name}")
    sample_dispatcher = None
    for module in model.named_modules():
        dispatcher = getattr(module[1], "token_dispatcher", None)
        if dispatcher is not None:
            sample_dispatcher = dispatcher
            break
    ep_process_group = getattr(sample_dispatcher, "ep_group", None) if sample_dispatcher is not None else None
    ep_group_ranks = get_process_group_ranks_safe(ep_process_group) if dist.is_initialized() else (rank,)
    model_revision_hash = hashlib.sha256(model_revision.encode("utf-8")).hexdigest()[:16]
    request_table_hash_digest = hashlib.sha256(request_table_hash.encode("utf-8")).hexdigest()[:16]
    runtime = RouterSenseInjectionRuntime(
        config=config,
        rank=rank,
        local_rank=local_rank,
        run_id=run_id,
        step_id=step_id,
        microbatch_id=microbatch_id,
        model_revision_hash=model_revision_hash,
        request_table_hash=request_table_hash_digest,
        hostname=hostname,
        observer=observer,
        ep_group_ranks=ep_group_ranks,
        ep_group_root_global_rank=get_process_group_root_safe(ep_process_group) if dist.is_initialized() else rank,
        ep_process_group=ep_process_group,
    )
    runtime.target_plan_control_group_handle = _create_control_group_handle(
        ep_process_group=ep_process_group,
        ep_group_ranks=ep_group_ranks,
        root_global_rank=get_process_group_root_safe(ep_process_group) if dist.is_initialized() else rank,
    )
    runtime.target_plan_control_group = runtime.target_plan_control_group_handle.process_group
    handle = RuntimeHandle(runtime=runtime)
    model._routersense_runtime_owner = str(run_id)
    handle.add_restore_callback(
        lambda _model=model: hasattr(_model, "_routersense_runtime_owner") and delattr(_model, "_routersense_runtime_owner")
    )
    transport_adapter = None
    original_all_to_all = None
    resolved_online_policy = resolve_online_policy_config(config)
    phase_policy_name = str(resolved_online_policy.builder_key) if resolved_online_policy is not None else ""
    if phase_policy_name and config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"} and sample_dispatcher is not None:
        try:
            import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod
        except ModuleNotFoundError:
            token_dispatcher_mod = None

        if token_dispatcher_mod is not None:
            original_all_to_all = token_dispatcher_mod.all_to_all
            p2p_group = None
            if config.execution_mode == "joint_window_async_p2p":
                p2p_group, p2p_status = _maybe_create_dedicated_p2p_group(
                    ep_group_ranks=ep_group_ranks,
                    local_rank=local_rank,
                )
                runtime._runtime_state.merge(p2p_status)
            transport_adapter = MegatronPhaseTransportAdapter(
                dispatcher_class=type(sample_dispatcher).__name__,
                dispatcher_module_sha256=None,
                p2p_group=p2p_group,
            )
            transport_adapter.timeline_hook = lambda event, **detail: runtime._timeline(
                event,
                layer_name=str(runtime.current_transport().get("layer_name") if runtime.current_transport() else "unknown"),
                **detail,
            )

            def wrapped_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, use_nccl_stream=False):
                return transport_adapter.maybe_execute(
                    group=group,
                    input_tensor=input_,
                    output_split_sizes=output_split_sizes_,
                    input_split_sizes=input_split_sizes,
                    original_all_to_all=original_all_to_all,
                    use_nccl_stream=use_nccl_stream,
                )

            token_dispatcher_mod.all_to_all = wrapped_all_to_all
            runtime.transport_adapter = transport_adapter
            runtime.original_all_to_all = original_all_to_all
            handle.add_restore_callback(lambda _mod=token_dispatcher_mod, _orig=original_all_to_all: setattr(_mod, "all_to_all", _orig))
    def _find_expert_timing_module(layer_module: torch.nn.Module) -> tuple[str, torch.nn.Module] | tuple[str, None]:
        candidates: list[tuple[str, torch.nn.Module]] = []
        for child_name, child_module in layer_module.named_modules():
            if not child_name:
                continue
            lowered = str(child_name).lower()
            if any(token in lowered for token in ("local_experts", "experts", "expert")) and getattr(child_module, "forward", None) is not None:
                candidates.append((str(child_name), child_module))
        if not candidates:
            return "", None
        preferred = [item for item in candidates if item[0].lower().endswith(("experts", "local_experts"))]
        return (preferred or candidates)[0]

    def _install_selected_layer_attribution_hooks(layer_name: str, layer_module: torch.nn.Module) -> None:
        if getattr(layer_module, "_routersense_selected_layer_attribution_wrapped", False):
            return

        def _selected_pre(_module, _args, _name=layer_name):
            runtime.record_selected_layer_enter(layer_name=str(_name))

        def _selected_post(_module, _args, _output, _name=layer_name):
            runtime.record_selected_layer_exit(layer_name=str(_name))

        selected_pre_handle = layer_module.register_forward_pre_hook(_selected_pre)
        selected_post_handle = layer_module.register_forward_hook(_selected_post)
        layer_module._routersense_selected_layer_attribution_wrapped = True
        def _restore_selected_layer_hooks(_module=layer_module, _pre=selected_pre_handle, _post=selected_post_handle):
            _pre.remove()
            _post.remove()
            if hasattr(_module, "_routersense_selected_layer_attribution_wrapped"):
                delattr(_module, "_routersense_selected_layer_attribution_wrapped")
        handle.add_restore_callback(_restore_selected_layer_hooks)
        expert_name, expert_module = _find_expert_timing_module(layer_module)
        if expert_module is None:
            runtime.record_expert_boundary_unavailable(layer_name=str(layer_name), reason="expert_module_not_found")
            return
        if getattr(expert_module, "_routersense_expert_attribution_wrapped", False):
            return

        def _expert_pre(_module, _args, _layer_name=layer_name, _expert_name=expert_name):
            runtime.record_expert_module_enter(layer_name=str(_layer_name), expert_module_name=str(_expert_name))

        def _expert_post(_module, _args, _output, _layer_name=layer_name, _expert_name=expert_name):
            runtime.record_expert_module_exit(layer_name=str(_layer_name), expert_module_name=str(_expert_name))

        expert_pre_handle = expert_module.register_forward_pre_hook(_expert_pre)
        expert_post_handle = expert_module.register_forward_hook(_expert_post)
        expert_module._routersense_expert_attribution_wrapped = True
        def _restore_expert_hooks(_module=expert_module, _pre=expert_pre_handle, _post=expert_post_handle):
            _pre.remove()
            _post.remove()
            if hasattr(_module, "_routersense_expert_attribution_wrapped"):
                delattr(_module, "_routersense_expert_attribution_wrapped")
        handle.add_restore_callback(_restore_expert_hooks)

    dispatcher_layer_names: list[str] = []
    for name, module in model.named_modules():
        if getattr(module, "token_dispatcher", None) is not None:
            dispatcher_layer_names.append(str(name))
    runtime.configure_hook_scope(available_layer_names=tuple(dispatcher_layer_names))
    if not getattr(model, "_routersense_forward_wrapped", False):
        original_forward = model.forward

        def wrapped_forward(*args: Any, _orig=original_forward, **kwargs: Any):
            runtime.handle(ForwardBeginEvent())
            try:
                result = _orig(*args, **kwargs)
            except BaseException as exc:
                runtime.handle(ForwardFailedEvent(error=exc))
                raise
            runtime.handle(ForwardEndEvent())
            return result

        model.forward = wrapped_forward
        model._routersense_forward_wrapped = True

        def _restore_forward_hooks(_model=model, _orig=original_forward):
            _model.forward = _orig
            if hasattr(_model, "_routersense_forward_wrapped"):
                delattr(_model, "_routersense_forward_wrapped")

        handle.add_restore_callback(_restore_forward_hooks)
    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None or getattr(dispatcher, "_routersense_facade_wrapped", False):
            continue
        layer_role = runtime.layer_role_for_name(str(name))
        if layer_role == "none":
            continue
        if layer_role == "selected":
            _install_selected_layer_attribution_hooks(str(name), module)

        dispatch_facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=dispatcher.token_dispatch,
            config=config,
        )
        combine_facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=dispatcher.token_combine,
            config=config,
        )

        def wrapped_dispatch(*args: Any, _facade=dispatch_facade, _dispatcher=dispatcher, _name=name, _role=layer_role, **kwargs: Any):
            hidden_states = args[0] if args else None
            probs = args[1] if len(args) > 1 else None
            runtime.handle(
                DispatchReadyEvent(
                    layer_name=str(_name),
                    dispatcher=_dispatcher,
                    packed_hidden_states=hidden_states,
                    packed_probs=probs,
                    layer_role=str(_role),
                )
            )
            try:
                result = _facade.dispatch(*args, **kwargs)
            except BaseException as exc:
                runtime.handle(
                    DispatchFailedEvent(
                        layer_name=str(_name),
                        dispatcher=_dispatcher,
                        packed_hidden_states=hidden_states,
                        error=exc,
                        layer_role=str(_role),
                    )
                )
                raise
            runtime.handle(
                DispatchCompleteEvent(
                    layer_name=str(_name),
                    dispatcher=_dispatcher,
                    packed_hidden_states=hidden_states,
                    result=result,
                    layer_role=str(_role),
                )
            )
            return result

        def wrapped_combine(*args: Any, _facade=combine_facade, _dispatcher=dispatcher, _name=name, **kwargs: Any):
            hidden_states = args[0] if args else None
            runtime.handle(
                CombineReadyEvent(
                    layer_name=str(_name),
                    dispatcher=_dispatcher,
                    packed_hidden_states=hidden_states,
                )
            )
            try:
                result = _facade.dispatch(*args, **kwargs)
            except BaseException as exc:
                runtime.handle(
                    CombineFailedEvent(
                        layer_name=str(_name),
                        dispatcher=_dispatcher,
                        packed_hidden_states=hidden_states,
                        error=exc,
                    )
                )
                raise
            runtime.handle(
                CombineCompleteEvent(
                    layer_name=str(_name),
                    dispatcher=_dispatcher,
                    packed_hidden_states=hidden_states,
                    result=result,
                )
            )
            return result

        dispatcher.token_dispatch = wrapped_dispatch
        if layer_role == "selected":
            dispatcher.token_combine = wrapped_combine
        dispatcher._routersense_facade_wrapped = True
        def _restore_dispatcher(_dispatcher=dispatcher, _orig_dispatch=dispatch_facade.native_dispatcher, _orig_combine=combine_facade.native_dispatcher, _role=layer_role):
            _dispatcher.token_dispatch = _orig_dispatch
            if _role == "selected":
                _dispatcher.token_combine = _orig_combine
            if hasattr(_dispatcher, "_routersense_facade_wrapped"):
                delattr(_dispatcher, "_routersense_facade_wrapped")
        handle.add_restore_callback(_restore_dispatcher)
    handle.add_close_callback(runtime._cleanup_target_plan_runtime)
    handle.add_close_callback(
        lambda _runtime=runtime: (
            _CONTROL_GROUP_REGISTRY.get(getattr(_runtime.target_plan_control_group_handle, "_registry_key", ()))
            and _CONTROL_GROUP_REGISTRY[getattr(_runtime.target_plan_control_group_handle, "_registry_key", ())].close(
                local_ep_group_ranks=tuple(int(rank) for rank in _runtime.ep_group_ranks)
            )
        )
    )
    return handle


def attach_formal_online_runtime(
    *,
    model: torch.nn.Module,
    runtime_config: OnlineRuntimeConfig,
    rank: int,
    local_rank: int,
    run_id: str,
    model_revision: str,
    request_table_hash: str,
    hostname: str,
    step_id: str = "unknown",
    microbatch_id: str = "unknown",
    observer: RouterSenseObserver | None = None,
) -> RuntimeHandle:
    p2_hint_mode = (
        "calibrated_artifact"
        if bool(runtime_config.policy_parameters.calibrated_p2_enabled)
        else runtime_config.policy_parameters.p2_hint_mode
    )
    injection_config = RouterSenseInjectionConfig(
        policy=runtime_config.policy_name,
        scheduler_mode="disabled",
        execution_mode=runtime_config.execution_mode,
        future_hint_mode="none",
        p2_hint_mode=p2_hint_mode,
        control_mode=runtime_config.control_mode,
        bucket_mode=str(getattr(runtime_config.execution_selection, "bucket_mode", "dynamic_current")),
        bucket_rows=runtime_config.execution_selection.bucket_rows,
        p0_weight=runtime_config.policy_parameters.p0_weight,
        p1_reservation_weight=runtime_config.policy_parameters.p1_reservation_weight,
        p2_hint_weight=runtime_config.policy_parameters.p2_hint_weight,
        residual_weight=float(getattr(runtime_config.policy_parameters, "residual_weight", 0.75)),
        barrier_weight=float(getattr(runtime_config.policy_parameters, "barrier_weight", 1.75)),
        age_weight=float(getattr(runtime_config.policy_parameters, "age_weight", 0.15)),
        prediction_weight=float(getattr(runtime_config.policy_parameters, "prediction_weight", 0.35)),
        p2_hint_artifact=runtime_config.policy_parameters.p2_hint_artifact,
        online_p2_predictor=str(getattr(runtime_config.policy_parameters, "online_p2_predictor", "copy_current_dispatch")),
        safe_projection_mode=str(getattr(runtime_config.policy_parameters, "safe_projection_mode", "host_select")),
        schedule_layer_selector=runtime_config.execution_selection.layer_selector,
        schedule_phase_selector=runtime_config.execution_selection.phase_selector,
        selected_layer_ids=tuple(str(item) for item in getattr(runtime_config.execution_selection, "selected_layer_ids", ()) or ()),
        capture_phase_tensors=bool(runtime_config.observation.get("capture_enabled", False)),
        capture_expert_trace=bool(runtime_config.observation.get("capture_expert_trace", False)),
        stop_after_selected_layer=bool(runtime_config.validation.stop_after_selected_layer),
        executor_heartbeat_path=str(runtime_config.validation.executor_heartbeat_path),
        executor_phase_timeout_sec=int(runtime_config.validation.executor_phase_timeout_sec),
        observation_profile=str(runtime_config.observation.get("profile", "minimal")),
        invariant_mode=str(runtime_config.observation.get("invariant_mode", "diagnostic")),
        legacy_compiler_bridge=bool(runtime_config.observation.get("legacy_compiler_bridge", False)),
        capture_layer_selector=str(runtime_config.observation.get("capture_layer_selector", "")),
        capture_phase_selector=str(runtime_config.observation.get("capture_phase_selector", "")),
        heartbeat_enabled=bool(runtime_config.observation.get("heartbeat_enabled", False)),
        per_wave_timing_enabled=bool(runtime_config.observation.get("per_wave_timing_enabled", False)),
        replay_trace_enabled=bool(runtime_config.observation.get("replay_trace_enabled", False)),
        preflight_mode=str(getattr(runtime_config.validation, "preflight_mode", "full")),
    )
    return attach_dispatch_facade(
        model=model,
        config=injection_config,
        rank=rank,
        local_rank=local_rank,
        run_id=run_id,
        model_revision=model_revision,
        request_table_hash=request_table_hash,
        hostname=hostname,
        step_id=step_id,
        microbatch_id=microbatch_id,
        observer=observer,
    )


# Small distributed/export helpers


def gather_rank_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return [payload]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return [item for item in gathered if item is not None]


def destroy_distributed() -> None:
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
