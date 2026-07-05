from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution import MegatronPhaseTransportAdapter
from rs.runtime.online.megatron_ep.observer import RouterSenseObserver
from rs.scheduling.policy.registry import supported_phase_policies
from rs.runtime.online.megatron_ep._facade import RouterSenseDispatcherFacade
from rs.runtime.online.megatron_ep._lifecycle import RouterSenseInjectionRuntime


@dataclass
class StageStatus:
    stage: str
    rank: int
    ok: bool
    detail: str = ""


def get_process_group_ranks_safe(group: dist.ProcessGroup | None) -> tuple[int, ...]:
    if group is None:
        return tuple(range(dist.get_world_size())) if dist.is_initialized() else (0,)
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    return tuple(range(dist.get_world_size(group)))


def get_process_group_root_safe(group: dist.ProcessGroup | None) -> int:
    ranks = get_process_group_ranks_safe(group)
    return int(ranks[0]) if ranks else 0


def _as_path(model: str) -> Path:
    return Path(model).expanduser()


def model_is_local_path(model: str) -> bool:
    return _as_path(model).exists()


def load_prompts(prompt_file: str | Path) -> list[str]:
    payload = json.loads(Path(prompt_file).read_text(encoding="utf-8"))
    prompts = [str(item) for item in payload.get("prompts", [])]
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


def _snapshot_value(value: Any, *, max_items: int = 256) -> Any:
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
        if tensor.numel() <= max_items:
            payload["values"] = tensor.cpu().tolist()
        return payload
    if isinstance(value, (list, tuple)):
        sequence = list(value)
        payload = sequence[:max_items]
        if len(sequence) > max_items:
            payload.append(f"... truncated {len(sequence) - max_items} items")
        return [_snapshot_value(item, max_items=max_items) for item in payload]
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        payload = {str(key): _snapshot_value(item, max_items=max_items) for key, item in items}
        if len(value) > max_items:
            payload["__truncated__"] = len(value) - max_items
        return payload
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _observer_safe_record(observer: RouterSenseObserver, **payload: Any) -> None:
    try:
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
                            routing_expert_totals_raw=_snapshot_value(routing_map.sum(dim=0) if isinstance(routing_map, torch.Tensor) else None),
                            local_expert_indices_raw=_snapshot_value(local_expert_indices),
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
                            input_splits_raw=_snapshot_value(getattr(_dispatcher, "input_splits", None)),
                            output_splits_raw=_snapshot_value(getattr(_dispatcher, "output_splits", None)),
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
                        input_splits_raw=_snapshot_value(getattr(_dispatcher, "input_splits", None)),
                        output_splits_raw=_snapshot_value(getattr(_dispatcher, "output_splits", None)),
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
                        input_splits_raw=_snapshot_value(getattr(_dispatcher, "input_splits", None)),
                        num_global_tokens_per_local_expert_raw=_snapshot_value(
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
                result = _orig(hidden_states)
                try:
                    _observer_safe_record(
                        observer,
                        phase="P1_comm",
                        layer=_name,
                        rank=rank,
                        local_rank=local_rank,
                        hidden_shape=_snapshot_shape(hidden_states),
                        output_splits_raw=_snapshot_value(getattr(_dispatcher, "output_splits", None)),
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


def validate_observer_mode(mode: str) -> str:
    if mode not in {"off", "lightweight"}:
        raise ValueError(f"Unsupported observer_mode={mode!r}; expected 'off' or 'lightweight'")
    return mode


def attach_dispatch_facade(
    *,
    model: torch.nn.Module,
    config: RouterSenseInjectionConfig,
    rank: int,
    local_rank: int,
    run_id: str,
    model_revision: str,
    request_table_hash: str,
    hostname: str,
    step_id: str = "unknown",
    microbatch_id: str = "unknown",
    observer: RouterSenseObserver | None = None,
) -> RouterSenseInjectionRuntime:
    sample_dispatcher = None
    for module in model.named_modules():
        dispatcher = getattr(module[1], "token_dispatcher", None)
        if dispatcher is not None:
            sample_dispatcher = dispatcher
            break
    ep_process_group = getattr(sample_dispatcher, "ep_group", None) if sample_dispatcher is not None else None
    ep_group_ranks = (
        get_process_group_ranks_safe(ep_process_group)
        if dist.is_initialized()
        else (rank,)
    )
    runtime = RouterSenseInjectionRuntime(
        config=config,
        rank=rank,
        local_rank=local_rank,
        run_id=run_id,
        step_id=step_id,
        microbatch_id=microbatch_id,
        model_revision_hash=hashlib.sha256(model_revision.encode("utf-8")).hexdigest()[:16],
        request_table_hash=hashlib.sha256(request_table_hash.encode("utf-8")).hexdigest()[:16],
        hostname=hostname,
        observer=observer,
        ep_group_ranks=ep_group_ranks,
        ep_group_root_global_rank=get_process_group_root_safe(ep_process_group) if dist.is_initialized() else rank,
        ep_process_group=ep_process_group,
    )
    transport_adapter = None
    original_all_to_all = None
    supported_policies = set(supported_phase_policies())
    phase_policy_name = config.policy or (
        config.scheduler_mode if config.scheduler_mode in supported_policies else ""
    )
    if phase_policy_name and config.execution_mode == "phase_sync_wave" and sample_dispatcher is not None:
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod

        original_all_to_all = token_dispatcher_mod.all_to_all
        transport_adapter = MegatronPhaseTransportAdapter(
            dispatcher_class=type(sample_dispatcher).__name__,
            dispatcher_module_sha256=None,
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
            )

        token_dispatcher_mod.all_to_all = wrapped_all_to_all
        runtime.transport_adapter = transport_adapter
        runtime.original_all_to_all = original_all_to_all
    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None or getattr(dispatcher, "_routersense_facade_wrapped", False):
            continue

        dispatch_facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=dispatcher.token_dispatch,
            config=config,
        )
        combine_facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=dispatcher.token_combine,
            config=config,
        )

        def wrapped_dispatch(*args: Any, _facade=dispatch_facade, _dispatcher=dispatcher, _name=name, **kwargs: Any):
            hidden_states = args[0] if args else None
            probs = args[1] if len(args) > 1 else None
            runtime.before_token_dispatch(
                layer_name=_name,
                dispatcher=_dispatcher,
                packed_hidden_states=hidden_states,
                packed_probs=probs,
            )
            runtime.mark_token_dispatch_committed(layer_name=_name)
            result = _facade.dispatch(*args, **kwargs)
            runtime.capture_phase_transport_output(layer_name=_name, phase="P0", result=result, dispatcher=_dispatcher)
            runtime.after_token_dispatch(layer_name=_name)
            runtime.on_dispatch(layer_name=_name, dispatcher=_dispatcher, hidden_states=hidden_states)
            return result

        def wrapped_combine(*args: Any, _facade=combine_facade, _dispatcher=dispatcher, _name=name, **kwargs: Any):
            hidden_states = args[0] if args else None
            runtime.before_token_combine(layer_name=_name, dispatcher=_dispatcher, packed_hidden_states=hidden_states)
            result = _facade.dispatch(*args, **kwargs)
            runtime.capture_phase_transport_output(layer_name=_name, phase="P1", result=result, dispatcher=_dispatcher)
            runtime.after_token_combine(layer_name=_name)
            runtime.on_combine(layer_name=_name, dispatcher=_dispatcher, hidden_states=hidden_states)
            return result

        dispatcher.token_dispatch = wrapped_dispatch
        dispatcher.token_combine = wrapped_combine
        dispatcher._routersense_facade_wrapped = True
    return runtime


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
