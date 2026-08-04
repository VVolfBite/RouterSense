from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import socket
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .input_contract import append_input_manifest, build_distributed_tokens, persist_input_artifact
from .model_support import ModelSupportError, inspect_hf_model, validate_generic_text_moe


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _capture_runtime_config() -> dict[str, Any] | None:
    path = os.environ.get("RS_SIM_CAPTURE_CONFIG")
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _capture_output_dir() -> Path | None:
    payload = _capture_runtime_config()
    if payload is None:
        return None
    try:
        return Path(payload["output_dir"]).expanduser().resolve()
    except Exception:
        return None


def _write_runner_report(payload: dict[str, Any]) -> None:
    output = _capture_output_dir()
    if output is None:
        return
    target = output / "runner"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"rank{_rank():04d}_megatron_runner.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dtype(name: str):
    import torch

    normalized = str(name).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported runner dtype: {name}") from exc


def _set_if_present(owner: Any, name: str, value: Any) -> bool:
    if not hasattr(owner, name):
        return False
    setattr(owner, name, value)
    return True


def _import_auto_bridge():
    # The public namespace has remained ``megatron.bridge`` across the 0.1 and
    # current releases.  Keep the import isolated so the error report contains
    # the installed environment rather than failing at module import time.
    from megatron.bridge import AutoBridge

    return AutoBridge


def _bridge_can_handle(AutoBridge: Any, model_path: str) -> tuple[bool | None, str | None]:
    method = getattr(AutoBridge, "can_handle", None)
    if not callable(method):
        return None, None
    try:
        return bool(method(model_path)), None
    except Exception as exc:  # old releases may not accept local paths here
        return None, f"{type(exc).__name__}: {exc}"


class _BridgeWeightLoadAudit(logging.Handler):
    """Collect bridge mapping failures without mutating third-party packages."""

    _PREFIX = "No mapping found for megatron_param:"

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.unmapped_parameter_names: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if self._PREFIX not in message:
            return
        name = message.split(self._PREFIX, 1)[1].strip()
        if name and name not in self.unmapped_parameter_names:
            self.unmapped_parameter_names.append(name)

    def report(self, *, task_stats: dict[str, int] | None = None) -> dict[str, Any]:
        names = sorted(self.unmapped_parameter_names)
        task_stats = task_stats or {}
        return {
            "status": "PASS" if not names else "FAILED",
            "mapped_conversion_task_count": int(task_stats.get("non_null_task_count", 0)),
            "rank_absent_task_count": int(task_stats.get("null_task_count", 0)),
            "critical_unmapped_parameter_count": len(names),
            "unmapped_parameter_count": len(names),
            "unmapped_parameter_names": names,
        }


@contextmanager
def _capture_bridge_weight_load_warnings():
    handler = _BridgeWeightLoadAudit()
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)


def _install_bridge_ep_none_task_compat(bridge: Any) -> dict[str, Any]:
    """Filter rank-absent ``None`` conversion tasks in Bridge 0.2.x.

    Some EP ranks legitimately do not own a parameter.  Megatron-Bridge 0.2.2
    may place ``None`` in the conversion-task list and then dereference it.
    Wrapping the bridge instance keeps the compatibility fix local to this
    process and avoids editing site-packages.
    """

    original = getattr(bridge, "build_conversion_tasks", None)
    stats: dict[str, Any] = {
        "installed": False,
        "total_task_count": 0,
        "non_null_task_count": 0,
        "null_task_count": 0,
    }
    if not callable(original):
        return stats
    if getattr(original, "__rs_sim_ep_none_task_compat__", False):
        stats["installed"] = True
        return stats

    def wrapped(*args: Any, **kwargs: Any):
        tasks = original(*args, **kwargs)
        if tasks is None:
            stats["total_task_count"] = 0
            stats["non_null_task_count"] = 0
            stats["null_task_count"] = 0
            return []
        materialized = list(tasks)
        filtered = [task for task in materialized if task is not None]
        stats["total_task_count"] = len(materialized)
        stats["non_null_task_count"] = len(filtered)
        stats["null_task_count"] = len(materialized) - len(filtered)
        return filtered

    setattr(wrapped, "__rs_sim_ep_none_task_compat__", True)
    setattr(bridge, "build_conversion_tasks", wrapped)
    stats["installed"] = True
    return stats



def _call_compatible(callable_obj: Any, /, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    filtered = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(*args, **filtered)


def _provider_from_hf(args, AutoBridge):
    bridge = _call_compatible(
        AutoBridge.from_hf_pretrained,
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    provider = _call_compatible(
        bridge.to_megatron_provider,
        load_weights=not args.random_init,
    )
    return bridge, provider, "HF"


def _provider_from_megatron(args, AutoBridge):
    if not args.hf_config_path:
        raise ValueError("--hf-config-path is required for --model-format=megatron")
    bridge = _call_compatible(
        AutoBridge.from_hf_pretrained,
        args.hf_config_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    provider = _call_compatible(bridge.to_megatron_provider, load_weights=False)
    return bridge, provider, "MEGATRON"


def _configure_provider(provider: Any, args, torch) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    values = {
        "tensor_model_parallel_size": int(args.tp),
        "pipeline_model_parallel_size": int(args.pp),
        "expert_model_parallel_size": int(args.ep),
        "expert_tensor_parallel_size": int(args.etp),
        "context_parallel_size": int(args.cp),
    }
    for name, value in values.items():
        if not _set_if_present(provider, name, value):
            if name in {"expert_model_parallel_size", "tensor_model_parallel_size", "pipeline_model_parallel_size"}:
                raise RuntimeError(f"installed Megatron Bridge provider lacks required attribute {name}")
        else:
            configured[name] = value

    dtype = _dtype(args.dtype)
    for name, value in (
        ("params_dtype", dtype),
        ("pipeline_dtype", dtype),
        ("bf16", dtype is torch.bfloat16),
        ("fp16", dtype is torch.float16),
        ("sequence_parallel", bool(args.tp > 1)),
    ):
        if _set_if_present(provider, name, value):
            configured[name] = str(value) if name.endswith("dtype") else value

    # Preserve model-family defaults for expert layout and permutation.  In
    # particular, OLMoE Bridge 0.2.2 requires ``moe_grouped_gemm=True`` for
    # its registered expert-weight mappings.  Forcing SequentialMLP here
    # changes parameter names to ``local_experts.*`` and silently leaves all
    # expert FC weights unmapped.  The capture hooks operate at MoELayer and
    # dispatcher boundaries and do not require disabling grouped GEMM.
    for name, value in (
        ("moe_token_dispatcher_type", args.dispatcher),
        ("cuda_graph_impl", "none"),
    ):
        if _set_if_present(provider, name, value):
            configured[name] = value
    for name in ("moe_grouped_gemm", "moe_permute_fusion"):
        if hasattr(provider, name):
            configured[name] = getattr(provider, name)
    return configured


def _forward_kwargs(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    kwargs: dict[str, Any] = {}
    accepts_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())

    # Different Bridge/MCore generations use either input_ids or tokens. Never
    # pass both: models accepting **kwargs may forward the duplicate semantic
    # input into deeper modules and fail with a multiple-value error.
    if "input_ids" in parameters or ("tokens" not in parameters and accepts_kwargs):
        kwargs["input_ids"] = batch["tokens"]
    elif "tokens" in parameters:
        kwargs["tokens"] = batch["tokens"]

    optional = {
        "position_ids": batch["position_ids"],
        "attention_mask": None,
        "labels": None,
        "inference_context": batch.get("inference_context"),
        "inference_params": None,
        "runtime_gather_output": True,
    }
    for name, value in optional.items():
        if name in parameters or accepts_kwargs:
            kwargs[name] = value
    if "input_ids" not in kwargs and "tokens" not in kwargs:
        raise RuntimeError(
            f"unsupported Megatron model forward signature: {signature}; no input_ids/tokens parameter"
        )
    return kwargs


class _SingleBatchIterator:
    def __init__(self, tokens, position_ids, inference_context=None):
        self._batch = {
            "tokens": tokens,
            "position_ids": position_ids,
            "inference_context": inference_context,
        }
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self._batch


def _forward_step(data_iterator, model, **_kwargs):
    batch = next(data_iterator)
    output = model(**_forward_kwargs(model, batch))
    if isinstance(output, tuple):
        output = output[0]

    def loss_func(value, **kwargs):
        del kwargs
        return value

    return output, loss_func


def _run_one_forward(models, tokens, position_ids) -> Any:
    import torch

    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    # Megatron-Core 0.15.0 does not expose InferenceMode.  Newer releases do.
    # Use the native context when available and a semantically equivalent
    # torch inference context otherwise.
    try:
        from megatron.core.inference.utils import InferenceMode

        inference_context_manager = InferenceMode.active()
    except (ImportError, AttributeError):
        inference_context_manager = torch.inference_mode()

    fwd_bwd = get_forward_backward_func()
    inference_context = StaticInferenceContext(
        max_batch_size=int(tokens.shape[0]),
        max_sequence_length=int(tokens.shape[1]),
    )
    with inference_context_manager:
        return fwd_bwd(
            forward_step_func=_forward_step,
            data_iterator=_SingleBatchIterator(tokens, position_ids, inference_context),
            model=models,
            num_microbatches=1,
            forward_only=True,
            seq_length=int(tokens.shape[1]),
            micro_batch_size=int(tokens.shape[0]),
            collect_non_loss_data=True,
        )


def _disable_mtp(models: list[Any]) -> None:
    try:
        from megatron.bridge.utils.common_utils import disable_mtp_for_inference
    except Exception:
        return
    for model in models:
        try:
            disable_mtp_for_inference(model)
        except Exception:
            pass


def _local_moe_layers(models: list[Any]) -> list[int]:
    layers: set[int] = set()
    fallback = 0
    for model in models:
        for module in model.modules():
            cls = type(module)
            identity = f"{cls.__module__}.{cls.__qualname__}".lower()
            if not (identity.endswith(".moelayer") or ".moe.moe_layer.moelayer" in identity):
                continue
            value = getattr(module, "layer_number", None)
            if value is None:
                value = fallback
                fallback += 1
            try:
                layers.add(int(value))
            except (TypeError, ValueError):
                layers.add(fallback)
                fallback += 1
    return sorted(layers)


def _load_models(bridge: Any, provider: Any, args, source_format: str):
    if source_format == "MEGATRON":
        if not hasattr(bridge, "load_megatron_model"):
            raise RuntimeError("installed Megatron Bridge cannot load native Megatron checkpoints")
        return bridge.load_megatron_model(
            args.model_path,
            mp_overrides={
                "tensor_model_parallel_size": int(args.tp),
                "pipeline_model_parallel_size": int(args.pp),
                "expert_model_parallel_size": int(args.ep),
                "expert_tensor_parallel_size": int(args.etp),
                "pipeline_dtype": _dtype(args.dtype),
            },
            wrap_with_ddp=False,
        )

    pre_wrap_hook = getattr(provider, "pre_wrap_hook", None)
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    # Older Bridge variants did not attach the HF import as a provider hook.
    # Only call the collective loader when no hook was registered.
    if not args.random_init and pre_wrap_hook is None and hasattr(bridge, "load_hf_weights"):
        bridge.load_hf_weights(models)
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RouterSense self-contained Megatron Bridge text-MoE trace runner")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-format", choices=("auto", "hf", "megatron"), default="auto")
    parser.add_argument("--hf-config-path")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dispatcher", choices=("alltoall", "allgather"), default="alltoall")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--qualification-samples", type=int, default=0)
    parser.add_argument("--qualification-threshold-ratio", type=float, default=0.05)
    parser.add_argument("--save-input-ids", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    return parser


def _resolve_format(args) -> str:
    if args.model_format != "auto":
        return args.model_format.upper()
    root = Path(args.model_path)
    if (root / "config.json").is_file():
        return "HF"
    if (root / "run_config.yaml").is_file() or any(root.glob("iter_*")):
        return "MEGATRON"
    raise ModelSupportError(
        f"cannot infer model format at {root}; expected Hugging Face config.json or a Megatron checkpoint"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time_ns()
    report: dict[str, Any] = {
        "schema_version": "RS_SIM_MEGATRON_BRIDGE_RUNNER",
        "status": "FAILED",
        "rank": _rank(),
        "world_size": _world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "node_rank": int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0"))),
        "hostname": socket.gethostname(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "arguments": vars(args),
        "started_at_unix_ns": started,
    }
    try:
        if min(args.tp, args.pp, args.ep, args.etp, args.cp, args.dp, args.seq_length, args.micro_batch_size, args.samples) <= 0:
            raise ValueError("parallel dimensions, sequence length, batch size, and samples must be positive")
        if int(args.warmup_samples) < 0 or int(args.qualification_samples) < 0:
            raise ValueError("warmup and qualification samples must be nonnegative")
        if not 0.0 <= float(args.qualification_threshold_ratio) <= 1.0:
            raise ValueError("qualification threshold ratio must be in [0,1]")
        global_batch_size = int(args.global_batch_size or (args.micro_batch_size * args.ep))
        if global_batch_size != int(args.micro_batch_size) * int(args.ep):
            raise ValueError("global batch size must equal micro batch size multiplied by EP")
        args.global_batch_size = global_batch_size
        expected_world = int(args.tp * args.pp * args.ep * args.dp)
        if _world_size() != expected_world:
            raise RuntimeError(
                f"WORLD_SIZE={_world_size()} but TP*PP*EP*DP={expected_world}; launch configuration is inconsistent"
            )
        # Current fixture semantics are EP-rank based.  TP/PP/DP support needs a
        # separate source-rank projection contract and is intentionally rejected
        # instead of producing duplicate or ambiguous rows.
        if args.tp != 1 or args.pp != 1 or args.dp != 1 or args.cp != 1:
            raise RuntimeError(
                "Current-P12 trace collection currently requires TP=PP=DP=CP=1; "
                "all supported model families are handled, but mixed parallel-rank projection is fail-closed"
            )

        source_format = _resolve_format(args)
        report["source_format"] = source_format
        inspection_path = args.model_path if source_format == "HF" else args.hf_config_path
        if not inspection_path:
            raise ModelSupportError("native Megatron checkpoints require --hf-config-path")
        inspection = inspect_hf_model(inspection_path)
        validate_generic_text_moe(inspection, ep=args.ep)
        report["model_inspection"] = inspection.to_dict()

        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the official Megatron runner requires NVIDIA GPUs")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("no local CUDA device is visible")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        report["torch_version"] = torch.__version__
        report["cuda_version"] = torch.version.cuda
        report["device"] = torch.cuda.get_device_name(local_rank)

        AutoBridge = _import_auto_bridge()
        can_handle, can_handle_error = _bridge_can_handle(AutoBridge, inspection_path)
        report["autobridge_can_handle"] = can_handle
        report["autobridge_can_handle_error"] = can_handle_error
        # Actual bridge/provider construction below is authoritative. Older
        # releases may return a conservative False for local paths.
        if source_format == "HF":
            bridge, provider, _ = _provider_from_hf(args, AutoBridge)
        else:
            bridge, provider, _ = _provider_from_megatron(args, AutoBridge)
        report["provider_class"] = f"{type(provider).__module__}.{type(provider).__qualname__}"
        report["provider_configuration"] = _configure_provider(provider, args, torch)
        if (
            str(inspection.model_type).lower() == "olmoe"
            and hasattr(provider, "moe_grouped_gemm")
            and not bool(getattr(provider, "moe_grouped_gemm"))
        ):
            raise RuntimeError(
                "OLMoE provider resolved moe_grouped_gemm=False; this produces "
                "SequentialMLP local_experts parameter names that Bridge 0.2.2 "
                "cannot map from the Hugging Face checkpoint"
            )
        bridge_compat = _install_bridge_ep_none_task_compat(bridge)
        report["bridge_ep_none_task_compat"] = bridge_compat
        provider.finalize()
        provider.initialize_model_parallel(seed=int(args.seed))

        with _capture_bridge_weight_load_warnings() as weight_audit:
            models = list(_load_models(bridge, provider, args, source_format))
        weight_report = weight_audit.report(task_stats=bridge_compat)
        report["weight_load_validation"] = weight_report
        if not args.random_init and weight_report["critical_unmapped_parameter_count"] != 0:
            preview = weight_report["unmapped_parameter_names"][:8]
            raise RuntimeError(
                "pretrained weight loading left unmapped Megatron parameters; "
                f"count={weight_report['critical_unmapped_parameter_count']} preview={preview}"
            )

        models = [model.cuda() for model in models]
        for model in models:
            model.eval()
        _disable_mtp(models)
        report["model_chunk_count"] = len(models)
        local_moe_layers = _local_moe_layers(models)
        report["local_moe_layer_ids"] = local_moe_layers
        if len(local_moe_layers) < 2:
            raise RuntimeError(
                "loaded model exposes fewer than two local Megatron MoELayer instances; "
                f"detected={local_moe_layers}"
            )

        vocab_size = int(
            getattr(provider, "vocab_size", 0)
            or inspection.vocab_size
            or 0
        )
        if vocab_size <= 1:
            raise RuntimeError("could not infer a valid vocabulary size from provider/config")
        report["vocab_size"] = vocab_size

        from rs_sim.trace.collection import (
            finish_capture_sample,
            flush_capture,
            set_capture_context,
            set_capture_enabled,
        )

        output_dir = _capture_output_dir()
        runtime_capture_config = _capture_runtime_config() or {}
        if output_dir is None:
            raise RuntimeError("capture output directory is unavailable")
        source_rank = _rank()
        capture_metadata = dict(runtime_capture_config.get("capture", {}))
        sample_id_prefix = str(capture_metadata.get("sample_id_prefix", capture_metadata.get("capture_id", "sample")))

        def make_input(sample_index: int, *, warmup: bool = False):
            tokens, global_indices = build_distributed_tokens(
                torch,
                vocab_size=vocab_size,
                seq_length=int(args.seq_length),
                local_batch_size=int(args.micro_batch_size),
                source_rank=source_rank,
                base_seed=int(args.seed),
                sample_index=int(sample_index),
                device="cuda",
                warmup=warmup,
            )
            position_ids = torch.arange(
                int(args.seq_length), device="cuda", dtype=torch.long
            ).unsqueeze(0).expand(int(args.micro_batch_size), -1)
            return tokens, position_ids, global_indices

        def timed_forward(tokens, position_ids) -> int:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            with torch.no_grad():
                _run_one_forward(models, tokens, position_ids)
            end_event.record()
            end_event.synchronize()
            return max(0, int(round(float(start_event.elapsed_time(end_event)) * 1_000_000.0)))

        # Warmup executes the identical distributed input contract with capture
        # disabled, so allocator/JIT initialization cannot enter fixtures.
        set_capture_enabled(False)
        for warmup in range(int(args.warmup_samples)):
            set_capture_context(request_id=f"warmup-{warmup}", decode_step=-(warmup + 1))
            tokens, position_ids, _ = make_input(warmup, warmup=True)
            timed_forward(tokens, position_ids)

        baseline_durations_ns: list[int] = []
        for sample in range(int(args.qualification_samples)):
            set_capture_context(request_id=f"qualification-disabled-{sample}", decode_step=-(1000 + sample))
            tokens, position_ids, _ = make_input(sample)
            baseline_durations_ns.append(timed_forward(tokens, position_ids))

        enabled_durations_ns: list[int] = []
        set_capture_enabled(True)
        for sample in range(int(args.samples)):
            set_capture_context(request_id=f"sample-{sample}", decode_step=sample)
            tokens, position_ids, global_indices = make_input(sample)
            enabled_durations_ns.append(timed_forward(tokens, position_ids))
            finish_capture_sample(decode_step=sample)
            input_payload = persist_input_artifact(
                tokens,
                output_dir=output_dir,
                rank=source_rank,
                sample_index=sample,
                global_indices=global_indices,
                seq_length=int(args.seq_length),
                local_batch_size=int(args.micro_batch_size),
                global_batch_size=int(args.global_batch_size),
                base_seed=int(args.seed),
                save_token_ids=bool(args.save_input_ids),
            )
            input_payload["capture_id"] = capture_metadata.get("capture_id")
            input_payload["request_id"] = f"sample-{sample}"
            input_payload["decode_step"] = int(sample)
            input_payload["sample_id"] = f"{sample_id_prefix}:step{sample}"
            input_payload["model_id"] = capture_metadata.get("model_id")
            input_payload["router_topk"] = inspection.router_topk
            input_payload["global_assignment_rows"] = (
                None if inspection.router_topk is None
                else int(args.global_batch_size) * int(args.seq_length) * int(inspection.router_topk)
            )
            append_input_manifest(output_dir, source_rank, input_payload)

        qualification: dict[str, Any]
        if int(args.qualification_samples) > 0:
            paired_count = min(len(baseline_durations_ns), len(enabled_durations_ns))
            baseline_median = float(statistics.median(baseline_durations_ns[:paired_count]))
            enabled_median = float(statistics.median(enabled_durations_ns[:paired_count]))
            local_overhead = max(-1.0, (enabled_median / baseline_median) - 1.0) if baseline_median > 0 else 1.0
            overhead_tensor = torch.tensor([local_overhead], device="cuda", dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(overhead_tensor, op=dist.ReduceOp.MAX)
            global_max_overhead = float(overhead_tensor.item())
            eligible = global_max_overhead <= float(args.qualification_threshold_ratio)
            qualification = {
                "status": "PASS" if eligible else "FAILED",
                "method": "HOOK_RECORDING_DISABLED_VS_ENABLED_PAIRED",
                "performance_eligible": eligible,
                "qualification_samples": paired_count,
                "baseline_median_ns": int(round(baseline_median)),
                "enabled_median_ns": int(round(enabled_median)),
                "local_overhead_ratio": local_overhead,
                "global_max_overhead_ratio": global_max_overhead,
                "threshold_ratio": float(args.qualification_threshold_ratio),
            }
        else:
            eligible = False
            qualification = {
                "status": "NOT_RUN",
                "method": "HOOK_RECORDING_DISABLED_VS_ENABLED_PAIRED",
                "performance_eligible": False,
                "qualification_samples": 0,
                "threshold_ratio": float(args.qualification_threshold_ratio),
            }
        from rs_sim.trace.collection import set_capture_performance_qualification
        set_capture_performance_qualification(eligible=eligible, evidence=qualification)
        flush_capture()
        report["capture_qualification"] = qualification
        report["input_contract"] = {
            "name": "COUNTER_SEEDED_GLOBAL_SAMPLE",
            "seq_length": int(args.seq_length),
            "local_micro_batch_size": int(args.micro_batch_size),
            "global_source_batch_size": int(args.global_batch_size),
            "global_input_tokens": int(args.global_batch_size) * int(args.seq_length),
            "save_input_ids": bool(args.save_input_ids),
        }
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        report["status"] = "PASS"
        report["warmup_samples_completed"] = int(args.warmup_samples)
        report["samples_completed"] = int(args.samples)
        return 0
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        return 1
    finally:
        report["finished_at_unix_ns"] = time.time_ns()
        _write_runner_report(report)
        if report.get("status") != "PASS":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)


def console_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
