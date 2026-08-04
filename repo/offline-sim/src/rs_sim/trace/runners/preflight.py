from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

from .model_support import ModelInspection, ModelSupportError, inspect_hf_model, validate_generic_text_moe


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None



def _call_compatible(callable_obj: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Call public Bridge APIs across releases without masking real failures."""
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


def _bridge_capability(model_path: str, *, trust_remote_code: bool) -> tuple[Any, dict[str, Any]]:
    try:
        from megatron.bridge import AutoBridge
    except Exception as exc:
        raise ModelSupportError(f"Megatron Bridge import failed: {type(exc).__name__}: {exc}") from exc

    report: dict[str, Any] = {
        "autobridge_class": f"{AutoBridge.__module__}.{AutoBridge.__qualname__}",
        "can_handle": None,
        "can_handle_error": None,
    }
    method = getattr(AutoBridge, "can_handle", None)
    if callable(method):
        try:
            report["can_handle"] = bool(method(model_path))
        except Exception as exc:
            report["can_handle_error"] = f"{type(exc).__name__}: {exc}"
        if report["can_handle"] is False:
            raise ModelSupportError(
                "the installed Megatron Bridge AutoBridge reports that this checkpoint architecture is unsupported"
            )

    # This is the authoritative architecture check. It loads configuration and
    # constructs a provider, but does not initialize distributed state or load
    # model weights.
    try:
        bridge = _call_compatible(
            AutoBridge.from_hf_pretrained,
            model_path,
            trust_remote_code=bool(trust_remote_code),
        )
        provider = _call_compatible(bridge.to_megatron_provider, load_weights=False)
    except Exception as exc:
        raise ModelSupportError(
            "Megatron Bridge could not construct a provider for this checkpoint: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    report["bridge_class"] = f"{type(bridge).__module__}.{type(bridge).__qualname__}"
    report["provider_class"] = f"{type(provider).__module__}.{type(provider).__qualname__}"
    return provider, report


def _capture_contract(
    *,
    require_fate_route: bool = False,
    require_compute_hooks: bool = True,
) -> dict[str, Any]:
    """Inspect the installed Megatron-Core MoE lifecycle.

    Megatron-Core <=0.14 exposed ``MoELayer.preprocess`` directly.  Core
    0.15 split that stage into ``MoELayer.router_and_preprocess`` and
    ``MoETokenDispatcher.dispatch_preprocess``.  Both lifecycles provide the
    same routing truth and are supported by the capture adapter.
    """
    try:
        moe_module = importlib.import_module("megatron.core.transformer.moe.moe_layer")
        dispatcher_module = importlib.import_module("megatron.core.transformer.moe.token_dispatcher")
    except Exception as exc:
        raise ModelSupportError(f"Megatron-Core MoE import failed: {type(exc).__name__}: {exc}") from exc

    layer_classes: list[dict[str, Any]] = []
    for _, cls in inspect.getmembers(moe_module, inspect.isclass):
        if not cls.__module__.startswith(moe_module.__name__):
            continue
        methods = [
            name
            for name in (
                "route",
                "preprocess",
                "router_and_preprocess",
                "dispatch",
                "routed_experts_compute",
                "combine",
                "forward",
            )
            if callable(getattr(cls, name, None))
        ]
        if methods:
            layer_classes.append({"class": f"{cls.__module__}.{cls.__qualname__}", "methods": methods})

    dispatcher_classes: list[dict[str, Any]] = []
    for _, cls in inspect.getmembers(dispatcher_module, inspect.isclass):
        if not cls.__module__.startswith(dispatcher_module.__name__):
            continue
        methods = [
            name
            for name in (
                "dispatch_preprocess",
                "dispatch_postprocess",
                "token_dispatch",
                "token_combine",
                "token_permutation",
                "token_unpermutation",
            )
            if callable(getattr(cls, name, None))
        ]
        if methods:
            dispatcher_classes.append({"class": f"{cls.__module__}.{cls.__qualname__}", "methods": methods})

    dispatcher_preprocess_classes = sum(
        1 for row in dispatcher_classes if "dispatch_preprocess" in row["methods"]
    )
    dispatcher_postprocess_classes = sum(
        1 for row in dispatcher_classes if "dispatch_postprocess" in row["methods"]
    )

    legacy_layers = sum(
        1
        for row in layer_classes
        if "preprocess" in row["methods"] and "forward" in row["methods"]
    )
    split_layers = sum(
        1
        for row in layer_classes
        if "router_and_preprocess" in row["methods"]
        and "forward" in row["methods"]
        and dispatcher_preprocess_classes > 0
    )
    compatible_layers = legacy_layers + split_layers
    route_compatible_layers = sum(
        1
        for row in layer_classes
        if "forward" in row["methods"]
        and (
            ("route" in row["methods"] and "preprocess" in row["methods"])
            or ("router_and_preprocess" in row["methods"] and dispatcher_preprocess_classes > 0)
        )
    )
    compute_compatible_layers = sum(
        1
        for row in layer_classes
        if "routed_experts_compute" in row["methods"] and "forward" in row["methods"]
    )

    if compatible_layers == 0:
        raise ModelSupportError(
            "installed Megatron-Core exposes neither the legacy MoELayer.preprocess lifecycle "
            "nor the Core 0.15 router_and_preprocess + dispatcher.dispatch_preprocess lifecycle"
        )
    if require_fate_route and route_compatible_layers == 0:
        raise ModelSupportError(
            "FATE_P2 requires either MoELayer.route or MoELayer.router_and_preprocess, "
            "but the installed Megatron-Core exposes neither supported lifecycle"
        )
    if require_compute_hooks and compute_compatible_layers == 0:
        raise ModelSupportError(
            "capture.local_compute requires routed_experts_compute, but the installed Megatron-Core lifecycle does not expose it"
        )
    if require_compute_hooks and dispatcher_postprocess_classes == 0:
        raise ModelSupportError(
            "capture.local_compute requires a dispatcher dispatch_postprocess hook, but none is available"
        )

    profiles: list[str] = []
    if legacy_layers:
        profiles.append("LEGACY_MOELAYER_PREPROCESS")
    if split_layers:
        profiles.append("SPLIT_ROUTER_DISPATCH")
    return {
        "supported_lifecycle_profiles": profiles,
        "compatible_moe_layer_class_count": compatible_layers,
        "legacy_preprocess_compatible_class_count": legacy_layers,
        "split_router_dispatch_compatible_class_count": split_layers,
        "fate_route_compatible_class_count": route_compatible_layers,
        "compute_compatible_moe_layer_class_count": compute_compatible_layers,
        "dispatch_preprocess_compatible_class_count": dispatcher_preprocess_classes,
        "dispatch_postprocess_compatible_class_count": dispatcher_postprocess_classes,
        "moe_layer_classes": layer_classes,
        "dispatcher_classes": dispatcher_classes,
    }


def run_megatron_model_preflight(
    *,
    model_path: str | Path,
    hf_config_path: str | Path | None,
    model_format: str,
    ep: int,
    tp: int,
    pp: int,
    dp: int,
    cp: int,
    etp: int,
    nproc_per_node: int,
    trust_remote_code: bool,
    require_cuda: bool = True,
    require_fate_route: bool = False,
    require_compute_hooks: bool = True,
) -> dict[str, Any]:
    normalized_format = str(model_format).lower()
    model_root = Path(model_path).expanduser().resolve()
    if normalized_format == "auto":
        normalized_format = "hf" if (model_root / "config.json").is_file() else "megatron"
    inspection_root = model_root if normalized_format == "hf" else Path(str(hf_config_path or "")).expanduser().resolve()
    if normalized_format == "megatron" and not hf_config_path:
        raise ModelSupportError("native Megatron checkpoints require model.hf_config_path")

    inspection: ModelInspection = inspect_hf_model(inspection_root)
    validate_generic_text_moe(inspection, ep=int(ep))
    if any(int(value) != 1 for value in (tp, pp, dp, cp)):
        raise ModelSupportError(
            "Current-P12 trace sources are EP ranks; the self-contained runner requires TP=PP=DP=CP=1"
        )
    if int(etp) <= 0 or int(ep) <= 0:
        raise ModelSupportError("EP and ETP must be positive")

    try:
        import torch
    except Exception as exc:
        raise ModelSupportError(f"PyTorch import failed: {type(exc).__name__}: {exc}") from exc
    cuda_available = bool(torch.cuda.is_available())
    local_devices = int(torch.cuda.device_count())
    if require_cuda and not cuda_available:
        raise ModelSupportError("CUDA is unavailable; official Megatron collection requires NVIDIA GPUs")
    if require_cuda and local_devices < int(nproc_per_node):
        raise ModelSupportError(
            f"visible CUDA devices={local_devices}, but launch.nproc_per_node={int(nproc_per_node)}"
        )

    bridge_path = str(inspection_root)
    _, bridge_report = _bridge_capability(bridge_path, trust_remote_code=trust_remote_code)
    capture_contract = _capture_contract(
        require_fate_route=require_fate_route,
        require_compute_hooks=require_compute_hooks,
    )
    return {
        "schema_version": "RS_SIM_MEGATRON_MODEL_PREFLIGHT",
        "status": "PASS",
        "support_contract": "INSTALLED_AUTOBRIDGE_DECODER_ONLY_TEXT_MOE",
        "model_format": normalized_format.upper(),
        "model_path": str(model_root),
        "inspection_path": str(inspection_root),
        "model_inspection": inspection.to_dict(),
        "parallel": {"ep": int(ep), "tp": int(tp), "pp": int(pp), "dp": int(dp), "cp": int(cp), "etp": int(etp)},
        "environment": {
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "visible_cuda_devices": local_devices,
            "megatron-core": _version("megatron-core"),
            "megatron-bridge": _version("megatron-bridge"),
            "transformer-engine": _version("transformer-engine"),
            "transformers": _version("transformers"),
        },
        "autobridge": bridge_report,
        "capture_contract": capture_contract,
    }


def write_preflight(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
