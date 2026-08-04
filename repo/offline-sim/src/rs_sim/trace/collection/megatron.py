"""Automatic Megatron-Core MoE instrumentation.

The adapter supports both the legacy ``MoELayer.preprocess`` lifecycle and
Megatron-Core 0.15's split ``router_and_preprocess`` /
``dispatcher.dispatch_preprocess`` lifecycle.  Routing truth is recorded at
the last local stage that still carries the exact post-router routing map.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from types import ModuleType
from typing import Any, Callable

from .api import current_capture_session

_PATCHED = False


def _wrap_method(cls: type, name: str, factory: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
    original = getattr(cls, name, None)
    if not callable(original) or getattr(original, "__rs_sim.trace.collection_wrapped__", False):
        return False
    wrapped = factory(original)
    setattr(wrapped, "__rs_sim.trace.collection_wrapped__", True)
    setattr(cls, name, wrapped)
    return True


def _layer_id(module: Any) -> int:
    session = current_capture_session(required=True)
    assert session is not None
    return session.layer_id_for(module)


def _set_dispatcher_owner(moe_layer: Any) -> Any | None:
    dispatcher = getattr(moe_layer, "token_dispatcher", None)
    if dispatcher is not None:
        try:
            setattr(dispatcher, "_rs_sim_moe_owner", moe_layer)
        except Exception:
            pass
    router = getattr(moe_layer, "router", None)
    if router is not None:
        try:
            setattr(router, "_rs_sim_moe_owner", moe_layer)
        except Exception:
            pass
    return dispatcher


def _start_router_stage(moe_layer: Any, hidden_states: Any) -> tuple[Any, int, int]:
    session = current_capture_session(required=True)
    assert session is not None
    layer = _layer_id(moe_layer)
    step = session.invocation_step(layer)
    session.close_previous_moe_to_router(next_layer_id=layer, decode_step=step)
    if session.fate_enabled:
        session.resolve_fate_for_target(
            target_module=moe_layer, target_layer_id=layer, decode_step=step
        )
        if hidden_states is None:
            raise RuntimeError("FATE_P2 capture could not locate router hidden_states")
        session.capture_fate_gate_input(
            layer_id=layer, hidden_states=hidden_states, decode_step=step
        )
    session.start_interval("router_and_pack_ns", layer_id=layer, decode_step=step)
    return session, layer, step


def _record_routing(
    *,
    owner: Any,
    dispatcher: Any,
    routing_map: Any,
    probs: Any,
    adapter_stage: str,
) -> None:
    session = current_capture_session(required=True)
    assert session is not None
    layer = _layer_id(owner)
    step = session.invocation_step(layer)
    local_experts = getattr(dispatcher, "local_expert_indices", None)
    drop_and_pad = bool(
        getattr(dispatcher, "drop_and_pad", False)
        or getattr(getattr(dispatcher, "config", None), "moe_pad_expert_input_to_capacity", False)
    )
    session.record_routing(
        layer_id=layer,
        routing_map=routing_map,
        probs=probs,
        local_expert_indices=local_experts,
        drop_and_pad=drop_and_pad,
        metadata={
            "adapter": "MEGATRON_CORE_AUTO",
            "adapter_stage": adapter_stage,
            "moe_layer_class": f"{type(owner).__module__}.{type(owner).__qualname__}",
            "dispatcher_class": f"{type(dispatcher).__module__}.{type(dispatcher).__qualname__}",
        },
        decode_step=step,
    )


def _patch_moe_layer_class(cls: type) -> int:
    patched = 0

    def route_factory(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states", kwargs.get("hidden_states_input"))
            _set_dispatcher_owner(self)
            _start_router_stage(self, hidden_states)
            return original(self, *args, **kwargs)
        return wrapped

    def preprocess_factory(original):
        @functools.wraps(original)
        def wrapped(self, hidden_states, probs, routing_map, *args, **kwargs):
            session = current_capture_session(required=True)
            assert session is not None
            layer = _layer_id(self)
            step = session.invocation_step(layer)
            dispatcher = _set_dispatcher_owner(self)
            _record_routing(
                owner=self,
                dispatcher=dispatcher if dispatcher is not None else self,
                routing_map=routing_map,
                probs=probs,
                adapter_stage="MOELAYER_PREPROCESS",
            )
            result = original(self, hidden_states, probs, routing_map, *args, **kwargs)
            session.end_interval("router_and_pack_ns", layer_id=layer, decode_step=step)
            session.invocation_step(layer, advance=True)
            return result
        return wrapped

    def router_and_preprocess_factory(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            session = current_capture_session(required=True)
            assert session is not None
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            dispatcher = _set_dispatcher_owner(self)
            _, layer, step = _start_router_stage(self, hidden_states)
            result = original(self, *args, **kwargs)
            # Routing itself is recorded by dispatcher.dispatch_preprocess,
            # which receives the exact routing_map in Megatron-Core 0.15.
            session.end_interval("router_and_pack_ns", layer_id=layer, decode_step=step)
            session.invocation_step(layer, advance=True)
            return result
        return wrapped

    def routed_factory(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            session = current_capture_session(required=True)
            assert session is not None
            layer = _layer_id(self)
            step = session.completed_invocation_step(layer)
            session.start_interval("routed_experts_compute_total_ns", layer_id=layer, decode_step=step)
            try:
                return original(self, *args, **kwargs)
            finally:
                session.end_interval("routed_experts_compute_total_ns", layer_id=layer, decode_step=step)
        return wrapped

    def forward_factory(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            session = current_capture_session(required=True)
            assert session is not None
            layer = _layer_id(self)
            before = session.invocation_step(layer)
            result = original(self, *args, **kwargs)
            step = max(before, session.completed_invocation_step(layer))
            session.mark_moe_output_ready(layer_id=layer, decode_step=step)
            return result
        return wrapped

    patched += int(_wrap_method(cls, "route", route_factory))
    patched += int(_wrap_method(cls, "preprocess", preprocess_factory))
    patched += int(_wrap_method(cls, "router_and_preprocess", router_and_preprocess_factory))
    patched += int(_wrap_method(cls, "routed_experts_compute", routed_factory))
    patched += int(_wrap_method(cls, "forward", forward_factory))
    return patched


def _extract_dispatch_preprocess_arguments(original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]):
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(None, *args, **kwargs)
        values = bound.arguments
        routing_map = values.get("routing_map")
        probs = values.get("probs")
    except (TypeError, ValueError):
        routing_map = kwargs.get("routing_map")
        probs = kwargs.get("probs")
    if routing_map is None and len(args) >= 2:
        routing_map = args[1]
    if probs is None and len(args) >= 3:
        probs = args[2]
    return routing_map, probs


def _patch_dispatcher_classes(module: ModuleType) -> int:
    patched = 0
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not cls.__module__.startswith(module.__name__):
            continue

        def preprocess_factory(original):
            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                owner = getattr(self, "_rs_sim_moe_owner", None)
                # Legacy MoELayer.preprocess records the same truth itself.  The
                # dispatcher hook is authoritative only for the split 0.15 path.
                if owner is not None and not callable(getattr(type(owner), "preprocess", None)):
                    routing_map, probs = _extract_dispatch_preprocess_arguments(original, args, kwargs)
                    if routing_map is None:
                        raise RuntimeError("Megatron dispatch_preprocess hook could not locate routing_map")
                    _record_routing(
                        owner=owner,
                        dispatcher=self,
                        routing_map=routing_map,
                        probs=probs,
                        adapter_stage="DISPATCHER_DISPATCH_PREPROCESS",
                    )
                return original(self, *args, **kwargs)
            return wrapped

        def postprocess_factory(original):
            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                session = current_capture_session(required=True)
                assert session is not None
                owner = getattr(self, "_rs_sim_moe_owner", self)
                layer = session.layer_id_for(owner)
                step = session.completed_invocation_step(layer)
                session.start_interval("dispatch_local_postprocess_ns", layer_id=layer, decode_step=step)
                try:
                    return original(self, *args, **kwargs)
                finally:
                    session.end_interval("dispatch_local_postprocess_ns", layer_id=layer, decode_step=step)
            return wrapped

        patched += int(_wrap_method(cls, "dispatch_preprocess", preprocess_factory))
        patched += int(_wrap_method(cls, "dispatch_postprocess", postprocess_factory))
    return patched


def install_megatron_auto_capture() -> dict[str, Any]:
    global _PATCHED
    if _PATCHED:
        return {"status": "ALREADY_INSTALLED"}
    session = current_capture_session(required=True)
    assert session is not None
    try:
        moe_layer = importlib.import_module("megatron.core.transformer.moe.moe_layer")
        token_dispatcher = importlib.import_module("megatron.core.transformer.moe.token_dispatcher")
    except Exception as exc:
        session.warning("MEGATRON_IMPORT_FAILED", str(exc), exception_type=type(exc).__name__)
        if session.strict:
            raise
        return {"status": "IMPORT_FAILED", "message": str(exc)}

    count = 0
    profiles: set[str] = set()
    for _, cls in inspect.getmembers(moe_layer, inspect.isclass):
        if not cls.__module__.startswith(moe_layer.__name__):
            continue
        if callable(getattr(cls, "preprocess", None)):
            profiles.add("LEGACY_MOELAYER_PREPROCESS")
        if callable(getattr(cls, "router_and_preprocess", None)):
            profiles.add("SPLIT_ROUTER_DISPATCH")
        if callable(getattr(cls, "forward", None)) and (
            callable(getattr(cls, "preprocess", None))
            or callable(getattr(cls, "router_and_preprocess", None))
        ):
            count += _patch_moe_layer_class(cls)
    count += _patch_dispatcher_classes(token_dispatcher)
    if count == 0:
        message = "no compatible Megatron MoE lifecycle methods were found"
        session.warning("MEGATRON_PATCH_EMPTY", message)
        if session.strict:
            raise RuntimeError(message)
    _PATCHED = True
    return {
        "status": "INSTALLED",
        "wrapped_method_count": count,
        "lifecycle_profiles": sorted(profiles),
    }
