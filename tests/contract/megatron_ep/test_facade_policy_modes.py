from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.runtime import RouterSenseDispatcherFacade, UnsupportedSchedulerMode


def test_facade_passthrough_for_supported_modes() -> None:
    seen = {}

    def native_dispatcher(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"ok": True}

    for mode in ("disabled", "native_order", "joint_shadow_p0p1"):
        facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=native_dispatcher,
            config=RouterSenseInjectionConfig(scheduler_mode=mode, future_hint_mode="none"),
        )
        assert facade.dispatch(1, x=2) == {"ok": True}
        assert facade.scheduler_mode == mode
        assert facade.future_hint_mode == "none"
        assert facade.control_mode == "default_continue"
        assert seen["args"] == (1,)
        assert seen["kwargs"] == {"x": 2}


def test_facade_accepts_supported_control_modes() -> None:
    for control_mode in ("default_continue", "sync_before_phase"):
        facade = RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(
                scheduler_mode="native_order",
                future_hint_mode="none",
                control_mode=control_mode,
            ),
        )
        assert facade.control_mode == control_mode


def test_facade_rejects_unsupported_mode() -> None:
    with pytest.raises(UnsupportedSchedulerMode):
        RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(scheduler_mode="oracle", future_hint_mode="none"),
        )


def test_facade_rejects_non_none_future_hint() -> None:
    with pytest.raises(UnsupportedSchedulerMode):
        RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(scheduler_mode="native_order", future_hint_mode="predicted"),
        )


def test_facade_rejects_unsupported_control_mode() -> None:
    with pytest.raises(UnsupportedSchedulerMode):
        RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(
                scheduler_mode="native_order",
                future_hint_mode="none",
                control_mode="priority_override",
            ),
        )
