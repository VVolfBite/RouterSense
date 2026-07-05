from __future__ import annotations

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.runtime import RouterSenseDispatcherFacade


def test_shadow_plan_mode_does_not_mutate_native_dispatch() -> None:
    seen = {"count": 0}

    def native_dispatcher(*args, **kwargs):
        seen["count"] += 1
        return {"marker": "native", "args": args, "kwargs": kwargs}

    facade = RouterSenseDispatcherFacade.from_config(
        native_dispatcher=native_dispatcher,
        config=RouterSenseInjectionConfig(scheduler_mode="joint_shadow_p0p1", future_hint_mode="none"),
    )
    result = facade.dispatch("tensor", split_sizes=(1, 2))
    assert result["marker"] == "native"
    assert result["args"] == ("tensor",)
    assert result["kwargs"] == {"split_sizes": (1, 2)}
    assert seen["count"] == 1
