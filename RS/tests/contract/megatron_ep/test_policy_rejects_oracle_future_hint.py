from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.runtime import RouterSenseDispatcherFacade, UnsupportedSchedulerMode


def test_policy_rejects_non_none_future_hint() -> None:
    with pytest.raises(UnsupportedSchedulerMode):
        RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(scheduler_mode="joint_shadow_p0p1", future_hint_mode="oracle"),
        )
