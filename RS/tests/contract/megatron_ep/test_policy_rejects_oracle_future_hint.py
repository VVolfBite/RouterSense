from __future__ import annotations

import pytest

from integrations.megatron_ep.routersense.contracts import RouterSenseInjectionConfig
from integrations.megatron_ep.routersense.dispatcher_facade import RouterSenseDispatcherFacade, UnsupportedSchedulerMode


def test_policy_rejects_non_none_future_hint() -> None:
    with pytest.raises(UnsupportedSchedulerMode):
        RouterSenseDispatcherFacade.from_config(
            native_dispatcher=lambda *args, **kwargs: None,
            config=RouterSenseInjectionConfig(scheduler_mode="joint_shadow_p0p1", future_hint_mode="oracle"),
        )
