from __future__ import annotations

from integrations.megatron_ep.routersense.dispatcher_facade import RouterSenseDispatcherFacade
from integrations.megatron_ep.routersense.observer import RouterSenseObserver


def test_observer_records() -> None:
    observer = RouterSenseObserver()
    observer.record(phase="P0", rows=3)
    assert observer.export_rows() == [{"phase": "P0", "rows": 3}]


def test_noop_facade_passthrough() -> None:
    seen = {}

    def native_dispatcher(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"ok": True}

    facade = RouterSenseDispatcherFacade(native_dispatcher=native_dispatcher)
    result = facade.dispatch(1, x=2)
    assert result == {"ok": True}
    assert seen["args"] == (1,)
    assert seen["kwargs"] == {"x": 2}
    assert facade.facade_mode == "no_op_native_passthrough"
