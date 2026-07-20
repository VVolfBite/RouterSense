from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.host import _snapshot_value, validate_observer_mode
from rs.runtime.online.megatron_ep.runtime import RouterSenseDispatcherFacade
from rs.runtime.online.megatron_ep.observation import RouterSenseObserver


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


def test_observer_mode_validation() -> None:
    assert validate_observer_mode("off") == "off"
    assert validate_observer_mode("lightweight") == "lightweight"


def test_snapshot_value_omits_tensor_values_by_default() -> None:
    payload = _snapshot_value(torch.tensor([1, 2, 3]))
    assert payload["kind"] == "tensor"
    assert payload["numel"] == 3
    assert "values" not in payload


def test_snapshot_value_includes_tensor_values_when_explicitly_enabled() -> None:
    payload = _snapshot_value(torch.tensor([1, 2, 3]), include_tensor_values=True)
    assert payload["values"] == [1, 2, 3]
