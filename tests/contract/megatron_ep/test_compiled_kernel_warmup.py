from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.target_planning import planner_service as service_module
from rs.runtime.online.megatron_ep.target_planning import warmup as warmup_module
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerService
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore
from rs.scheduling.p012_future._kernel import future as future_kernel


def _reset_process_warmup_state() -> None:
    warmup_module._COMPILED_KERNEL_WARMUP_STATE.update(  # noqa: SLF001
        status="not_started",
        duration_us=0.0,
        planner_ids=(),
        error=None,
    )


def test_process_warmup_runs_event_global_and_binders_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _Planner:
        def __init__(self, planner_id: str) -> None:
            self.planner_id = planner_id

        def plan(self, request):
            calls.append(f"plan:{self.planner_id}:{request.identity.request_id}")
            return object()

    monkeypatch.setattr(
        warmup_module.PlannerRegistry,
        "create",
        lambda planner_id, _config, usage=None: _Planner(planner_id),
    )
    monkeypatch.setattr(warmup_module, "validate_window_plan_for_request", lambda _plan, _request: None)
    monkeypatch.setattr(future_kernel, "warmup_future_bind_kernel", lambda: calls.append("binders"))
    _reset_process_warmup_state()

    first = warmup_module.ensure_compiled_kernel_warmup()  # noqa: SLF001
    second = warmup_module.ensure_compiled_kernel_warmup()  # noqa: SLF001

    assert first["status"] == "passed"
    assert second["status"] == "passed"
    assert tuple(first["planner_ids"]) == (
        "future:p012:joint:event:rscf",
        "future:p012:joint:global:rscf",
    )
    assert calls == [
        "plan:future:p012:joint:event:rscf:compiled-kernel-warmup",
        "plan:future:p012:joint:global:rscf:compiled-kernel-warmup",
        "binders",
    ]
    _reset_process_warmup_state()


def test_start_records_successful_warmup_before_accepting_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service_module,
        "ensure_compiled_kernel_warmup",
        lambda: {
            "status": "passed",
            "duration_us": 123.0,
            "planner_ids": (
                "future:p012:joint:event:rscf",
                "future:p012:joint:global:rscf",
            ),
            "error": None,
        },
    )
    service = TargetLayerPlannerService(store=TargetPlanStore())
    service.start()
    try:
        assert service.is_alive() is True
        warmup = service.timeline()[-1]
        assert warmup["event"] == "compiled_kernel_warmup"
        assert warmup["status"] == "passed"
        assert warmup["duration_us"] == 123.0
    finally:
        service.close()


def test_start_fails_closed_when_warmup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail():
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(service_module, "ensure_compiled_kernel_warmup", _fail)
    service = TargetLayerPlannerService(store=TargetPlanStore())

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        service.start()

    assert service.is_alive() is False
    assert service._closed is True  # noqa: SLF001
    assert isinstance(service._last_error, RuntimeError)  # noqa: SLF001
    warmup = service.timeline()[-1]
    assert warmup["event"] == "compiled_kernel_warmup"
    assert warmup["status"] == "failed"
