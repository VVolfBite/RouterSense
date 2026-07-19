from __future__ import annotations

from experiments.paper.hiding_evaluation import ready_margin_us


def test_ready_margin_computation() -> None:
    assert ready_margin_us(target_dispatch_started_ns=10_000, store_ready_ns=4_000) == 6.0
    assert ready_margin_us(target_dispatch_started_ns=None, store_ready_ns=4_000) is None
