from __future__ import annotations

from .contracts import HidingTimelineRecord, RecordMetadata


def ready_margin_us(*, target_dispatch_started_ns: int | None, store_ready_ns: int | None) -> float | None:
    if target_dispatch_started_ns is None or store_ready_ns is None:
        return None
    return (int(target_dispatch_started_ns) - int(store_ready_ns)) / 1_000.0


def evaluate_hiding_gap(*, metadata: RecordMetadata, model_id: str) -> dict[str, object]:
    record = HidingTimelineRecord(
        model_id=str(model_id),
        prompt_id="not_measured",
        layer_id="not_measured",
        current_dispatch_visible_ns=None,
        observation_ready_ns=None,
        prediction_ready_ns=None,
        planning_ready_ns=None,
        publication_ready_ns=None,
        store_ready_ns=None,
        target_dispatch_started_ns=None,
        plan_consumed_ns=None,
        available_window_us=None,
        total_prepare_us=None,
        ready_margin_us=ready_margin_us(target_dispatch_started_ns=None, store_ready_ns=None),
        plan_source="NOT_SEPARATELY_MEASURABLE",
        fallback_count=0,
        metadata=metadata,
        status="PARTIAL",
    )
    return {
        "records": [record.to_dict()],
        "status": "TEST_HARNESS_GAP",
        "reason": "no stable public timeline API for current_dispatch_visible/store_ready extraction without modifying frozen runtime",
    }
