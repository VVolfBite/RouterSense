from __future__ import annotations

"""Canonicalize semantically identical trace serialization labels at I/O.

This is an input-boundary normalization only.  It never changes routing,
payload bytes, compute timing, mapping, FATE metadata, or window identity.
Unknown labels fail closed.
"""

import copy
from typing import Any

from ..schema.constants import (
    DESCRIPTOR_METADATA_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    RECEIVER_MODEL,
    TRACE_SCHEMA_VERSION,
)

_FIXTURE_LABELS = {FIXTURE_SCHEMA_VERSION, "RS_SIM_FIXTURE_V4_1_1_R2_2"}
_TRACE_LABELS = {TRACE_SCHEMA_VERSION, "RS_SIM_TRACE_V4_1_1_R2_2"}
_PAYLOAD_LABELS = {PAYLOAD_SCHEMA_VERSION, "RS_SIM_PAYLOAD_V4_1_1_R2_2"}
_DESCRIPTOR_LABELS = {
    DESCRIPTOR_METADATA_SCHEMA_VERSION,
    "RS_SIM_DESCRIPTOR_METADATA_V4_1_1_R2_2",
}
_RECEIVER_LABELS = {RECEIVER_MODEL, "RECEIVER_DECOUPLED_P12_V1"}
_PADDING_LABELS = {
    "NONE": "NONE",
    "EDGE_TOTAL_ALIGN_UP": "EDGE_TOTAL_ALIGN_UP",
    "EDGE_TOTAL_ALIGN_UP_V1": "EDGE_TOTAL_ALIGN_UP",
}


def _require_known(value: str, allowed: set[str], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"unsupported {field}: {value!r}")


def canonicalize_fixture_serialization(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = copy.deepcopy(data)
    changed = False

    fixture_label = str(normalized.get("schema_version", FIXTURE_SCHEMA_VERSION))
    _require_known(fixture_label, _FIXTURE_LABELS, "fixture schema")
    if fixture_label != FIXTURE_SCHEMA_VERSION:
        normalized["schema_version"] = FIXTURE_SCHEMA_VERSION
        changed = True

    initial = normalized.setdefault("initial_state", {})
    receiver_label = str(initial.get("receiver_model", RECEIVER_MODEL))
    _require_known(receiver_label, _RECEIVER_LABELS, "receiver model")
    if receiver_label != RECEIVER_MODEL:
        initial["receiver_model"] = RECEIVER_MODEL
        changed = True

    windows = normalized.get("windows", ())
    if not isinstance(windows, list):
        raise ValueError("fixture windows must be a list")
    for window in windows:
        trace_label = str(window.get("schema_version", TRACE_SCHEMA_VERSION))
        _require_known(trace_label, _TRACE_LABELS, "trace window schema")
        if trace_label != TRACE_SCHEMA_VERSION:
            window["schema_version"] = TRACE_SCHEMA_VERSION
            changed = True
        for key in ("dispatch_payload_spec", "combine_payload_spec"):
            payload = window[key]
            payload_label = str(payload.get("schema_version", PAYLOAD_SCHEMA_VERSION))
            _require_known(payload_label, _PAYLOAD_LABELS, "payload schema")
            if payload_label != PAYLOAD_SCHEMA_VERSION:
                payload["schema_version"] = PAYLOAD_SCHEMA_VERSION
                changed = True
            raw_padding = str(payload.get("padding_rule", ""))
            if raw_padding not in _PADDING_LABELS:
                raise ValueError(f"unsupported payload padding rule: {raw_padding!r}")
            canonical_padding = _PADDING_LABELS[raw_padding]
            if canonical_padding != raw_padding:
                payload["padding_rule"] = canonical_padding
                changed = True
        descriptor = window["descriptor_metadata_spec"]
        descriptor_label = str(
            descriptor.get("schema_version", DESCRIPTOR_METADATA_SCHEMA_VERSION)
        )
        _require_known(descriptor_label, _DESCRIPTOR_LABELS, "descriptor schema")
        if descriptor_label != DESCRIPTOR_METADATA_SCHEMA_VERSION:
            descriptor["schema_version"] = DESCRIPTOR_METADATA_SCHEMA_VERSION
            changed = True

    return normalized, changed


__all__ = ["canonicalize_fixture_serialization"]
