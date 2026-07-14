from __future__ import annotations

import json
from typing import Any, Mapping

from rs.core.contracts.result import ResultBundle
from rs.core.contracts.trace import ReferenceTraceBundle


class EvidenceSerializer:
    def serialize_trace(self, bundle: ReferenceTraceBundle) -> str:
        return json.dumps(bundle.to_dict(), ensure_ascii=True, sort_keys=True)

    def serialize_result(self, bundle: ResultBundle) -> str:
        return json.dumps(bundle.to_dict(), ensure_ascii=True, sort_keys=True)

    def deserialize_trace(self, payload: str) -> dict[str, Any]:
        return dict(json.loads(payload))

    def deserialize_result(self, payload: str) -> dict[str, Any]:
        return dict(json.loads(payload))
