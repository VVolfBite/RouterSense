from __future__ import annotations

import json
from typing import Mapping

from rs.core.contracts.result import ResultBundle
from rs.core.contracts.trace import ReferenceTraceBundle


class EvidenceSerializer:
    def serialize_trace(self, bundle: ReferenceTraceBundle) -> str:
        return json.dumps(bundle.to_dict(), ensure_ascii=True, sort_keys=True)

    def serialize_result(self, bundle: ResultBundle) -> str:
        return json.dumps(bundle.to_dict(), ensure_ascii=True, sort_keys=True)

    def deserialize_trace(self, payload: str) -> ReferenceTraceBundle:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("serialized trace payload must decode to a mapping")
        return ReferenceTraceBundle.from_dict(data)

    def deserialize_result(self, payload: str) -> ResultBundle:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("serialized result payload must decode to a mapping")
        return ResultBundle.from_dict(data)
