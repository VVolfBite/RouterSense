from __future__ import annotations

import hashlib
from typing import Any

from rs_sim.contracts.digest import stable_json_dumps


def semantic_ordinal(*parts: Any) -> int:
    encoded = stable_json_dumps(tuple(parts)).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16)


__all__ = ["semantic_ordinal"]
