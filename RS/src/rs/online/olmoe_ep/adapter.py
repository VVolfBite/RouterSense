from __future__ import annotations

from .runtime import feature_probe_online_olmoe_runtime

def probe_online_olmoe_adapter_support(model) -> dict[str, object]:
    return feature_probe_online_olmoe_runtime(model)
