from __future__ import annotations


def probe_online_olmoe_adapter_support(model) -> dict[str, object]:
    del model
    return {
        "supported": False,
        "reason": "online OLMoE EP adapter is not implemented yet in the RS mainline",
    }
