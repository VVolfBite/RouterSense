from __future__ import annotations


def require_online_native_ep_runtime() -> None:
    raise NotImplementedError(
        "online_native_a2a_ep is not implemented yet; use offline router_prediction "
        "or legacy trace replay only for the current RS mainline"
    )
