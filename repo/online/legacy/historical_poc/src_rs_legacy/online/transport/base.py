from __future__ import annotations


def transport_backend_realizes_matching(transport_backend: str) -> bool:
    return transport_backend == "scheduled_p2p"
