#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online.transport.scheduled_p2p import require_scheduled_p2p_backend


if __name__ == "__main__":
    require_scheduled_p2p_backend()
