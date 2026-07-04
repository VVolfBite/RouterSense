#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.online.olmoe_ep.runtime import require_online_native_ep_runtime


if __name__ == "__main__":
    require_online_native_ep_runtime()
