from __future__ import annotations

import importlib


def test_formal_import_smoke_under_pythonpath_src() -> None:
    modules = [
        "rs.scheduling",
        "rs.scheduling.matching",
        "rs.runtime.offline",
        "rs.runtime.online.megatron_ep",
        "rs.runtime.online.megatron_ep.phase",
        "rs.runtime.online.megatron_ep.runtime",
    ]
    for name in modules:
        importlib.import_module(name)
