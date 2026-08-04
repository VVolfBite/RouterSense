from __future__ import annotations

import importlib
import sys


def test_cli_import_does_not_eagerly_import_collect_or_experiment() -> None:
    sys.modules.pop("rs_sim.app.cli", None)
    sys.modules.pop("rs_sim.app.collect", None)
    sys.modules.pop("rs_sim.app.experiment", None)
    importlib.import_module("rs_sim.app.cli")
    assert "rs_sim.app.collect" not in sys.modules
    assert "rs_sim.app.experiment" not in sys.modules
