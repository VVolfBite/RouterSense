#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest


class TimingPlugin:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.phase_durations: dict[str, float] = {}
        self.phase_outcomes: dict[str, str] = {}
        self.longrepr: str | None = None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.nodeid != self.nodeid or report.when not in {"setup", "call", "teardown"}:
            return
        self.phase_durations[report.when] = float(report.duration)
        self.phase_outcomes[report.when] = str(report.outcome)
        if report.failed and self.longrepr is None:
            self.longrepr = str(report.longrepr)

    def payload(self, return_code: int) -> dict[str, Any]:
        return {
            "schema_version": "RS_SIM_PYTEST_ITEM_RESULT",
            "nodeid": self.nodeid,
            "return_code": int(return_code),
            "phase_durations_seconds": self.phase_durations,
            "phase_outcomes": self.phase_outcomes,
            "item_duration_seconds": round(sum(self.phase_durations.values()), 9),
            "passed": int(return_code) == 0,
            "longrepr": self.longrepr,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exactly one pytest item and force process exit")
    parser.add_argument("nodeid")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)

    plugin = TimingPlugin(args.nodeid)
    code = int(pytest.main(["-q", args.nodeid], plugins=[plugin]))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(plugin.payload(code), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(code))
