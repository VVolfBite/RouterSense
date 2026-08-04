from __future__ import annotations

import os
import signal

import pytest


DEFAULT_ITEM_TIMEOUT_SECONDS = 5.0


def _timeout_seconds() -> float:
    raw = os.environ.get("RS_SIM_PYTEST_ITEM_TIMEOUT_SECONDS", str(DEFAULT_ITEM_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("RS_SIM_PYTEST_ITEM_TIMEOUT_SECONDS must be numeric") from exc
    if value <= 0:
        raise RuntimeError("RS_SIM_PYTEST_ITEM_TIMEOUT_SECONDS must be positive")
    return value


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Apply one deadline to setup, call, and teardown.

    The authoritative runner launches every item in a fresh process and also
    enforces an OS process deadline.  SIGALRM gives pytest a useful failure
    report before that outer process deadline is reached.
    """

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    timeout = _timeout_seconds()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def on_timeout(_signum, _frame):
        raise TimeoutError(
            f"pytest item exceeded {timeout:g}s across setup/call/teardown: {item.nodeid}"
        )

    signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
