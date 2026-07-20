"""Deprecated module name retained only inside the runtime source tree.

New code imports :mod:`window_shadow`; no legacy scheduling implementation is
used here.
"""
from .window_shadow import build_window_shadow, classify_flow, executable_now
build_pending_window_shadow = build_window_shadow
__all__=["build_window_shadow","build_pending_window_shadow","classify_flow","executable_now"]
