"""Independent RouterSense trace capture and collect→fixture→simulate pipeline.

The package deliberately has no hard Torch or Megatron dependency.  The
Megatron adapter imports those libraries only inside an instrumented model
process.  The simulator and trace finalizer remain usable on CPU-only hosts.
"""

from .api import (
    capture_routing,
    current_capture_session,
    flush_capture,
    finish_capture_sample,
    set_capture_enabled,
    set_capture_performance_qualification,
    set_capture_context,
)
from .config import CaptureConfigError, load_pipeline_config

__all__ = [
    "CaptureConfigError",
    "capture_routing",
    "current_capture_session",
    "flush_capture",
    "finish_capture_sample",
    "set_capture_enabled",
    "set_capture_performance_qualification",
    "load_pipeline_config",
    "set_capture_context",
]
