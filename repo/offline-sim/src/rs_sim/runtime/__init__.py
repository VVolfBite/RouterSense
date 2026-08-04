"""Current-P12 integration layer."""

from .assembly.bindings import (
    SchemaEdgeKeyFactory,
    SchemaExpectationFactory,
    SchemaPermitFactory,
    SchedulingStack,
    build_scheduling_stack,
    make_phase_semantics,
    make_schema_adapter,
    shared_binding_digest,
)
from .adapters.kernel import BackendKernelBridge
from .adapters.backend import BackendControlPlaneAdapter
from .adapters.scheduler import (
    SchedulerBackendCompletionAdapter,
    SchedulerResourceReleaseAdapter,
)
from .core.engine import (
    CurrentP12WindowRecord,
    FormalIntegrationRuntime,
    build_current_p12_integration_runtime,
)
from .adapters.trace import TraceWindowKeys, keys_for_trace_window, payload_bytes_for_phase
from .config.profiles import (
    RUNTIME_PROFILE_SCHEMA,
    RuntimeProfileBundle,
    load_runtime_profile_bundle_json,
    make_default_synthetic_runtime_profile,
    make_runtime_profile_bundle,
    write_runtime_profile_bundle_json,
)

__all__ = [
    "BackendControlPlaneAdapter",
    "RUNTIME_PROFILE_SCHEMA",
    "RuntimeProfileBundle",
    "CurrentP12WindowRecord",
    "FormalIntegrationRuntime",
    "BackendKernelBridge",
    "SchemaEdgeKeyFactory",
    "SchemaExpectationFactory",
    "SchemaPermitFactory",
    "SchedulerBackendCompletionAdapter",
    "SchedulerResourceReleaseAdapter",
    "SchedulingStack",
    "TraceWindowKeys",
    "build_current_p12_integration_runtime",
    "build_scheduling_stack",
    "keys_for_trace_window",
    "load_runtime_profile_bundle_json",
    "make_default_synthetic_runtime_profile",
    "make_runtime_profile_bundle",
    "make_phase_semantics",
    "make_schema_adapter",
    "payload_bytes_for_phase",
    "shared_binding_digest",
    "write_runtime_profile_bundle_json",
]
