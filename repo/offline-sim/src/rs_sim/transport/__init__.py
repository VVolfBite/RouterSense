"""Formal RS-SIM transport DataPlane and ControlPlane implementations."""

from .runtime.builders import (
    FormalTransportBundle,
    build_formal_transports,
    make_default_synthetic_control_profile,
    make_default_synthetic_hardware_profile,
)
from .control.channel import FormalControlPlaneTransport
from .data.channel import FormalDataPlaneTransport
from .runtime.driver import (
    AtomicBatchSubmission,
    FormalTransportRuntimeDriver,
    build_formal_transport_runtime_driver,
)
from .core.errors import (
    ReceiptStateError,
    RejectionCode,
    TransportRejection,
    UnsupportedFormalExecutionModeError,
    validate_formal_execution_mode,
)
from .observability.metrics import (
    PhysicalBusyInterval,
    PhysicalLaunchMetric,
    PhysicalMetricsView,
    PhysicalTaskMetric,
)
from .core.lifecycle import capture_process_resource_snapshot
from .config.profiles import (
    BANDWIDTH_MODE_FIXED_PER_LANE,
    BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE,
    TRANSPORT_PROFILE_SCHEMA,
    PROFILE_KIND_CALIBRATED,
    PROFILE_KIND_SYNTHETIC,
    BandwidthContentionSensitivity,
    TransportProfileBundle,
    TransportProfileProvider,
    StaticTransportProfileProvider,
    SyntheticTransportProfileSet,
    fixed_per_lane_bandwidth_sensitivity,
    load_transport_profile_bundle_json,
    make_calibrated_transport_profile_bundle,
    make_transport_profile_bundle,
    make_synthetic_profile_sensitivity_set,
    visible_fabric_equal_share_sensitivity,
    write_transport_profile_bundle_json,
)

__all__ = [
    "AtomicBatchSubmission",
    "BANDWIDTH_MODE_FIXED_PER_LANE",
    "BANDWIDTH_MODE_VISIBLE_FABRIC_EQUAL_SHARE",
    "BandwidthContentionSensitivity",
    "FormalControlPlaneTransport",
    "FormalDataPlaneTransport",
    "FormalTransportRuntimeDriver",
    "FormalTransportBundle",
    "TRANSPORT_PROFILE_SCHEMA",
    "TransportProfileBundle",
    "TransportProfileProvider",
    "PROFILE_KIND_CALIBRATED",
    "PROFILE_KIND_SYNTHETIC",
    "PhysicalBusyInterval",
    "PhysicalLaunchMetric",
    "PhysicalMetricsView",
    "PhysicalTaskMetric",
    "ReceiptStateError",
    "RejectionCode",
    "StaticTransportProfileProvider",
    "SyntheticTransportProfileSet",
    "TransportRejection",
    "UnsupportedFormalExecutionModeError",
    "build_formal_transport_runtime_driver",
    "build_formal_transports",
    "capture_process_resource_snapshot",
    "fixed_per_lane_bandwidth_sensitivity",
    "load_transport_profile_bundle_json",
    "make_calibrated_transport_profile_bundle",
    "make_default_synthetic_control_profile",
    "make_default_synthetic_hardware_profile",
    "make_transport_profile_bundle",
    "make_synthetic_profile_sensitivity_set",
    "visible_fabric_equal_share_sensitivity",
    "write_transport_profile_bundle_json",
    "validate_formal_execution_mode",
]
