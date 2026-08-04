"""Stable constants for trace capture and fixture validation."""

TRACE_SCHEMA_VERSION = "RS_SIM_TRACE"
FIXTURE_SCHEMA_VERSION = "RS_SIM_FIXTURE"
PAYLOAD_SCHEMA_VERSION = "RS_SIM_PAYLOAD"
DESCRIPTOR_METADATA_SCHEMA_VERSION = "RS_SIM_DESCRIPTOR_METADATA"
THREE_LINE_PROFILE_SCHEMA_VERSION = "RS_SIM_THREE_LINE_PROFILE"

RECEIVER_MODEL = "RECEIVER_DECOUPLED_P12"
DESCRIPTOR_PAYLOAD_RELATION = "SIMULTANEOUS_AT_LOCAL_PATH_COMPLETE"

PHASE_DISPATCH = "DISPATCH"
PHASE_COMBINE = "COMBINE"
VALID_PHASE_KINDS = frozenset({PHASE_DISPATCH, PHASE_COMBINE})

PADDING_NONE = "NONE"
PADDING_EDGE_TOTAL_ALIGN_UP = "EDGE_TOTAL_ALIGN_UP"
VALID_PADDING_RULES = frozenset({PADDING_NONE, PADDING_EDGE_TOTAL_ALIGN_UP})

VALID_DATASET_SPLITS = frozenset({"train", "validation", "test"})
VALID_DATASET_PURPOSES = frozenset({"training", "tuning", "final_evaluation", "contract_validation"})
ALLOWED_SPLITS_BY_PURPOSE = {
    "training": frozenset({"train"}),
    "tuning": frozenset({"validation"}),
    "final_evaluation": frozenset({"test"}),
    "contract_validation": VALID_DATASET_SPLITS,
}

FORBIDDEN_COMPUTE_COMPONENTS = frozenset({
    "network_wait", "collective_wait", "scheduler_wait", "receiver_wait",
    "transport_wait", "barrier_wait", "old_hook_absolute_time",
})
REQUIRED_EXCLUDED_COMPUTE_COMPONENTS = tuple(sorted(FORBIDDEN_COMPUTE_COMPONENTS))

DATA_PLANE_ACCOUNTING = "DATA_PLANE"
CONTROL_PLANE_ACCOUNTING = "CONTROL_PLANE"

VALID_PROFILE_PLANES = frozenset({"RECEIVER", "CONTROL_PLANE", "DATA_PLANE"})
PROFILE_COMPONENTS_BY_PLANE = {
    "RECEIVER": frozenset({"receiver_posting_service", "receiver_drain"}),
    "CONTROL_PLANE": frozenset({"descriptor_delivery"}),
    "DATA_PLANE": frozenset({"payload_transfer"}),
}
VALID_PROFILE_SOURCE_KINDS = frozenset({"synthetic_format_example", "measured"})

LINE_PREDICTION = "PredictionLine"
LINE_CONTROL = "ControlLine"
LINE_EXECUTION_BINDING = "ExecutionBindingLine"
VALID_SERVICE_LINES = frozenset({LINE_PREDICTION, LINE_CONTROL, LINE_EXECUTION_BINDING})
VALID_LINE_PROFILE_SOURCE_KINDS = frozenset({"synthetic", "measured", "calibrated"})
