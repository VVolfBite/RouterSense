"""Unified Current-P12 scheduler.

The scheduler exposes one path:
algorithm core -> scope decorator -> planning decorator -> live plan -> executor.
No historical horizon, profile, or policy-selection APIs are exported.
"""

from .decorators.composition import (
    REGISTERED_CORE_IDS,
    SchedulingAlgorithm,
    parse_algorithm_expression,
)
from .decorators.planning_gate import (
    PlanningDecision,
    PlanningGate,
    PlanningMode,
    PlanningTrigger,
)
from .execution.authority import (
    AuthoritySupersessionRecord,
    PhaseAuthorityManager,
    WindowAuthorityProposal,
)
from .execution.binder import PlanBinder, PreparedOrderTemplate, PreparedSlot
from .execution.compiler import (
    BatchCompiler,
    BatchValidator,
    ExecutionStabilizer,
    FormalTransportResourceAdapter,
)
from .execution.completion import (
    PhaseCompletionRecord,
    WindowCompletionRecord,
    WindowCompletionTracker,
)
from .execution.controller import SchedulingController
from .execution.lines import PlanningCostModel, ServiceLineMetrics, ThreeLineServices
from .execution.live import (
    LiveCompileSelection,
    LiveFairnessInputs,
    LiveObservationResult,
    LivePlanActivation,
    LivePolicySession,
    LivePolicySpec,
    PreparedLiveActivation,
    ReleaseMode,
    SchedulerWindow,
    build_live_policy_session,
    canonical_order_only_wave_membership_digest,
    current_p12_phase_keys,
)
from .execution.runtime_adapter import (
    CoalescedObservationBatch,
    FormalSchedulingRuntimeAdapter,
    FormalSchedulingRuntimeMetrics,
    GlobalClosureTruth,
    ObservationEnvelope,
    PhaseObservationAccumulator,
    PlanningPipelineJob,
    RuntimeActivationEvidence,
)
from .execution.state import TaskRuntimeIndex
from .execution.window_arbiter import (
    PhaseFrontier,
    PrefixWindowArbiter,
    ReleaseFrontierWindowArbiter,
    WindowArbiter,
    WindowArbitrationContext,
    WindowArbitrationDecision,
    make_window_decision,
    validate_window_decision,
)
from .metrics.reporting import (
    FormalRuntimeRecord,
    PairedBootstrapResult,
    PairedInstanceKey,
    Provenance,
    RunStatus,
    make_formal_runtime_record,
    make_paired_key,
    paired_bootstrap_formal_runtime_records,
    validate_anchor_local_formal_runtime_records,
)
from .planning.catalogue import PhaseCatalogueSeal, TaskCatalogue
from .planning.current_p12 import (
    CurrentP12Window,
    P12InformationMode,
    P12PredictionEvidenceState,
    P2Prediction,
    PredictedP2Slot,
    PreparedP12PlanTemplate,
    build_current_p12_windows,
    build_p2_prediction,
    build_predicted_p2_slots,
    evaluate_p12_prediction_evidence,
    normalize_p12_information_mode,
)
from .planning.phase_identity import assert_same_dispatch_authority, next_layer_dispatch_phase
from .planning.planner import (
    AlgorithmPlan,
    AlgorithmWave,
    ExecutionMode,
    FairnessContract,
    OrderOnlyPlanner,
    PlannerScope,
    SchedulingProblem,
    SchedulingTask,
    build_problem_from_catalogue,
    validate_order_only_pair,
)
from .planning.schema_api import DataclassSchemaAdapter, SharedSchemaAdapter, SharedSchemaConstructors
from .planning.taskization import CanonicalTaskizer, TaskizationSpec
from .prediction.metrics import P2PredictionQuality, evaluate_p2_prediction
from .prediction.timing import P12RankTimingProfile, causal_last_observed_timing_estimate

__all__ = [name for name in globals() if not name.startswith("_")]
