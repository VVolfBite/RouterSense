"""Discrete-event backend.

The backend owns simulated time, causal state, receiver capacity, and backend
observability.  Scheduler code consumes only the public objects exported here.
"""

from .adapters.schema import AttributeSharedObjectAdapter, CallablePhaseSemantics
from .adapters.trace_window import (
    BackendTraceFixtureBuilder,
    RegisteredTraceWindow,
    TraceFixtureRegistration,
)
from .core.engine import SimulationBackend
from .core.errors import (
    BackendContractError,
    BackendError,
    CapacityConfigurationError,
    DuplicateRegistrationError,
    IllegalTransitionError,
    UnknownObjectError,
)
from .core.internal import RankState, ReceiverJobStatus
from .core.ports import BackendCausalTimingPort, BackendSealReadinessPort, ExactRowPublisherPort
from .observability.metrics import (
    PhaseCausalTimingObservation,
    PhaseClosureSummary,
    PhaseRankMetricsSnapshot,
    ReceiverMetricsSnapshot,
    RemoteCanonicalTaskExpectationInput,
    WindowTerminalEvidence,
)
from .resources.capacity import (
    align_up,
    compute_fixture_staging_capacity_bytes_by_rank,
    compute_staging_capacity_bytes,
)
from .resources.costs import LinearReceiverCostModel
from .resources.rank_actor import RankActor, RankTransition
from .resources.receiver import ReceiverService

__all__ = [name for name in globals() if not name.startswith("_")]
