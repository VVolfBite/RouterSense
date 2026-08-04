from __future__ import annotations

"""Live execution session for the unified Current-P12 scheduler."""

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from rs_sim.contracts.schema import PhaseKey, PhaseKind, WindowKey
from rs_sim.scheduler.core.rscf_core import RSCFWireCostModel
from rs_sim.scheduler.decorators.composition import REGISTERED_CORE_IDS
from rs_sim.scheduler.decorators.planning_gate import (
    PlanningDecision, PlanningGate, PlanningMode, PlanningTrigger,
)
from rs_sim.scheduler.planning.current_p12 import PreparedP12PlanTemplate, PredictedP2Slot
from rs_sim.scheduler.planning.planner import (
    AlgorithmPlan, AlgorithmWave, FairnessContract, ExecutionMode,
    OrderOnlyPlanner, PlannerScope, SchedulingProblem, SchedulingTask,
    _critical_completion_objective_for_waves,
    _estimated_p1_release_tail_for_waves,
    build_problem_from_catalogue,
)
from rs_sim.scheduler.stable import stable_digest, stable_json
from rs_sim.scheduler.execution.state import PENDING_DEPENDENCY, READY_UNCOMMITTED
from rs_sim.scheduler.execution.window_arbiter import (
    PhaseFrontier, PrefixWindowArbiter, ReleaseFrontierWindowArbiter,
    WindowArbitrationContext, WindowArbitrationDecision,
)


class ReleaseMode(str):
    RANK_LOCAL = "RANK_LOCAL"
    PHASE_BARRIER = "PHASE_BARRIER"

    def __new__(cls, value: str):
        normalized = str(value).upper()
        if normalized not in {cls.RANK_LOCAL, cls.PHASE_BARRIER}:
            raise ValueError(f"unsupported release mode {value!r}")
        return str.__new__(cls, normalized)


def current_p12_phase_keys(
    *, run_id: str, sample_id: str, base_layer_index: int
) -> tuple[PhaseKey, PhaseKey]:
    layer = int(base_layer_index)
    if layer < 0:
        raise ValueError("base_layer_index must be non-negative")
    return (
        PhaseKey(str(run_id), str(sample_id), layer, PhaseKind.COMBINE),
        PhaseKey(str(run_id), str(sample_id), layer + 1, PhaseKind.DISPATCH),
    )


@dataclass(frozen=True, slots=True)
class SchedulerWindow:
    window_key: WindowKey
    base_layer_index: int
    phase_keys: tuple[PhaseKey, PhaseKey]
    window_digest: str

    @property
    def anchor_layer_id(self) -> int:
        return int(self.base_layer_index)

    @property
    def referenced_phase_keys(self) -> tuple[PhaseKey, PhaseKey]:
        return self.phase_keys

    @property
    def planning_window_digest(self) -> str:
        return self.window_digest

    @classmethod
    def build(
        cls, *, run_id: str, sample_id: str, window_index: int, base_layer_index: int
    ) -> "SchedulerWindow":
        window_key = WindowKey(str(run_id), str(sample_id), int(window_index))
        phase_keys = current_p12_phase_keys(
            run_id=run_id, sample_id=sample_id, base_layer_index=base_layer_index
        )
        payload = {
            "window_key": window_key,
            "base_layer_index": int(base_layer_index),
            "phase_keys": phase_keys,
        }
        return cls(
            window_key=window_key,
            base_layer_index=int(base_layer_index),
            phase_keys=phase_keys,
            window_digest=stable_digest(payload),
        )


@dataclass(frozen=True, slots=True)
class LiveFairnessInputs:
    receiver_contract_rule_digest: str
    buffer_profile_digest: str
    compiler_digest: str
    transport_digest: str
    release_model_digest: str
    information_digest: str
    cost_model_digest: str

    def __post_init__(self) -> None:
        for name in (
            "receiver_contract_rule_digest", "buffer_profile_digest",
            "compiler_digest", "transport_digest", "release_model_digest",
            "information_digest", "cost_model_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class LivePolicySpec:
    core_id: str
    planning_mode: PlanningMode
    release_mode: ReleaseMode
    scope: PlannerScope
    rank_count: int
    fairness: LiveFairnessInputs
    event_triggers: tuple[PlanningTrigger, ...] = ()
    rscf_wire_cost_model: RSCFWireCostModel | None = None
    rank_to_node: tuple[int, ...] | None = None
    safe_scope_selection: bool = False
    oracle_time_limit_ms: int = 30_000
    oracle_relative_gap: float = 0.0
    oracle_require_certified: bool = True

    def __post_init__(self) -> None:
        if self.core_id not in REGISTERED_CORE_IDS:
            raise ValueError(f"unregistered algorithm core {self.core_id!r}")
        object.__setattr__(self, "planning_mode", PlanningMode(self.planning_mode))
        object.__setattr__(self, "release_mode", ReleaseMode(self.release_mode))
        object.__setattr__(self, "scope", PlannerScope(self.scope))
        object.__setattr__(
            self, "event_triggers", tuple(PlanningTrigger(item) for item in self.event_triggers)
        )
        if not isinstance(self.rank_count, int) or isinstance(self.rank_count, bool) or self.rank_count <= 0:
            raise ValueError("rank_count must be positive")
        if self.rank_to_node is not None:
            topology = tuple(int(value) for value in self.rank_to_node)
            if len(topology) != self.rank_count or any(value < 0 for value in topology):
                raise ValueError("rank_to_node must contain one non-negative node id per rank")
            object.__setattr__(self, "rank_to_node", topology)
        if self.planning_mode is PlanningMode.EVENT and not self.event_triggers:
            raise ValueError("event(...) requires explicit event triggers")
        if self.safe_scope_selection and self.scope is not PlannerScope.WINDOW_JOINT:
            raise ValueError("safe(...) requires joint(...) scope")
        if not isinstance(self.oracle_time_limit_ms, int) or self.oracle_time_limit_ms <= 0:
            raise ValueError("oracle_time_limit_ms must be a positive int")
        if not 0.0 <= float(self.oracle_relative_gap) < 1.0:
            raise ValueError("oracle_relative_gap must be in [0, 1)")
        if not isinstance(self.oracle_require_certified, bool):
            raise ValueError("oracle_require_certified must be bool")


@dataclass(frozen=True, slots=True)
class LivePlanActivation:
    activation_index: int
    algorithm_plan: AlgorithmPlan
    phase_plan_ids: tuple[tuple[PhaseKey, str], ...]
    canonical_wave_membership_digest: str
    activated_at_ns: int
    activation_digest: str


@dataclass(frozen=True, slots=True)
class PreparedLiveActivation:
    algorithm_plan: AlgorithmPlan
    phase_orders: tuple[tuple[PhaseKey, tuple[str, ...]], ...]
    canonical_wave_membership_digest: str
    task_catalogue_digest: str
    task_boundary_digest: str
    prepared_at_ns: int
    prepared_digest: str


@dataclass(frozen=True, slots=True)
class LiveObservationResult:
    decision: PlanningDecision
    activation: LivePlanActivation | None


@dataclass(frozen=True, slots=True)
class LiveCompileSelection:
    decision: WindowArbitrationDecision
    phase_key: PhaseKey | None
    compile_attempt: Any | None


def canonical_order_only_wave_membership_digest(problem: SchedulingProblem) -> str:
    """Freeze identical logical wave membership for every ORDER_ONLY policy.

    Singleton logical waves deliberately isolate ordering from taskization and
    wave membership.  The common compiler may still co-pack compatible ready
    tasks according to the same resource snapshot for all policies.
    """

    return stable_digest(tuple((task.task_id,) for task in problem.tasks))


def _merge_phase_local_rank_release_waves(
    waves: Iterable[Iterable[str]],
    *,
    p1_task_ids: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """Merge independent Local phase orders without changing either order.

    A rank-local release treatment must allow a ready P2 head to overtake later
    P1 waves.  Concatenating all P1 waves before all P2 waves silently restores
    a phase barrier at the priority layer.  Zipping the two phase-local wave
    sequences preserves each phase's exact internal order while exposing P2
    after the corresponding earlier Local wave position.  Runtime readiness
    and transport legality remain authoritative.
    """

    p1_ids = frozenset(str(item) for item in p1_task_ids)
    normalized = tuple(tuple(str(item) for item in wave) for wave in waves if tuple(wave))
    flattened = tuple(item for wave in normalized for item in wave)
    if len(set(flattened)) != len(flattened):
        raise ValueError("phase-local rank-release merge received duplicate task IDs")
    p1_waves = tuple(
        tuple(item for item in wave if item in p1_ids)
        for wave in normalized
    )
    p2_waves = tuple(
        tuple(item for item in wave if item not in p1_ids)
        for wave in normalized
    )
    p1_waves = tuple(wave for wave in p1_waves if wave)
    p2_waves = tuple(wave for wave in p2_waves if wave)
    merged: list[tuple[str, ...]] = []
    for index in range(max(len(p1_waves), len(p2_waves))):
        wave = (
            (p1_waves[index] if index < len(p1_waves) else ())
            + (p2_waves[index] if index < len(p2_waves) else ())
        )
        if wave:
            merged.append(wave)
    merged_flat = tuple(item for wave in merged for item in wave)
    if set(merged_flat) != set(flattened) or len(merged_flat) != len(flattened):
        raise ValueError("phase-local rank-release merge lost canonical tasks")
    return tuple(merged)


class LivePolicySession:
    """Experiment-facing live scheduler session used by the Integration Owner."""

    def __init__(
        self,
        *,
        controller: Any,
        adapter: Any,
        horizon_window: SchedulerWindow,
        spec: LivePolicySpec,
    ) -> None:
        self.controller = controller
        self.adapter = adapter
        self.horizon_window = horizon_window
        self.spec = spec
        self.gate = PlanningGate(
            spec.planning_mode,
            event_triggers=(spec.event_triggers if spec.planning_mode is PlanningMode.EVENT else None),
        )
        self._activations: list[LivePlanActivation] = []
        self._external_preferred_task_ids: tuple[str, ...] | None = None
        self._external_preferred_waves: tuple[tuple[str, ...], ...] | None = None
        self._phase_by_token = {
            stable_json(adapter.phase_payload(phase_key)): phase_key
            for phase_key in horizon_window.phase_keys
        }

    @property
    def activations(self) -> tuple[LivePlanActivation, ...]:
        return tuple(self._activations)

    @property
    def activation_count(self) -> int:
        return len(self._activations)

    @property
    def phase_keys(self) -> tuple[PhaseKey, ...]:
        return self.horizon_window.phase_keys

    def _require_phase(self, phase_key: PhaseKey) -> PhaseKey:
        if phase_key not in self.horizon_window.phase_keys:
            raise ValueError("phase_key is outside this session horizon")
        return phase_key

    def _phase_has_unfinished_tasks(self, phase_key: PhaseKey) -> bool:
        task_ids = tuple(self.controller.catalogue.task_ids_for_phase(phase_key))
        if not task_ids:
            return False
        return any(
            self.controller.runtime.facts(task_id).state != "COMPLETED"
            for task_id in task_ids
        )

    def phase_has_unfinished_tasks(self, phase_key: PhaseKey) -> bool:
        """Return canonical phase liveness without mutating planning authority."""

        return self._phase_has_unfinished_tasks(self._require_phase(phase_key))

    def _eligible_activation_phases(self) -> tuple[PhaseKey, ...]:
        unfinished = tuple(
            phase_key
            for phase_key in self.horizon_window.phase_keys
            if self._phase_has_unfinished_tasks(phase_key)
        )
        if self.spec.release_mode == ReleaseMode.PHASE_BARRIER:
            return unfinished[:1]
        return unfinished

    def _build_problem(self, phase_keys: Iterable[PhaseKey]) -> SchedulingProblem:
        fairness = self.spec.fairness
        return build_problem_from_catalogue(
            adapter=self.adapter,
            catalogue=self.controller.catalogue,
            runtime=self.controller.runtime,
            phase_keys=tuple(phase_keys),
            rank_count=self.spec.rank_count,
            receiver_contract_rule_digest=fairness.receiver_contract_rule_digest,
            buffer_profile_digest=fairness.buffer_profile_digest,
            compiler_digest=fairness.compiler_digest,
            transport_digest=fairness.transport_digest,
            release_model_digest=fairness.release_model_digest,
            information_digest=fairness.information_digest,
            cost_model_digest=fairness.cost_model_digest,
            eligible_states=(PENDING_DEPENDENCY, READY_UNCOMMITTED),
        )

    def _algorithm_plan(self, problem: SchedulingProblem) -> AlgorithmPlan:
        planner = OrderOnlyPlanner()

        def plan_for(scope: PlannerScope) -> AlgorithmPlan:
            semantic_phase_ordinal: int | None = None
            if self.spec.core_id in {"rscf", "oracle"} and len(problem.phase_tokens) == 1:
                phase_token = problem.phase_tokens[0]
                matching_keys = tuple(
                    key
                    for key in self.horizon_window.phase_keys
                    if stable_json(self.adapter.phase_payload(key)) == phase_token
                )
                if len(matching_keys) != 1:
                    raise ValueError(
                        "single-phase RSCF problem must map to one horizon phase"
                    )
                phase_kind = matching_keys[0].phase_kind.value
                if phase_kind == "COMBINE":
                    semantic_phase_ordinal = 1
                elif phase_kind == "DISPATCH":
                    semantic_phase_ordinal = 2
                else:
                    raise ValueError(
                        f"unsupported Current-P12 phase kind {phase_kind!r}"
                    )
            return planner.plan(
                problem,
                algorithm_id=self.spec.core_id,
                rscf_wire_cost_model=self.spec.rscf_wire_cost_model,
                planner_scope=scope,
                rank_to_node=self.spec.rank_to_node,
                rscf_semantic_phase_ordinal=semantic_phase_ordinal,
                release_mode=str(self.spec.release_mode),
                oracle_time_limit_ms=int(self.spec.oracle_time_limit_ms),
                oracle_relative_gap=float(self.spec.oracle_relative_gap),
                oracle_require_certified=bool(self.spec.oracle_require_certified),
            )

        joint_or_declared = plan_for(self.spec.scope)
        if not self.spec.safe_scope_selection:
            return joint_or_declared
        if self.spec.scope is not PlannerScope.WINDOW_JOINT:
            raise ValueError("safe scheduling requires a declared Joint scope")
        if self.spec.planning_mode is not PlanningMode.GLOBAL:
            raise ValueError("safe scheduling currently requires global_(...) cadence")

        local_candidate = plan_for(PlannerScope.PHASE_LOCAL)
        joint_candidate = joint_or_declared
        model = self.spec.rscf_wire_cost_model
        local_objective = _critical_completion_objective_for_waves(
            problem, local_candidate.waves, model, release_mode=str(self.spec.release_mode)
        )
        joint_objective = _critical_completion_objective_for_waves(
            problem, joint_candidate.waves, model, release_mode=str(self.spec.release_mode)
        )
        if joint_objective <= local_objective:
            selected = joint_candidate
            choice = "WINDOW_JOINT"
        else:
            selected = local_candidate
            choice = "PHASE_LOCAL"
        diagnostics = dict(selected.diagnostics)
        diagnostics.update(
            {
                "safe_selector_choice": choice,
                "safe_selector_reason": "SAME_CORE_EXPECTED_COMPLETION",
                "safe_selector_local_estimated_objective": int(local_objective),
                "safe_selector_joint_estimated_objective": int(joint_objective),
            }
        )
        return replace(
            selected,
            plan_digest=stable_digest(
                {
                    "safe_core": self.spec.core_id,
                    "selected_plan_digest": selected.plan_digest,
                    "choice": choice,
                    "local_objective": int(local_objective),
                    "joint_objective": int(joint_objective),
                }
            ),
            diagnostics=tuple(sorted(diagnostics.items())),
        )

    def set_external_preferred_task_ids(self, task_ids: Iterable[str]) -> None:
        """Install one immutable-window preferred order over currently bound tasks.

        Current-P12 uses this order instead of ``activations[-1]`` so incremental
        P2 binding cannot replace the P1/P2 joint priority state with a P2-only
        activation.
        """

        ordered = tuple(str(item) for item in task_ids)
        if len(set(ordered)) != len(ordered):
            raise ValueError("external preferred order contains duplicate task IDs")
        self._external_preferred_task_ids = ordered
        # A flat externally supplied order is authoritative.  Do not retain
        # wave membership from an earlier Joint template, otherwise a real
        # phase-local fallback would still be submitted through stale Joint
        # waves.
        # Empty tuple means an explicitly flat order.  ``None`` would fall back
        # to the latest activation's waves in ``arbitrate`` and reintroduce
        # stale Joint wave membership.
        self._external_preferred_waves = ()

    @property
    def external_preferred_task_ids(self) -> tuple[str, ...] | None:
        return self._external_preferred_task_ids

    def set_external_preferred_waves(self, waves: Iterable[Iterable[str]]) -> None:
        normalized = tuple(tuple(str(item) for item in wave) for wave in waves)
        if any(not wave for wave in normalized):
            raise ValueError("external preferred waves cannot contain an empty wave")
        flattened = tuple(item for wave in normalized for item in wave)
        if len(set(flattened)) != len(flattened):
            raise ValueError("external preferred waves contain duplicate task IDs")
        self._external_preferred_waves = normalized
        self._external_preferred_task_ids = flattened

    @property
    def external_preferred_waves(self) -> tuple[tuple[str, ...], ...] | None:
        return self._external_preferred_waves

    def prepare_current_p12_template(
        self,
        *,
        p1_phase_key: PhaseKey,
        p2_phase_key: PhaseKey,
        predicted_slots: Iterable[PredictedP2Slot],
        planning_window_digest: str,
        now_ns: int,
    ) -> tuple[PreparedP12PlanTemplate, PreparedLiveActivation | None]:
        """Run the algorithm once over exact P1 tasks plus predicted P2 slots."""

        p1 = self._require_phase(p1_phase_key)
        p2 = self._require_phase(p2_phase_key)
        p1_problem = self._build_problem((p1,))
        p1_tasks = tuple(p1_problem.tasks)
        p2_token = stable_json(self.adapter.phase_payload(p2))
        slots = tuple(predicted_slots)
        synthetic = tuple(
            SchedulingTask(
                task_id=item.slot_id,
                phase_token=p2_token,
                phase_ordinal=1,
                src_rank=int(item.src_rank),
                dst_rank=int(item.dst_rank),
                payload_bytes=int(item.payload_bytes),
                chunk_index=int(item.chunk_index),
                byte_offset=int(item.byte_offset),
                ready_at_ns=0,
            )
            for item in slots
        )
        tasks = tuple(p1_tasks) + synthetic
        if not tasks:
            raise ValueError("Current P12 template requires at least one P1 or P2 task")
        phase_tokens = tuple(dict.fromkeys(item.phase_token for item in tasks))
        fairness_base = self.spec.fairness
        fairness = FairnessContract(
            task_catalogue_digest=stable_digest(tuple(item.task_id for item in tasks)),
            task_boundary_digest=stable_digest(
                tuple(
                    (
                        item.task_id,
                        item.phase_token,
                        item.src_rank,
                        item.dst_rank,
                        item.chunk_index,
                        item.byte_offset,
                        item.payload_bytes,
                    )
                    for item in tasks
                )
            ),
            taskization_digest=stable_digest(self.controller.catalogue.taskizer.spec.stable_payload()),
            receiver_contract_rule_digest=fairness_base.receiver_contract_rule_digest,
            buffer_profile_digest=fairness_base.buffer_profile_digest,
            compiler_digest=fairness_base.compiler_digest,
            transport_digest=fairness_base.transport_digest,
            release_model_digest=fairness_base.release_model_digest,
            information_digest=fairness_base.information_digest,
            cost_model_digest=fairness_base.cost_model_digest,
        )
        problem = SchedulingProblem(
            rank_count=self.spec.rank_count,
            tasks=tasks,
            phase_tokens=phase_tokens,
            fairness=fairness,
            # Preserve the P0-time P2 prediction as advisory geometry.  The
            # synthetic slots remain non-executable until the P1 release DAG
            # allows them, while RSCF may use the same matrix to score P1.
        )
        plan = self._algorithm_plan(problem)
        p1_ids = {item.task_id for item in p1_tasks}
        execution_waves = tuple(tuple(wave.task_ids) for wave in plan.waves if wave.task_ids)
        execution_merge = "PLANNER_ORDER"
        if (
            self.spec.scope is PlannerScope.PHASE_LOCAL
            and self.spec.release_mode == ReleaseMode.RANK_LOCAL
            and slots
        ):
            execution_waves = _merge_phase_local_rank_release_waves(
                execution_waves, p1_task_ids=p1_ids
            )
            execution_merge = "PHASE_LOCAL_RANK_RELEASE_WAVE_ZIP"
        execution_order = tuple(item for wave in execution_waves for item in wave)
        p1_order = tuple(item for item in execution_order if item in p1_ids)
        semantic = {
            "schema_version": "PREPARED_P12_PLAN_TEMPLATE",
            "planning_window_digest": str(planning_window_digest),
            "algorithm_id": str(plan.algorithm_id),
            "ordered_tokens": execution_order,
            "ordered_waves": execution_waves,
            "execution_merge": execution_merge,
            "p1_task_ids": tuple(item.task_id for item in p1_tasks),
            "p2_slot_digests": tuple(item.slot_digest for item in slots),
            "created_at_ns": int(now_ns),
            "algorithm_plan_digest": str(plan.plan_digest),
        }
        template = PreparedP12PlanTemplate(
            planning_window_digest=str(planning_window_digest),
            algorithm_id=str(plan.algorithm_id),
            ordered_tokens=execution_order,
            p1_task_ids=tuple(item.task_id for item in p1_tasks),
            p2_slots=slots,
            created_at_ns=int(now_ns),
            algorithm_plan_digest=str(plan.plan_digest),
            template_digest=stable_digest(semantic),
            ordered_waves=execution_waves,
        )
        p1_waves = tuple(
            tuple(task_id for task_id in wave if task_id in p1_ids)
            for wave in execution_waves
        )
        p1_waves = tuple(wave for wave in p1_waves if wave)
        if p1_waves:
            self.set_external_preferred_waves(p1_waves)
        else:
            self.set_external_preferred_task_ids(p1_order)
        if not p1_order:
            return template, None
        wave_digest = canonical_order_only_wave_membership_digest(problem)
        prepared_payload = {
            "algorithm_plan_digest": plan.plan_digest,
            "phase_orders": ((p1, p1_order),),
            "canonical_wave_membership_digest": wave_digest,
            "task_catalogue_digest": fairness.task_catalogue_digest,
            "task_boundary_digest": fairness.task_boundary_digest,
            "prepared_at_ns": int(now_ns),
            "p12_template_digest": template.template_digest,
        }
        prepared = PreparedLiveActivation(
            algorithm_plan=plan,
            phase_orders=((p1, p1_order),),
            canonical_wave_membership_digest=wave_digest,
            task_catalogue_digest=fairness.task_catalogue_digest,
            task_boundary_digest=fairness.task_boundary_digest,
            prepared_at_ns=int(now_ns),
            prepared_digest=stable_digest(prepared_payload),
        )
        return template, prepared

    def prepare_activation(
        self,
        *,
        now_ns: int,
        phase_keys: Iterable[PhaseKey] | None = None,
        respect_release_barrier: bool = True,
    ) -> PreparedLiveActivation | None:
        """Compute one immutable live plan without activating phase authority.

        This method is the ControlLine completion boundary.  The returned
        object can be queued on ExecutionBindingLine and activated later.
        """

        eligible = (
            self._eligible_activation_phases()
            if respect_release_barrier
            else tuple(
                phase_key
                for phase_key in self.horizon_window.phase_keys
                if self._phase_has_unfinished_tasks(phase_key)
            )
        )
        if phase_keys is None:
            selected_phase_keys = eligible
        else:
            requested = tuple(self._require_phase(item) for item in phase_keys)
            selected_phase_keys = tuple(item for item in eligible if item in requested)
        if not selected_phase_keys:
            return None
        problem = self._build_problem(selected_phase_keys)
        if not problem.tasks:
            return None
        plan = self._algorithm_plan(problem)
        task_by_id = {item.task_id: item for item in problem.tasks}
        token_to_ids: dict[str, list[str]] = {token: [] for token in problem.phase_tokens}
        for task_id in plan.ordered_task_ids:
            token_to_ids[task_by_id[task_id].phase_token].append(task_id)
        phase_orders: list[tuple[PhaseKey, tuple[str, ...]]] = []
        for phase_key in selected_phase_keys:
            token = stable_json(self.adapter.phase_payload(phase_key))
            ordered = tuple(token_to_ids.get(token, ()))
            if ordered:
                phase_orders.append((phase_key, ordered))
        wave_digest = canonical_order_only_wave_membership_digest(problem)
        payload = {
            "algorithm_plan_digest": plan.plan_digest,
            "phase_orders": tuple(phase_orders),
            "canonical_wave_membership_digest": wave_digest,
            "task_catalogue_digest": problem.fairness.task_catalogue_digest,
            "task_boundary_digest": problem.fairness.task_boundary_digest,
            "prepared_at_ns": int(now_ns),
        }
        return PreparedLiveActivation(
            algorithm_plan=plan,
            phase_orders=tuple(phase_orders),
            canonical_wave_membership_digest=wave_digest,
            task_catalogue_digest=problem.fairness.task_catalogue_digest,
            task_boundary_digest=problem.fairness.task_boundary_digest,
            prepared_at_ns=int(now_ns),
            prepared_digest=stable_digest(payload),
        )

    def prepared_is_current(self, prepared: PreparedLiveActivation) -> bool:
        for phase_key, ordered in prepared.phase_orders:
            record = self.controller.authority.record_view(phase_key)
            frozen = set(record.committed_task_ids) | set(record.running_task_ids) | set(
                record.completed_task_ids
            )
            expected = set(record.canonical_task_ids) - frozen
            if set(ordered) != expected:
                return False
        return True

    def event_replan_improves_current_frontier(
        self, prepared: PreparedLiveActivation
    ) -> bool:
        """Accept an EVENT suffix only when it improves the committed frontier.

        EVENT planning observes exact readiness after execution has begun, but
        replacing a good immutable-window plan can fragment future matchings.
        Compare the candidate and the still-valid committed suffix under the
        same wire/readiness/compute objective.  A small margin pays for the
        extra control and binding work and prevents plan churn on numerical
        ties.  GLOBAL and non-Joint modes are unaffected.
        """

        if (
            self.spec.planning_mode is not PlanningMode.EVENT
            or self.spec.scope is not PlannerScope.WINDOW_JOINT
        ):
            return True
        current_waves = self._external_preferred_waves
        if not current_waves:
            return True
        phase_keys = tuple(phase_key for phase_key, _order in prepared.phase_orders)
        if not phase_keys:
            return False
        problem = self._build_problem(phase_keys)
        remaining = {task.task_id for task in problem.tasks}
        if not remaining:
            return False

        def projected_waves(
            source: Iterable[Iterable[str]],
        ) -> tuple[AlgorithmWave, ...]:
            result: list[AlgorithmWave] = []
            covered: set[str] = set()
            phase_by_id = {task.task_id: task.phase_token for task in problem.tasks}
            for wave_ids in source:
                ids = tuple(
                    task_id
                    for task_id in wave_ids
                    if task_id in remaining and task_id not in covered
                )
                if not ids:
                    continue
                covered.update(ids)
                result.append(
                    AlgorithmWave(
                        wave_id=len(result),
                        task_ids=ids,
                        phase_tokens=tuple(dict.fromkeys(phase_by_id[item] for item in ids)),
                    )
                )
            if covered != remaining:
                return ()
            return tuple(result)

        current = projected_waves(current_waves)
        candidate = projected_waves(
            tuple(tuple(wave.task_ids) for wave in prepared.algorithm_plan.waves)
        )
        if not current:
            return True
        if not candidate:
            return False
        model = self.spec.rscf_wire_cost_model
        current_objective = _critical_completion_objective_for_waves(
            problem, current, model, release_mode=str(self.spec.release_mode)
        )
        candidate_objective = _critical_completion_objective_for_waves(
            problem, candidate, model, release_mode=str(self.spec.release_mode)
        )
        current_p1_tail, _ = _estimated_p1_release_tail_for_waves(
            problem, current, model, release_mode=str(self.spec.release_mode)
        )
        candidate_p1_tail, _ = _estimated_p1_release_tail_for_waves(
            problem, candidate, model, release_mode=str(self.spec.release_mode)
        )
        # Require a 0.25% completion margin and never delay the P1 release
        # frontier.  The objective is integer nanoseconds, so retain at least a
        # one-nanosecond strict improvement for small synthetic tests.
        required = max(1, int(round(float(current_objective) * 0.0025)))
        return (
            candidate_objective + required < current_objective
            and candidate_p1_tail <= current_p1_tail
        )

    def activate_prepared(
        self,
        prepared: PreparedLiveActivation,
        *,
        activated_at_ns: int,
        skip_if_stale: bool = False,
    ) -> LivePlanActivation | None:
        """Activate a ControlLine result after ExecutionBindingLine service."""

        if not isinstance(prepared, PreparedLiveActivation):
            raise TypeError("prepared must be PreparedLiveActivation")
        if not self.prepared_is_current(prepared):
            if skip_if_stale:
                return None
            raise ValueError("prepared activation is stale against canonical phase authority")
        phase_plan_ids: list[tuple[PhaseKey, str]] = []
        for phase_key, ordered in prepared.phase_orders:
            activated = self.controller.activate_plan(
                phase_key=phase_key,
                window_key=self.horizon_window.window_key,
                ordered_task_ids=ordered,
                now_ns=int(activated_at_ns),
            )
            phase_plan_ids.append(
                (phase_key, str(self.adapter.plan_view(activated).plan_id))
            )
        activation_index = len(self._activations)
        payload = {
            "activation_index": activation_index,
            "prepared_digest": prepared.prepared_digest,
            "phase_plan_ids": tuple(phase_plan_ids),
            "activated_at_ns": int(activated_at_ns),
        }
        activation = LivePlanActivation(
            activation_index=activation_index,
            algorithm_plan=prepared.algorithm_plan,
            phase_plan_ids=tuple(phase_plan_ids),
            canonical_wave_membership_digest=prepared.canonical_wave_membership_digest,
            activated_at_ns=int(activated_at_ns),
            activation_digest=stable_digest(payload),
        )
        self._activations.append(activation)
        return activation

    def _activate(self, *, now_ns: int) -> LivePlanActivation | None:
        prepared = self.prepare_activation(now_ns=int(now_ns))
        if prepared is None:
            return None
        return self.activate_prepared(prepared, activated_at_ns=int(now_ns))

    def on_observation(
        self,
        observation_id: str,
        *,
        trigger: PlanningTrigger | str,
        changed: bool,
        closure_satisfied: bool = False,
        now_ns: int,
    ) -> LiveObservationResult:
        decision = self.gate.on_observation(
            str(observation_id),
            trigger=PlanningTrigger(trigger),
            changed=bool(changed),
            closure_satisfied=bool(closure_satisfied),
        )
        activation = (
            self._activate(now_ns=int(now_ns))
            if decision.action == "CREATE_PLAN_VERSION"
            else None
        )
        return LiveObservationResult(decision=decision, activation=activation)

    def register_expectation(
        self, expectation: Any, *, registered_at_ns: int
    ) -> tuple[Any, ...]:
        view = self.adapter.expectation_view(expectation)
        self._require_phase(view.phase_key)
        tasks = self.controller.register_expectation(
            expectation, registered_at_ns=int(registered_at_ns)
        )
        expectation_trigger = (
            PlanningTrigger.DESCRIPTOR_DELIVERY
            if view.phase_key.phase_kind.value == "DISPATCH"
            else PlanningTrigger.EXPECTATION_AVAILABLE
        )
        self.on_observation(
            f"EXPECTATION:{view.expectation_digest}",
            trigger=expectation_trigger,
            changed=True,
            now_ns=int(registered_at_ns),
        )
        return tasks

    def note_receive_permit(self, task_id: str, *, at_ns: int) -> tuple[LiveObservationResult, ...]:
        before = self.controller.runtime.facts(task_id).state
        self.controller.note_receive_permit(task_id, at_ns=int(at_ns))
        after = self.controller.runtime.facts(task_id).state
        results = [
            self.on_observation(
                f"PERMIT:{task_id}:{int(at_ns)}",
                trigger=PlanningTrigger.PERMIT_GRANTED,
                changed=True,
                now_ns=int(at_ns),
            )
        ]
        if before != READY_UNCOMMITTED and after == READY_UNCOMMITTED:
            results.append(
                self.on_observation(
                    f"TASK_READY:{task_id}:{int(at_ns)}",
                    trigger=PlanningTrigger.TASK_READY,
                    changed=True,
                    now_ns=int(at_ns),
                )
            )
        return tuple(results)

    def note_source_payload_ready(
        self, task_id: str, *, at_ns: int
    ) -> tuple[LiveObservationResult, ...]:
        before = self.controller.runtime.facts(task_id).state
        self.controller.note_source_payload_ready(task_id, at_ns=int(at_ns))
        after = self.controller.runtime.facts(task_id).state
        results = [
            self.on_observation(
                f"SOURCE_PAYLOAD_READY:{task_id}:{int(at_ns)}",
                trigger=PlanningTrigger.SOURCE_PAYLOAD_READY,
                changed=True,
                now_ns=int(at_ns),
            )
        ]
        if before != READY_UNCOMMITTED and after == READY_UNCOMMITTED:
            results.append(
                self.on_observation(
                    f"TASK_READY:{task_id}:{int(at_ns)}",
                    trigger=PlanningTrigger.TASK_READY,
                    changed=True,
                    now_ns=int(at_ns),
                )
            )
        return tuple(results)

    def close_observations(self, *, now_ns: int) -> LiveObservationResult:
        return self.on_observation(
            f"WINDOW_CLOSURE:{self.horizon_window.window_digest}",
            trigger=PlanningTrigger.OBSERVATION_CLOSURE,
            changed=False,
            closure_satisfied=True,
            now_ns=int(now_ns),
        )

    def on_resource_release(self) -> PlanningDecision:
        return self.gate.on_resource_release()

    def advance_phase_barrier(self, *, now_ns: int) -> LivePlanActivation | None:
        if self.spec.release_mode != ReleaseMode.PHASE_BARRIER:
            raise ValueError("advance_phase_barrier is valid only for PHASE_BARRIER")
        return self._activate(now_ns=int(now_ns))

    def _live_frontiers(self) -> tuple[PhaseFrontier, ...]:
        frontiers: list[PhaseFrontier] = []
        phases = self._eligible_activation_phases()
        for phase_key in phases:
            active = self.controller.authority.active_plan(phase_key)
            if active is None:
                continue
            view = self.adapter.plan_view(active)
            ready_ids = tuple(
                task_id
                for task_id in view.remaining_task_ids
                if self.controller.runtime.facts(task_id).state == READY_UNCOMMITTED
            )
            frontiers.append(
                PhaseFrontier.build(
                    phase_key=phase_key,
                    authority_stamp=self.controller.authority.authority_stamp(phase_key),
                    ready_task_ids=ready_ids,
                )
            )
        return tuple(frontiers)

    def arbitrate(
        self,
        *,
        transport_snapshot_digest: str,
        observed_at_ns: int,
        max_prefix_tasks: int = 1,
    ) -> tuple[WindowArbitrationContext, WindowArbitrationDecision]:
        context = WindowArbitrationContext.build(
            window_key=self.horizon_window.window_key,
            frontiers=self._live_frontiers(),
            transport_snapshot_digest=str(transport_snapshot_digest),
            observed_at_ns=int(observed_at_ns),
        )
        preferred = (
            self._external_preferred_task_ids
            if self._external_preferred_task_ids is not None
            else (self._activations[-1].algorithm_plan.ordered_task_ids if self._activations else None)
        )
        preferred_waves = (
            self._external_preferred_waves
            if self._external_preferred_waves is not None
            else (
                tuple(tuple(wave.task_ids) for wave in self._activations[-1].algorithm_plan.waves if wave.task_ids)
                if self._activations else None
            )
        )
        if preferred is not None:
            # All formal P12 treatments consume the approved plan as a
            # ready-aware priority template.  Scope controls which phase
            # authorities may become live.  The authoritative release_mode
            # controls whether P2 is exposed after a global phase barrier or
            # by rank-local release; scope never overrides that contract.
            # Within the currently eligible frontiers, a task
            # that is not ready never reserves a slot; when it becomes ready it
            # catches up at the next non-preemptive canonical-task boundary.
            # This matches the authoritative metric replay and avoids silently
            # judging live execution with rigid planning-wave barriers.
            arbiter = ReleaseFrontierWindowArbiter(
                preferred_task_ids=preferred,
                max_prefix_tasks=int(max_prefix_tasks),
            )
        else:
            arbiter = PrefixWindowArbiter(max_prefix_tasks=int(max_prefix_tasks))
        return context, arbiter.select(context)

    def compile_arbitrated(
        self,
        *,
        snapshot: Any,
        transport_snapshot_digest: str,
        now_ns: int,
        max_prefix_tasks: int = 1,
    ) -> LiveCompileSelection:
        _, decision = self.arbitrate(
            transport_snapshot_digest=transport_snapshot_digest,
            observed_at_ns=int(now_ns),
            max_prefix_tasks=int(max_prefix_tasks),
        )
        if decision.selected_phase_token is None:
            return LiveCompileSelection(decision=decision, phase_key=None, compile_attempt=None)
        phase_key = self._phase_by_token[decision.selected_phase_token]
        attempt = self.controller.compiler.compile_next(
            phase_key=phase_key,
            snapshot=snapshot,
            now_ns=int(now_ns),
            allowed_task_ids=decision.selected_task_ids,
        )
        return LiveCompileSelection(
            decision=decision, phase_key=phase_key, compile_attempt=attempt
        )


def build_live_policy_session(
    *,
    controller: Any,
    adapter: Any,
    run_id: str,
    sample_id: str,
    window_index: int,
    base_layer_index: int,
    core_id: str,
    planning_mode: PlanningMode | str,
    release_mode: ReleaseMode | str,
    scope: PlannerScope | str,
    rank_count: int,
    fairness: LiveFairnessInputs,
    event_triggers: Iterable[PlanningTrigger | str] = (),
    rscf_wire_cost_model: RSCFWireCostModel | None = None,
    rank_to_node: tuple[int, ...] | None = None,
    safe_scope_selection: bool = False,
    oracle_time_limit_ms: int = 30_000,
    oracle_relative_gap: float = 0.0,
    oracle_require_certified: bool = True,
) -> LivePolicySession:
    """Build the only live scheduler session used by Current-P12 runtime."""

    window = SchedulerWindow.build(
        run_id=str(run_id),
        sample_id=str(sample_id),
        window_index=int(window_index),
        base_layer_index=int(base_layer_index),
    )
    spec = LivePolicySpec(
        core_id=str(core_id),
        planning_mode=PlanningMode(planning_mode),
        release_mode=ReleaseMode(release_mode),
        scope=PlannerScope(scope),
        rank_count=int(rank_count),
        fairness=fairness,
        event_triggers=tuple(PlanningTrigger(item) for item in event_triggers),
        rscf_wire_cost_model=rscf_wire_cost_model,
        rank_to_node=rank_to_node,
        safe_scope_selection=bool(safe_scope_selection),
        oracle_time_limit_ms=int(oracle_time_limit_ms),
        oracle_relative_gap=float(oracle_relative_gap),
        oracle_require_certified=bool(oracle_require_certified),
    )
    return LivePolicySession(
        controller=controller, adapter=adapter, horizon_window=window, spec=spec
    )


__all__ = [
    "SchedulerWindow",
    "LiveCompileSelection",
    "LiveFairnessInputs",
    "LiveObservationResult",
    "LivePlanActivation",
    "LivePolicySession",
    "LivePolicySpec",
    "PreparedLiveActivation",
    "ReleaseMode",
    "build_live_policy_session",
    "canonical_order_only_wave_membership_digest",
    "current_p12_phase_keys",
]
