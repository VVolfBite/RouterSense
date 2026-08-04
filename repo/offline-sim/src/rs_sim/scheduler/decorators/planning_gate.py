from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable


class PlanningMode(enum.Enum):
    EVENT = "EVENT"
    GLOBAL = "GLOBAL"


class PlanningTrigger(enum.Enum):
    DESCRIPTOR_DELIVERY = "DESCRIPTOR_DELIVERY"
    EXPECTATION_AVAILABLE = "EXPECTATION_AVAILABLE"
    SOURCE_PAYLOAD_READY = "SOURCE_PAYLOAD_READY"
    PERMIT_GRANTED = "PERMIT_GRANTED"
    TASK_READY = "TASK_READY"
    OBSERVATION_CLOSURE = "OBSERVATION_CLOSURE"


_LEGACY_TRIGGER_PREFIXES: tuple[tuple[str, PlanningTrigger], ...] = (
    ("RECEIVE_EXPECTATION_AVAILABLE:", PlanningTrigger.DESCRIPTOR_DELIVERY),
    ("SOURCE_PAYLOAD_READY:", PlanningTrigger.TASK_READY),
    ("RECEIVE_PERMIT_GRANTED:", PlanningTrigger.PERMIT_GRANTED),
    ("DISPATCH_DESCRIPTOR_CLOSED:", PlanningTrigger.OBSERVATION_CLOSURE),
    ("COMBINE_EXPECTATION_CLOSED:", PlanningTrigger.OBSERVATION_CLOSURE),
)


@dataclass(frozen=True)
class PlanningDecision:
    action: str
    reason: str
    observation_id: str | None = None
    trigger: PlanningTrigger | None = None
    plan_count: int = 0


class PlanningGate:
    """Deterministic planning gate separated from execution stabilization.

    GLOBAL closure may defer finalization until the canonical catalogue is
    sealed. EVENT mode replans only on explicitly enabled frontier changes.
    """

    def __init__(
        self,
        mode: PlanningMode,
        *,
        event_triggers: Iterable[PlanningTrigger | str] | None = None,
        defer_global_finalize: bool = False,
        max_event_plans: int | None = None,
    ) -> None:
        self.mode = PlanningMode(mode)
        self._delegated_event_filter = event_triggers is None
        self.event_triggers = (
            frozenset()
            if event_triggers is None
            else frozenset(PlanningTrigger(item) for item in event_triggers)
        )
        if (
            self.mode is PlanningMode.EVENT
            and event_triggers is not None
            and not self.event_triggers
        ):
            raise ValueError("EVENT mode requires a non-empty configured trigger set")
        if max_event_plans is not None and (
            not isinstance(max_event_plans, int)
            or isinstance(max_event_plans, bool)
            or max_event_plans <= 0
        ):
            raise ValueError("max_event_plans must be a positive int or None")
        self.defer_global_finalize = bool(defer_global_finalize)
        self.max_event_plans = max_event_plans
        self._seen_observations: set[tuple[PlanningTrigger, str]] = set()
        self._plan_count = 0
        self._global_finalize_requested = False
        self._global_seal_digest: str | None = None

    @property
    def plan_count(self) -> int:
        return self._plan_count

    @property
    def global_finalize_requested(self) -> bool:
        return self._global_finalize_requested

    @property
    def global_seal_digest(self) -> str | None:
        return self._global_seal_digest

    @staticmethod
    def _infer_legacy_trigger(observation_id: str) -> PlanningTrigger:
        for prefix, trigger in _LEGACY_TRIGGER_PREFIXES:
            if observation_id.startswith(prefix):
                return trigger
        raise ValueError(
            "trigger is required when observation_id does not use a known integration prefix"
        )

    def on_observation(
        self,
        observation_id: str,
        *,
        trigger: PlanningTrigger | str | None = None,
        changed: bool,
        closure_satisfied: bool = False,
    ) -> PlanningDecision:
        observation_id = str(observation_id)
        normalized_trigger = (
            self._infer_legacy_trigger(observation_id)
            if trigger is None
            else PlanningTrigger(trigger)
        )

        if self.mode is PlanningMode.GLOBAL:
            if closure_satisfied and self._plan_count == 0:
                if self.defer_global_finalize:
                    self._global_finalize_requested = True
                    return PlanningDecision(
                        "SCHEDULE_GLOBAL_CATALOGUE_FINALIZE",
                        "GLOBAL closure reached; wait for later fixed-point catalogue seal",
                        observation_id,
                        normalized_trigger,
                        self._plan_count,
                    )
                self._plan_count = 1
                return PlanningDecision(
                    "CREATE_PLAN_VERSION",
                    "GLOBAL observation closure reached",
                    observation_id,
                    normalized_trigger,
                    self._plan_count,
                )
            if changed:
                self._seen_observations.add((normalized_trigger, observation_id))
            return PlanningDecision(
                "NO_ACTION",
                "GLOBAL waits for catalogue finalization"
                if self._plan_count == 0
                else "GLOBAL plan already created",
                observation_id,
                normalized_trigger,
                self._plan_count,
            )

        key = (normalized_trigger, observation_id)
        if not changed or key in self._seen_observations:
            return PlanningDecision(
                "NO_ACTION",
                "duplicate or non-changing observation",
                observation_id,
                normalized_trigger,
                self._plan_count,
            )
        self._seen_observations.add(key)

        if self.mode is PlanningMode.EVENT:
            if (
                not self._delegated_event_filter
                and normalized_trigger not in self.event_triggers
            ):
                return PlanningDecision(
                    "NO_ACTION",
                    "EVENT trigger is not enabled by the run manifest",
                    observation_id,
                    normalized_trigger,
                    self._plan_count,
                )
            if self.max_event_plans is not None and self._plan_count >= self.max_event_plans:
                return PlanningDecision(
                    "NO_ACTION",
                    "EVENT plan bound reached",
                    observation_id,
                    normalized_trigger,
                    self._plan_count,
                )
            self._plan_count += 1
            return PlanningDecision(
                "CREATE_PLAN_VERSION",
                "EVENT configured observation changed",
                observation_id,
                normalized_trigger,
                self._plan_count,
            )

        raise AssertionError(f"unsupported planning mode {self.mode}")

    def on_global_catalogue_finalized(self, *, seal_digest: str) -> PlanningDecision:
        if self.mode is not PlanningMode.GLOBAL:
            raise ValueError("catalogue finalization is valid only in GLOBAL mode")
        if not self.defer_global_finalize:
            raise ValueError("this GLOBAL gate does not use deferred finalization")
        if not self._global_finalize_requested:
            raise ValueError("GLOBAL finalization was not requested by closure truth")
        if not isinstance(seal_digest, str) or not seal_digest:
            raise ValueError("seal_digest must be non-empty")
        if self._plan_count:
            if self._global_seal_digest != seal_digest:
                raise ValueError("GLOBAL phase was finalized with conflicting seal digest")
            return PlanningDecision(
                "NO_ACTION",
                "GLOBAL plan already created from this sealed catalogue",
                trigger=PlanningTrigger.OBSERVATION_CLOSURE,
                plan_count=self._plan_count,
            )
        self._global_seal_digest = str(seal_digest)
        self._plan_count = 1
        return PlanningDecision(
            "CREATE_PLAN_VERSION",
            "GLOBAL sealed catalogue finalized in a later fixed-point round",
            trigger=PlanningTrigger.OBSERVATION_CLOSURE,
            plan_count=self._plan_count,
        )

    def on_resource_release(self) -> PlanningDecision:
        return PlanningDecision(
            "STABILIZE_EXECUTION",
            "resource release retries compiler without ControlLine replan",
            plan_count=self._plan_count,
        )
