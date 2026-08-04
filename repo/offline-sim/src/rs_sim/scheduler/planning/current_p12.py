from __future__ import annotations

"""Current P12 formal semantics.

The paper-facing runtime has one execution line, ``CURRENT_P12``:

``P0_l truth -> derive P1_l -> predict P2_l -> plan P1_l + P2_l``.

P0 is an observation/trigger phase.  The physical planning window owns exactly
``Combine_l`` (P1) and ``Dispatch_{l+1}`` (P2), so every physical ``PhaseKey``
continues to have one and only one execution authority.  The first Dispatch is
an explicit bootstrap phase and the final Combine is an explicit tail phase.
"""

import enum
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rs_sim.contracts.schema import PhaseKey, PhaseKind, WindowKey

from rs_sim.scheduler.stable import stable_digest
from rs_sim.scheduler.prediction.fate_p2 import FateP2Artifact


class P12InformationMode(str, enum.Enum):
    FATE_P2 = "FATE_P2"
    PERFECT_P2 = "PERFECT_P2"
    ZERO_P2 = "ZERO_P2"


_INFORMATION_ALIASES = {
    "FATE_P2": P12InformationMode.FATE_P2,
    "FATE": P12InformationMode.FATE_P2,
    "PERFECT_P2": P12InformationMode.PERFECT_P2,
    "PERFECT": P12InformationMode.PERFECT_P2,
    "ZERO_P2": P12InformationMode.ZERO_P2,
    "ZERO": P12InformationMode.ZERO_P2,
}


def normalize_p12_information_mode(value: Any) -> P12InformationMode:
    text = getattr(value, "value", value)
    try:
        return _INFORMATION_ALIASES[str(text).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported Current P12 information mode {value!r}") from exc


@dataclass(frozen=True, slots=True)
class P12PredictionEvidenceState:
    """Normalized prediction evidence shared by formal and metric paths.

    ``ZERO_P2`` is a valid no-information ablation, not a generated or consumed
    prediction.  ``FATE_P2`` must be non-empty and bind at least one real P2 task
    whenever the target window contains remote P2 work.
    """

    information_mode: str
    generated: bool
    validated: bool
    nonempty: bool
    consumed: bool
    fallback: bool
    fallback_reason: str | None


def evaluate_p12_prediction_evidence(
    *,
    information_mode: P12InformationMode | str,
    matrix: Iterable[Iterable[int]] | None,
    rank_count: int,
    plan_materialized: bool,
    algorithm_core_run_count: int,
    bound_task_count: int,
    expected_target_task_count: int,
    fallback_reason: str | None = None,
) -> P12PredictionEvidenceState:
    """Evaluate generated/non-empty/consumed prediction evidence once.

    The function intentionally contains no scheduler or transport behavior.  It
    only normalizes evidence semantics so the formal runtime and lightweight
    metric collector cannot silently disagree.
    """

    information_text = str(getattr(information_mode, "value", information_mode)).upper()
    non_predictive_modes = {
        "NO_P2_INFORMATION_PHASE_LOCAL",
        "EXACT_P2_PHASE_LOCAL_PLANNING_RANK_LOCAL_RELEASE",
        "EXACT_TASK_CATALOGUE_NON_PREDICTIVE",
    }
    if information_text in non_predictive_modes:
        return P12PredictionEvidenceState(
            information_mode=information_text,
            generated=False,
            validated=True,
            nonempty=False,
            consumed=False,
            fallback=False,
            fallback_reason=None,
        )

    mode = normalize_p12_information_mode(information_text)
    rows = (
        None
        if matrix is None
        else tuple(tuple(int(value) for value in row) for row in matrix)
    )
    rank_count = int(rank_count)
    if rank_count <= 0:
        raise ValueError("rank_count must be positive")

    if mode is P12InformationMode.ZERO_P2:
        return P12PredictionEvidenceState(
            information_mode=mode.value,
            generated=False,
            validated=True,
            nonempty=False,
            consumed=False,
            fallback=False,
            fallback_reason=None,
        )

    generated = rows is not None
    validated = bool(
        generated
        and len(rows or ()) == rank_count
        and all(len(row) == rank_count for row in (rows or ()))
        and all(value >= 0 for row in (rows or ()) for value in row)
    )
    nonempty = bool(
        validated
        and any(
            int((rows or ())[src][dst]) > 0
            for src in range(rank_count)
            for dst in range(rank_count)
            if src != dst
        )
    )

    predictive_mode = mode is P12InformationMode.FATE_P2
    target_work_exists = int(expected_target_task_count) > 0
    consumed = bool(
        generated
        and validated
        and bool(plan_materialized)
        and int(algorithm_core_run_count) > 0
        and (
            not predictive_mode
            or (nonempty and int(bound_task_count) > 0)
        )
    )

    reason = str(fallback_reason).strip() if fallback_reason else None
    if predictive_mode and target_work_exists and reason is None:
        if not generated:
            reason = "PREDICTION_NOT_GENERATED"
        elif not validated:
            reason = "PREDICTION_INVALID"
        elif not nonempty:
            reason = "PREDICTION_EMPTY"
        elif (
            bool(plan_materialized)
            and int(algorithm_core_run_count) > 0
            and int(bound_task_count) <= 0
        ):
            reason = "PREDICTION_NOT_BOUND"

    return P12PredictionEvidenceState(
        information_mode=mode.value,
        generated=generated,
        validated=validated,
        nonempty=nonempty,
        consumed=consumed,
        fallback=bool(reason) if predictive_mode else False,
        fallback_reason=reason if predictive_mode else None,
    )


@dataclass(frozen=True, slots=True)
class CurrentP12Window:
    window_key: WindowKey
    anchor_layer_id: int
    p0_trigger_phase_key: PhaseKey
    p1_combine_phase_key: PhaseKey
    p2_dispatch_phase_key: PhaseKey
    planning_window_digest: str

    def __post_init__(self) -> None:
        if self.anchor_layer_id < 0:
            raise ValueError("anchor_layer_id must be non-negative")
        if self.p0_trigger_phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("P0 trigger must be Dispatch")
        if self.p1_combine_phase_key.phase_kind is not PhaseKind.COMBINE:
            raise ValueError("P1 must be Combine")
        if self.p2_dispatch_phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("P2 must be next Dispatch")
        if self.p0_trigger_phase_key.layer_index != self.anchor_layer_id:
            raise ValueError("P0 layer does not match anchor")
        if self.p1_combine_phase_key.layer_index != self.anchor_layer_id:
            raise ValueError("P1 layer does not match anchor")
        if self.p2_dispatch_phase_key.layer_index != self.anchor_layer_id + 1:
            raise ValueError("P2 must be next-layer Dispatch")

    @property
    def referenced_phase_keys(self) -> tuple[PhaseKey, PhaseKey]:
        return (self.p1_combine_phase_key, self.p2_dispatch_phase_key)

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        sample_id: str,
        anchor_layer_id: int,
        window_index: int,
    ) -> "CurrentP12Window":
        anchor = int(anchor_layer_id)
        p0 = PhaseKey(str(run_id), str(sample_id), anchor, PhaseKind.DISPATCH)
        p1 = PhaseKey(str(run_id), str(sample_id), anchor, PhaseKind.COMBINE)
        p2 = PhaseKey(str(run_id), str(sample_id), anchor + 1, PhaseKind.DISPATCH)
        key = WindowKey(str(run_id), str(sample_id), int(window_index))
        semantic = {
            "schema_version": "CURRENT_P12_WINDOW",
            "window_key": key,
            "anchor_layer_id": anchor,
            "p0_trigger_phase_key": p0,
            "p1_combine_phase_key": p1,
            "p2_dispatch_phase_key": p2,
        }
        return cls(
            window_key=key,
            anchor_layer_id=anchor,
            p0_trigger_phase_key=p0,
            p1_combine_phase_key=p1,
            p2_dispatch_phase_key=p2,
            planning_window_digest=stable_digest(semantic),
        )




@dataclass(frozen=True, slots=True)
class PredictedP2Slot:
    """One immutable predicted P2 task boundary prepared at P0 time.

    Slots are advisory planning objects only. They never enter shared schema, transport, or backend, never
    create ReceiveExpectation/Permit state, and are bound to real canonical P2
    tasks only after exact descriptors arrive.
    """

    slot_id: str
    phase_token: str
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    slot_digest: str

    @property
    def exact_boundary_key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.src_rank),
            int(self.dst_rank),
            int(self.chunk_index),
            int(self.byte_offset),
            int(self.payload_bytes),
        )

    @property
    def edge_key(self) -> tuple[int, int]:
        return (int(self.src_rank), int(self.dst_rank))


@dataclass(frozen=True, slots=True)
class PreparedP12PlanTemplate:
    planning_window_digest: str
    algorithm_id: str
    ordered_tokens: tuple[str, ...]
    p1_task_ids: tuple[str, ...]
    p2_slots: tuple[PredictedP2Slot, ...]
    created_at_ns: int
    algorithm_plan_digest: str
    template_digest: str
    ordered_waves: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        expected = set(self.p1_task_ids) | {item.slot_id for item in self.p2_slots}
        if len(self.ordered_tokens) != len(expected) or set(self.ordered_tokens) != expected:
            raise ValueError("PreparedP12PlanTemplate must order every P1 task and P2 slot exactly once")
        waves = tuple(tuple(str(token) for token in wave) for wave in self.ordered_waves)
        if not waves:
            waves = tuple((token,) for token in self.ordered_tokens)
            object.__setattr__(self, "ordered_waves", waves)
        flattened = tuple(token for wave in waves for token in wave)
        if flattened != self.ordered_tokens:
            raise ValueError("PreparedP12PlanTemplate wave flattening must equal ordered_tokens")
        if any(not wave for wave in waves):
            raise ValueError("PreparedP12PlanTemplate cannot contain an empty wave")

    @property
    def slot_by_id(self) -> Mapping[str, PredictedP2Slot]:
        return {item.slot_id: item for item in self.p2_slots}


def build_predicted_p2_slots(
    *,
    planning_window_digest: str,
    p2_phase_token: str,
    predicted_matrix: Iterable[Iterable[int]],
    chunk_bytes: int,
) -> tuple[PredictedP2Slot, ...]:
    step = int(chunk_bytes)
    if step <= 0:
        raise ValueError("chunk_bytes must be positive")
    result: list[PredictedP2Slot] = []
    for src_rank, row in enumerate(predicted_matrix):
        for dst_rank, raw_bytes in enumerate(row):
            total = int(raw_bytes)
            if src_rank == dst_rank or total <= 0:
                continue
            offset = 0
            chunk_index = 0
            while offset < total:
                payload_bytes = min(step, total - offset)
                semantic = {
                    "schema_version": "FATE_P2_SLOT",
                    "planning_window_digest": str(planning_window_digest),
                    "phase_token": str(p2_phase_token),
                    "src_rank": int(src_rank),
                    "dst_rank": int(dst_rank),
                    "chunk_index": int(chunk_index),
                    "byte_offset": int(offset),
                    "payload_bytes": int(payload_bytes),
                }
                digest = stable_digest(semantic)
                result.append(
                    PredictedP2Slot(
                        slot_id=f"p2slot:{digest}",
                        phase_token=str(p2_phase_token),
                        src_rank=int(src_rank),
                        dst_rank=int(dst_rank),
                        chunk_index=int(chunk_index),
                        byte_offset=int(offset),
                        payload_bytes=int(payload_bytes),
                        slot_digest=digest,
                    )
                )
                offset += payload_bytes
                chunk_index += 1
    return tuple(result)


@dataclass(frozen=True, slots=True)
class P2Prediction:
    mode: P12InformationMode
    matrix: tuple[tuple[int, ...], ...]
    provenance: str
    confidence_ppm: int | None
    confidence_provenance: str
    source_p0_workload_digest: str
    target_payload_spec_digest: str
    prediction_digest: str


def _matrix_from_rows(rows: Iterable[Iterable[int]], payload_spec: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(payload_spec.edge_payload_bytes(int(value))) for value in row)
        for row in rows
    )


def build_p2_prediction(
    *,
    current_trace_window: Any,
    next_trace_window: Any,
    information_mode: P12InformationMode | str,
) -> P2Prediction:
    """Build the causal P2 advisory used by Current P12.

    ``FATE_P2`` consumes a validated cross-layer gate artifact embedded in
    the current P0 trace metadata. It never falls back to last-value routing.
    ``PERFECT_P2`` is the explicit upper-bound ablation and ``ZERO_P2`` is the
    no-cross-layer-information ablation.
    """

    mode = normalize_p12_information_mode(information_mode)
    world_size = int(current_trace_window.mapping.world_size)
    if int(next_trace_window.mapping.world_size) != world_size:
        raise ValueError("Current and next trace windows have different world_size")
    if mode is P12InformationMode.PERFECT_P2:
        matrix = tuple(
            tuple(int(value) for value in row)
            for row in next_trace_window.payload_matrix("DISPATCH")
        )
        provenance = "PERFECT_NEXT_DISPATCH_TRUTH_UPPER_BOUND"
        confidence_ppm = 1_000_000
        confidence_provenance = "EXACT_UPPER_BOUND"
    elif mode is P12InformationMode.ZERO_P2:
        matrix = tuple(tuple(0 for _ in range(world_size)) for _ in range(world_size))
        provenance = "ZERO_P2_ABLATION"
        confidence_ppm = 0
        confidence_provenance = "ZERO_INFORMATION_ABLATION"
    elif mode is P12InformationMode.FATE_P2:
        artifact = FateP2Artifact.from_metadata(
            current_trace_window.metadata,
            world_size=world_size,
            source_layer_id=int(current_trace_window.layer_id),
            target_layer_id=int(next_trace_window.layer_id),
        )
        matrix = artifact.payload_bytes(next_trace_window.dispatch_payload_spec)
        provenance = (
            f"FATE_ARTIFACT:{artifact.predictor_id}:{artifact.estimator_kind}:"
            f"{artifact.artifact_digest}"
        )
        confidence_ppm = artifact.confidence_ppm
        confidence_provenance = (
            "EXTERNAL_FATE_ARTIFACT_CALIBRATED"
            if artifact.confidence_ppm > 0
            else "EXTERNAL_FATE_ARTIFACT_UNCALIBRATED"
        )
    else:  # pragma: no cover - normalize_p12_information_mode is exhaustive
        raise AssertionError(f"unhandled P2 information mode: {mode}")
    semantic = {
        "schema_version": "P2_PREDICTION",
        "mode": mode.value,
        "matrix": matrix,
        "provenance": provenance,
        "confidence_ppm": confidence_ppm,
        "confidence_provenance": confidence_provenance,
        "source_p0_workload_digest": current_trace_window.workload_identity_digest(),
        "target_payload_spec_digest": next_trace_window.dispatch_payload_spec.digest(),
    }
    return P2Prediction(
        mode=mode,
        matrix=matrix,
        provenance=provenance,
        confidence_ppm=confidence_ppm,
        confidence_provenance=confidence_provenance,
        source_p0_workload_digest=semantic["source_p0_workload_digest"],
        target_payload_spec_digest=semantic["target_payload_spec_digest"],
        prediction_digest=stable_digest(semantic),
    )


def build_current_p12_windows(*, fixture_input: Any, run_id: str) -> tuple[CurrentP12Window, ...]:
    windows = tuple(fixture_input.windows)
    if len(windows) < 2:
        return ()
    first = windows[0]
    sample_id = f"{first.request_id}:step{first.decode_step}"
    result: list[CurrentP12Window] = []
    for index, current in enumerate(windows[:-1]):
        nxt = windows[index + 1]
        if (
            nxt.request_id != current.request_id
            or nxt.decode_step != current.decode_step
            or int(nxt.layer_id) != int(current.layer_id) + 1
        ):
            continue
        result.append(
            CurrentP12Window.build(
                run_id=str(run_id),
                sample_id=sample_id,
                anchor_layer_id=int(current.layer_id),
                window_index=index,
            )
        )
    return tuple(result)


__all__ = [
    "CurrentP12Window",
    "PredictedP2Slot",
    "PreparedP12PlanTemplate",
    "P12InformationMode",
    "P12PredictionEvidenceState",
    "P2Prediction",
    "build_current_p12_windows",
    "build_predicted_p2_slots",
    "build_p2_prediction",
    "evaluate_p12_prediction_evidence",
    "normalize_p12_information_mode",
]
