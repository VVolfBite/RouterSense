from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.scheduler.errors import BindingError
from rs_sim.scheduler.planning.schema_api import SharedSchemaAdapter
from rs_sim.scheduler.stable import stable_digest, stable_json


@dataclass(frozen=True)
class PreparedSlot:
    phase_token: str
    src_rank: int
    dst_rank: int
    slot_ordinal: int


@dataclass(frozen=True)
class PreparedOrderTemplate:
    template_id: str
    target_phase_token: str
    slots: tuple[PreparedSlot, ...]
    prediction_digest: str
    created_at_ns: int

    @classmethod
    def build(
        cls,
        *,
        adapter: SharedSchemaAdapter,
        target_phase_key: Any,
        predicted_edges: Iterable[tuple[int, int]],
        prediction_digest: str,
        created_at_ns: int,
    ) -> "PreparedOrderTemplate":
        phase_token = stable_json(adapter.phase_payload(target_phase_key))
        slots = tuple(
            PreparedSlot(
                phase_token=phase_token,
                src_rank=int(src_rank),
                dst_rank=int(dst_rank),
                slot_ordinal=int(index),
            )
            for index, (src_rank, dst_rank) in enumerate(predicted_edges)
        )
        template_id = "prepared:" + stable_digest(
            {
                "phase_token": phase_token,
                "slots": slots,
                "prediction_digest": str(prediction_digest),
                "created_at_ns": int(created_at_ns),
            }
        )[:24]
        return cls(
            template_id=template_id,
            target_phase_token=phase_token,
            slots=slots,
            prediction_digest=str(prediction_digest),
            created_at_ns=int(created_at_ns),
        )


@dataclass(frozen=True)
class BindingReport:
    prepared_template_id: str
    bound_task_ids: tuple[str, ...]
    predicted_nonzero_but_real_zero: tuple[tuple[int, int], ...]
    predicted_zero_but_real_nonzero: tuple[tuple[int, int], ...]
    ignored_predicted_slots: tuple[int, ...]
    unmatched_real_task_ids: tuple[str, ...]
    binding_digest: str


class PlanBinder:
    """Deterministic projection from predicted edge slots to real tasks."""

    def __init__(self, *, adapter: SharedSchemaAdapter) -> None:
        self.adapter = adapter

    def bind(self, template: PreparedOrderTemplate, tasks: Iterable[Any]) -> BindingReport:
        views = [self.adapter.task_view(task) for task in tasks]
        by_edge: dict[tuple[str, int, int], list[Any]] = {}
        for view in views:
            phase_token = stable_json(self.adapter.phase_payload(view.phase_key))
            if phase_token != template.target_phase_token:
                raise BindingError("real task phase does not match prepared target phase")
            key = (phase_token, int(view.src_rank), int(view.dst_rank))
            by_edge.setdefault(key, []).append(view)
        for edge_views in by_edge.values():
            edge_views.sort(key=lambda item: (item.chunk_index, item.byte_offset, item.task_id))

        bound: list[str] = []
        seen: set[str] = set()
        ignored_slots: list[int] = []
        predicted_edges: set[tuple[int, int]] = set()
        missing_predicted_edges: list[tuple[int, int]] = []
        consumed_edges: set[tuple[str, int, int]] = set()
        for slot in template.slots:
            edge = (int(slot.src_rank), int(slot.dst_rank))
            predicted_edges.add(edge)
            key = (slot.phase_token, edge[0], edge[1])
            edge_views = by_edge.get(key, [])
            if not edge_views:
                missing_predicted_edges.append(edge)
                ignored_slots.append(int(slot.slot_ordinal))
                continue
            if key in consumed_edges:
                ignored_slots.append(int(slot.slot_ordinal))
                continue
            consumed_edges.add(key)
            for view in edge_views:
                if view.task_id in seen:
                    raise BindingError(f"task {view.task_id} would be bound twice")
                seen.add(view.task_id)
                bound.append(view.task_id)

        unmatched_views = [view for view in views if view.task_id not in seen]
        unmatched_views.sort(
            key=lambda item: (
                stable_json(self.adapter.phase_payload(item.phase_key)),
                item.src_rank,
                item.dst_rank,
                item.chunk_index,
                item.byte_offset,
                item.task_id,
            )
        )
        unmatched_ids = tuple(view.task_id for view in unmatched_views)
        bound.extend(unmatched_ids)
        if len(bound) != len(views) or len(set(bound)) != len(bound):
            raise BindingError("binding must cover each real task exactly once")

        real_edges = {(view.src_rank, view.dst_rank) for view in views}
        new_real_edges = sorted(real_edges - predicted_edges)
        missing_predicted = tuple(sorted(set(missing_predicted_edges)))
        digest = stable_digest(
            {
                "template_id": template.template_id,
                "bound_task_ids": tuple(bound),
                "predicted_nonzero_but_real_zero": missing_predicted,
                "predicted_zero_but_real_nonzero": tuple(new_real_edges),
                "ignored_predicted_slots": tuple(ignored_slots),
                "unmatched_real_task_ids": unmatched_ids,
            }
        )
        return BindingReport(
            prepared_template_id=template.template_id,
            bound_task_ids=tuple(bound),
            predicted_nonzero_but_real_zero=missing_predicted,
            predicted_zero_but_real_nonzero=tuple(new_real_edges),
            ignored_predicted_slots=tuple(ignored_slots),
            unmatched_real_task_ids=unmatched_ids,
            binding_digest=digest,
        )

