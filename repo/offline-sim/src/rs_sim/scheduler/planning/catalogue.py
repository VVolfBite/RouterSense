from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.scheduler.errors import CatalogueSealError, TaskizationError
from rs_sim.scheduler.planning.schema_api import CanonicalTaskView, SharedSchemaAdapter
from rs_sim.scheduler.stable import stable_digest, stable_json
from rs_sim.scheduler.planning.taskization import CanonicalTaskizer




@dataclass(frozen=True, slots=True)
class PhaseCatalogueSeal:
    phase_token: str
    expected_expectation_count: int
    expected_task_count: int
    task_catalogue_digest: str
    closure_digest: str
    sealed_at_ns: int
    seal_digest: str


class TaskCatalogue:
    """Immutable semantic catalogue with idempotent edge registration."""

    def __init__(self, *, adapter: SharedSchemaAdapter, taskizer: CanonicalTaskizer) -> None:
        self.adapter = adapter
        self.taskizer = taskizer
        self._tasks_by_id: dict[str, Any] = {}
        self._views_by_id: dict[str, CanonicalTaskView] = {}
        self._task_ordinals: dict[str, int] = {}
        self._task_ids_by_phase: dict[str, list[str]] = defaultdict(list)
        self._task_ids_by_edge: dict[str, tuple[str, ...]] = {}
        self._edge_registration_payload: dict[str, dict[str, Any]] = {}
        self._expectation_tokens_by_phase: dict[str, list[str]] = defaultdict(list)
        self._sealed_phases: dict[str, PhaseCatalogueSeal] = {}
        self._phase_digest_cache: dict[str, str] = {}
        self._phase_revision_by_token: dict[str, int] = defaultdict(int)
        self._all_digest_cache: str | None = None
        self._phase_token_cache: dict[int, tuple[Any, str]] = {}
        self._edge_token_cache: dict[int, tuple[Any, str]] = {}

    @staticmethod
    def _identity_cached_token(
        cache: dict[int, tuple[Any, str]], value: Any, payload: Any
    ) -> str:
        cache_key = id(value)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        token = stable_json(payload)
        cache[cache_key] = (value, token)
        return token

    def _phase_token(self, phase_key: Any) -> str:
        return self._identity_cached_token(
            self._phase_token_cache, phase_key, self.adapter.phase_payload(phase_key)
        )

    def _edge_token(self, edge_key: Any) -> str:
        return self._identity_cached_token(
            self._edge_token_cache, edge_key, self.adapter.edge_payload(edge_key)
        )

    def register_expectation(self, expectation: Any, *, registered_at_ns: int) -> tuple[Any, ...]:
        expectation_view = self.adapter.expectation_view(expectation)
        phase_token = self._phase_token(expectation_view.phase_key)
        if phase_token in self._sealed_phases:
            raise CatalogueSealError(
                "late canonical task registration after GLOBAL catalogue seal"
            )
        edge_token = self._edge_token(expectation_view.edge_key)
        registration_payload = {
            "phase": self.adapter.phase_payload(expectation_view.phase_key),
            "edge": self.adapter.edge_payload(expectation_view.edge_key),
            "src_rank": int(expectation_view.src_rank),
            "dst_rank": int(expectation_view.dst_rank),
            "total_expected_payload_bytes": int(expectation_view.total_expected_payload_bytes),
            "expectation_digest": str(expectation_view.expectation_digest),
            "zero_edge": bool(expectation_view.zero_edge),
        }
        existing_payload = self._edge_registration_payload.get(edge_token)
        if existing_payload is not None:
            if existing_payload != registration_payload:
                raise TaskizationError("same EdgeKey was registered with conflicting expectation semantics")
            return tuple(self._tasks_by_id[task_id] for task_id in self._task_ids_by_edge[edge_token])

        tasks = self.taskizer.taskize(expectation, registered_at_ns=int(registered_at_ns))
        task_ids: list[str] = []
        for task in tasks:
            view = self.adapter.task_view(task)
            if view.task_id in self._tasks_by_id:
                raise TaskizationError(f"duplicate canonical task ID {view.task_id}")
            self._task_ordinals[view.task_id] = len(self._task_ordinals)
            self._tasks_by_id[view.task_id] = task
            self._views_by_id[view.task_id] = view
            self._task_ids_by_phase[phase_token].append(view.task_id)
            task_ids.append(view.task_id)
        self._task_ids_by_edge[edge_token] = tuple(task_ids)
        self._edge_registration_payload[edge_token] = registration_payload
        self._expectation_tokens_by_phase[phase_token].append(edge_token)
        self._phase_revision_by_token[phase_token] += 1
        self._phase_digest_cache.pop(phase_token, None)
        self._all_digest_cache = None
        self.validate_phase(expectation_view.phase_key)
        return tasks


    def phase_revision(self, phase_key: Any) -> int:
        """Monotonic in-memory catalogue revision for hot-path invalidation."""

        return int(self._phase_revision_by_token.get(self._phase_token(phase_key), 0))

    def registered_expectation_count(self, phase_key: Any) -> int:
        return len(self._expectation_tokens_by_phase.get(self._phase_token(phase_key), ()))

    def phase_snapshot(self, phase_key: Any) -> dict[str, Any]:
        return {
            "phase_token": self._phase_token(phase_key),
            "expectation_count": self.registered_expectation_count(phase_key),
            "task_count": len(self.task_ids_for_phase(phase_key)),
            "task_catalogue_digest": self.phase_digest(phase_key),
        }

    def is_phase_sealed(self, phase_key: Any) -> bool:
        return self._phase_token(phase_key) in self._sealed_phases

    def phase_seal(self, phase_key: Any) -> PhaseCatalogueSeal | None:
        return self._sealed_phases.get(self._phase_token(phase_key))

    def seal_phase(
        self,
        phase_key: Any,
        *,
        expected_expectation_count: int,
        expected_task_count: int,
        closure_digest: str,
        sealed_at_ns: int,
        expected_catalogue_digest: str | None = None,
    ) -> PhaseCatalogueSeal:
        phase_token = self._phase_token(phase_key)
        if not isinstance(expected_expectation_count, int) or expected_expectation_count < 0:
            raise CatalogueSealError("expected_expectation_count must be non-negative")
        if not isinstance(expected_task_count, int) or expected_task_count < 0:
            raise CatalogueSealError("expected_task_count must be non-negative")
        if not isinstance(sealed_at_ns, int) or sealed_at_ns < 0:
            raise CatalogueSealError("sealed_at_ns must be non-negative")
        if not isinstance(closure_digest, str) or not closure_digest:
            raise CatalogueSealError("closure_digest must be non-empty")
        snapshot = self.phase_snapshot(phase_key)
        if snapshot["expectation_count"] != int(expected_expectation_count):
            raise CatalogueSealError(
                "GLOBAL closure expectation count does not match registered catalogue"
            )
        if snapshot["task_count"] != int(expected_task_count):
            raise CatalogueSealError(
                "GLOBAL closure task count does not match registered catalogue"
            )
        if (
            expected_catalogue_digest is not None
            and str(expected_catalogue_digest) != snapshot["task_catalogue_digest"]
        ):
            raise CatalogueSealError(
                "GLOBAL closure catalogue digest does not match registered catalogue"
            )
        payload = {
            **snapshot,
            "expected_expectation_count": int(expected_expectation_count),
            "expected_task_count": int(expected_task_count),
            "closure_digest": str(closure_digest),
            "sealed_at_ns": int(sealed_at_ns),
        }
        seal = PhaseCatalogueSeal(
            phase_token=phase_token,
            expected_expectation_count=int(expected_expectation_count),
            expected_task_count=int(expected_task_count),
            task_catalogue_digest=str(snapshot["task_catalogue_digest"]),
            closure_digest=str(closure_digest),
            sealed_at_ns=int(sealed_at_ns),
            seal_digest=stable_digest(payload),
        )
        existing = self._sealed_phases.get(phase_token)
        if existing is not None:
            if existing != seal:
                raise CatalogueSealError("phase catalogue was sealed with conflicting truth")
            return existing
        self._sealed_phases[phase_token] = seal
        return seal

    def get(self, task_id: str) -> Any:
        try:
            return self._tasks_by_id[str(task_id)]
        except KeyError as exc:
            raise TaskizationError(f"unknown task_id {task_id}") from exc

    def view(self, task_id: str) -> CanonicalTaskView:
        key = str(task_id)
        try:
            return self._views_by_id[key]
        except KeyError as exc:
            raise TaskizationError(f"unknown task_id {task_id}") from exc

    def ordinal(self, task_id: str) -> int:
        return self._task_ordinals[str(task_id)]

    def tasks_for_phase(self, phase_key: Any) -> tuple[Any, ...]:
        ids = self._task_ids_by_phase.get(self._phase_token(phase_key), [])
        return tuple(self._tasks_by_id[task_id] for task_id in ids)

    def task_ids_for_phase(self, phase_key: Any) -> tuple[str, ...]:
        return tuple(self._task_ids_by_phase.get(self._phase_token(phase_key), []))

    def all_tasks(self) -> tuple[Any, ...]:
        ordered_ids = sorted(self._tasks_by_id, key=self._task_ordinals.__getitem__)
        return tuple(self._tasks_by_id[task_id] for task_id in ordered_ids)

    def _semantic_task_payloads(self, tasks: Iterable[Any]) -> list[dict[str, Any]]:
        payloads = [self.view(self.adapter.task_view(task).task_id).semantic_payload(self.adapter) for task in tasks]
        payloads.sort(
            key=lambda item: (
                stable_json(item["phase_key"]),
                int(item["src_rank"]),
                int(item["dst_rank"]),
                int(item["chunk_index"]),
                str(item["task_id"]),
            )
        )
        return payloads

    def digest(self) -> str:
        cached = self._all_digest_cache
        if cached is None:
            cached = stable_digest(self._semantic_task_payloads(self.all_tasks()))
            self._all_digest_cache = cached
        return cached

    def phase_digest(self, phase_key: Any) -> str:
        phase_token = self._phase_token(phase_key)
        cached = self._phase_digest_cache.get(phase_token)
        if cached is None:
            task_ids = self._task_ids_by_phase.get(phase_token, ())
            payloads = [self._views_by_id[task_id].semantic_payload(self.adapter) for task_id in task_ids]
            payloads.sort(
                key=lambda item: (
                    stable_json(item["phase_key"]),
                    int(item["src_rank"]),
                    int(item["dst_rank"]),
                    int(item["chunk_index"]),
                    str(item["task_id"]),
                )
            )
            cached = stable_digest(payloads)
            self._phase_digest_cache[phase_token] = cached
        return cached

    def validate_phase(self, phase_key: Any) -> None:
        by_edge: dict[str, list[CanonicalTaskView]] = defaultdict(list)
        for task in self.tasks_for_phase(phase_key):
            view = self.adapter.task_view(task)
            by_edge[self._edge_token(view.edge_key)].append(view)
        for views in by_edge.values():
            views.sort(key=lambda item: (item.byte_offset, item.chunk_index, item.task_id))
            cursor = 0
            taskization_digests = {view.taskization_digest for view in views}
            if len(taskization_digests) != 1:
                raise TaskizationError("one edge contains multiple taskization digests")
            for view in views:
                if view.byte_offset != cursor:
                    raise TaskizationError("edge task ranges are not contiguous")
                cursor += view.payload_bytes
            edge_token = self._edge_token(views[0].edge_key)
            expected_total = int(self._edge_registration_payload[edge_token]["total_expected_payload_bytes"])
            if cursor != expected_total:
                raise TaskizationError(
                    f"edge catalogue covers {cursor} bytes, expected {expected_total}"
                )

    def stable_payload(self) -> dict[str, Any]:
        return {
            "task_count": len(self._tasks_by_id),
            "task_catalogue_digest": self.digest(),
            "tasks": self._semantic_task_payloads(self.all_tasks()),
            "phase_seals": tuple(
                self._sealed_phases[token] for token in sorted(self._sealed_phases)
            ),
        }
