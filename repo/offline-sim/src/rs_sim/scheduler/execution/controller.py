from __future__ import annotations

from typing import Any, Iterable

from rs_sim.scheduler.execution.authority import PhaseAuthorityManager
from rs_sim.scheduler.planning.catalogue import TaskCatalogue
from rs_sim.scheduler.execution.compiler import BatchCompiler, BatchValidator, ExecutionStabilizer
from rs_sim.scheduler.execution.state import TaskRuntimeIndex


class SchedulingController:
    """Own canonical scheduler state and activate explicit algorithm plans.

    Algorithm selection never occurs here.  A registered core and its outer
    decorators must produce the complete order before the controller is called.
    """

    def __init__(
        self,
        *,
        catalogue: TaskCatalogue,
        runtime: TaskRuntimeIndex,
        authority: PhaseAuthorityManager,
        compiler: BatchCompiler,
        validator: BatchValidator,
    ) -> None:
        self.catalogue = catalogue
        self.runtime = runtime
        self.authority = authority
        self.compiler = compiler
        self.validator = validator
        self.stabilizer = ExecutionStabilizer(
            compiler=compiler, validator=validator, authority=authority
        )

    def register_expectation(self, expectation: Any, *, registered_at_ns: int) -> tuple[Any, ...]:
        tasks = self.catalogue.register_expectation(
            expectation, registered_at_ns=int(registered_at_ns)
        )
        self.runtime.register_catalogue(tasks)
        return tasks

    def note_receive_permit(self, task_id: str, *, at_ns: int) -> None:
        self.runtime.note_permit(task_id, at_ns=int(at_ns))

    def note_source_payload_ready(self, task_id: str, *, at_ns: int) -> None:
        self.runtime.note_source_payload_ready(task_id, at_ns=int(at_ns))

    def activate_plan(
        self,
        *,
        phase_key: Any,
        window_key: Any,
        ordered_task_ids: Iterable[str],
        now_ns: int,
    ) -> Any:
        order = tuple(str(item) for item in ordered_task_ids)
        return self.authority.create_and_activate(
            phase_key=phase_key,
            window_key=window_key,
            ordered_task_ids=order,
            now_ns=int(now_ns),
        )
