from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim.scheduler.execution.authority import PhaseAuthorityManager
from rs_sim.scheduler.planning.catalogue import TaskCatalogue
from rs_sim.scheduler.execution.compiler import BatchCompiler, BatchValidator, EndpointConflictResourceAdapter
from rs_sim.scheduler.execution.controller import SchedulingController
from rs_sim.scheduler.planning.schema_api import DataclassSchemaAdapter, SharedSchemaConstructors
from rs_sim.scheduler.execution.state import TaskRuntimeIndex
from rs_sim.scheduler.planning.taskization import CanonicalTaskizer, TaskizationSpec
from rs_sim import (
    CanonicalTransferTask,
    PhaseExecutionRecord,
    PlanStatus,
    PlanVersion,
)


@dataclass(frozen=True)
class Snapshot:
    max_batch_tasks: int = 4
    busy_src_ranks: tuple[int, ...] = ()
    busy_dst_ranks: tuple[int, ...] = ()
    topology_digest: str = "CONTRACT_STUB_TOPOLOGY"


@dataclass
class Stack:
    adapter: DataclassSchemaAdapter
    taskizer: CanonicalTaskizer
    catalogue: TaskCatalogue
    runtime: TaskRuntimeIndex
    authority: PhaseAuthorityManager
    resources: EndpointConflictResourceAdapter
    compiler: BatchCompiler
    validator: BatchValidator
    controller: SchedulingController


def build_stack(*, chunk_bytes: int = 64, alignment_bytes: int = 16) -> Stack:
    adapter = DataclassSchemaAdapter(
        SharedSchemaConstructors(
            canonical_task=CanonicalTransferTask,
            phase_execution_record=PhaseExecutionRecord,
            plan_version=PlanVersion,
            plan_status=lambda name: PlanStatus[str(name)],
        )
    )
    taskizer = CanonicalTaskizer(
        adapter=adapter,
        spec=TaskizationSpec(chunk_bytes=chunk_bytes, alignment_bytes=alignment_bytes),
    )
    catalogue = TaskCatalogue(adapter=adapter, taskizer=taskizer)
    runtime = TaskRuntimeIndex(catalogue=catalogue)
    authority = PhaseAuthorityManager(
        adapter=adapter, catalogue=catalogue, runtime=runtime
    )
    resources = EndpointConflictResourceAdapter()
    compiler = BatchCompiler(
        catalogue=catalogue,
        runtime=runtime,
        authority=authority,
        resources=resources,
    )
    validator = BatchValidator(
        catalogue=catalogue,
        runtime=runtime,
        authority=authority,
        resources=resources,
    )
    controller = SchedulingController(
        catalogue=catalogue,
        runtime=runtime,
        authority=authority,
        compiler=compiler,
        validator=validator,
    )
    return Stack(
        adapter=adapter,
        taskizer=taskizer,
        catalogue=catalogue,
        runtime=runtime,
        authority=authority,
        resources=resources,
        compiler=compiler,
        validator=validator,
        controller=controller,
    )


@pytest.fixture
def stack() -> Stack:
    return build_stack()
