from __future__ import annotations

"""Production bindings between shared-schema immutable contracts and backend/scheduler ports."""

from dataclasses import dataclass
from typing import Any

from rs_sim import (
    CanonicalTransferTask,
    EdgeKey,
    ExpectationOrigin,
    PhaseExecutionRecord,
    PhaseKind,
    PlanStatus,
    PlanVersion,
    ReceiveExpectation,
    ReceivePermit,
)
from rs_sim.backend import AttributeSharedObjectAdapter, CallablePhaseSemantics
from rs_sim.scheduler import (
    BatchCompiler,
    BatchValidator,
    CanonicalTaskizer,
    DataclassSchemaAdapter,
    PhaseAuthorityManager,
    SchedulingController,
    SharedSchemaConstructors,
    TaskCatalogue,
    TaskRuntimeIndex,
    TaskizationSpec,
)
from rs_sim.contracts.digest import stable_digest, stable_json_dumps


class SchemaEdgeKeyFactory:
    def make_edge_key(self, *, phase_key: Any, src_rank: int, dst_rank: int) -> EdgeKey:
        return EdgeKey(
            phase_key=phase_key,
            src_rank=int(src_rank),
            dst_rank=int(dst_rank),
        )


class SchemaExpectationFactory:
    _ORIGIN_MAP = {
        "DELIVERED_DISPATCH_DESCRIPTOR": ExpectationOrigin.DISPATCH_DESCRIPTOR,
        "DISPATCH_DESCRIPTOR": ExpectationOrigin.DISPATCH_DESCRIPTOR,
        "REALIZED_DISPATCH_TRANSPOSE": ExpectationOrigin.COMBINE_REALIZED,
        "COMBINE_REALIZED": ExpectationOrigin.COMBINE_REALIZED,
    }

    def create_receive_expectation(self, **fields: Any) -> ReceiveExpectation:
        raw_origin = str(fields.pop("origin"))
        try:
            origin = self._ORIGIN_MAP[raw_origin]
        except KeyError as exc:
            raise ValueError(f"unsupported backend expectation origin {raw_origin!r}") from exc
        return ReceiveExpectation(origin=origin, **fields)


class SchemaPermitFactory:
    def create_receive_permit(self, **fields: Any) -> ReceivePermit:
        return ReceivePermit(**fields)


def make_phase_semantics() -> CallablePhaseSemantics:
    return CallablePhaseSemantics(
        phase_kind=lambda phase: phase.phase_kind.value,
        phase_sort_key=lambda phase: stable_json_dumps(phase),
    )


def make_schema_adapter() -> DataclassSchemaAdapter:
    def parse_plan_status(value: str) -> PlanStatus:
        token = str(value).rsplit(".", 1)[-1]
        return PlanStatus[token]

    return DataclassSchemaAdapter(
        SharedSchemaConstructors(
            canonical_task=CanonicalTransferTask,
            phase_execution_record=PhaseExecutionRecord,
            plan_version=PlanVersion,
            plan_status=parse_plan_status,
        )
    )


@dataclass(frozen=True, slots=True)
class SchedulingStack:
    adapter: DataclassSchemaAdapter
    taskizer: CanonicalTaskizer
    catalogue: TaskCatalogue
    runtime: TaskRuntimeIndex
    authority: PhaseAuthorityManager
    compiler: BatchCompiler
    validator: BatchValidator
    controller: SchedulingController


def build_scheduling_stack(
    *,
    taskization_spec: TaskizationSpec,
    resources: Any | None = None,
) -> SchedulingStack:
    adapter = make_schema_adapter()
    taskizer = CanonicalTaskizer(adapter=adapter, spec=taskization_spec)
    catalogue = TaskCatalogue(adapter=adapter, taskizer=taskizer)
    runtime = TaskRuntimeIndex(catalogue=catalogue)
    authority = PhaseAuthorityManager(
        adapter=adapter,
        catalogue=catalogue,
        runtime=runtime,
    )
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
    return SchedulingStack(
        adapter=adapter,
        taskizer=taskizer,
        catalogue=catalogue,
        runtime=runtime,
        authority=authority,
        compiler=compiler,
        validator=validator,
        controller=controller,
    )


def shared_binding_digest() -> str:
    objects = (
        CanonicalTransferTask,
        EdgeKey,
        PhaseExecutionRecord,
        PlanVersion,
        ReceiveExpectation,
        ReceivePermit,
        PhaseKind,
        PlanStatus,
    )
    payload = tuple(
        {
            "qualified_name": f"{obj.__module__}.{obj.__qualname__}",
            "fields": tuple(getattr(obj, "__dataclass_fields__", {}).keys()),
            "members": tuple(member.name for member in obj) if hasattr(obj, "__members__") else (),
        }
        for obj in objects
    )
    return stable_digest(payload, domain="RS_SIM_RUNTIME_BINDINGS")


__all__ = [
    "AttributeSharedObjectAdapter",
    "SchemaEdgeKeyFactory",
    "SchemaExpectationFactory",
    "SchemaPermitFactory",
    "SchedulingStack",
    "build_scheduling_stack",
    "make_phase_semantics",
    "make_schema_adapter",
    "shared_binding_digest",
]
