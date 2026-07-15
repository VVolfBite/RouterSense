from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from rs.core.contracts.execution import ActualPhaseContext, ExecutionOutcome
from rs.core.contracts.measurement import MeasurementEvent
from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.runtime.debug.null_probe import NullDebugProbe
from rs.runtime.measurement.api import NullMeasurementSink, PerfLightMeasurementSink
from rs.runtime.observation.instrumentation import RuntimeInstrumentation
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.execution.pipeline import RuntimeExecutionPipeline
from rs.runtime.online.megatron_ep.execution.transport_adapter import HostAPIDriftError, MegatronPhaseTransportAdapter
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


@dataclass
class _StubExecutionPlan:
    transport_mutation: bool = True
    execution_mode: str = "phase_sync_wave"


class _RecordingTargetPlanStore:
    def __init__(self) -> None:
        self.failed: list[tuple[object, str]] = []

    def fail(self, key: object, *, execution_origin: str) -> None:
        self.failed.append((key, str(execution_origin)))


class _RecordingRuntime:
    def __init__(self, *, instrumentation: RuntimeInstrumentation, target_plan_store: _RecordingTargetPlanStore | None = None) -> None:
        self.run_id = "run"
        self._forward_epoch = 0
        self.runtime_instrumentation = instrumentation
        self.target_plan_store = target_plan_store

    def _record_instrumentation_measurement(
        self,
        *,
        event_type: str,
        layer_id: str | None,
        phase: str | None,
        started_at_ns: int,
        ended_at_ns: int,
        details: dict[str, object] | None = None,
    ) -> None:
        self.runtime_instrumentation.record_measurement(
            MeasurementEvent(
                run_id=str(self.run_id),
                rank=0,
                forward_generation=int(self._forward_epoch),
                microbatch_id="mb",
                event_type=str(event_type),
                started_at_ns=int(started_at_ns),
                ended_at_ns=int(ended_at_ns),
                layer_id=None if layer_id is None else str(layer_id),
                phase=None if phase is None else str(phase),
                details=dict(details or {}),
            )
        )

    def _target_plan_key(self, *, layer_name: str) -> tuple[str, int, str]:
        return (str(self.run_id), int(self._forward_epoch), str(layer_name))


class _FailingExecutor:
    def execute(self, *, plan, invocation, context) -> ExecutionOutcome:
        submitted = tuple(
            str(item.task_id)
            for batch in plan.batches
            for item in batch.slices
            if str(item.payload_role) == str(invocation.payload_role)
        )
        return ExecutionOutcome(
            success=False,
            output_payload=None,
            submitted_task_ids=submitted,
            completed_task_ids=tuple(),
            failed_task_ids=tuple(),
            unresolved_task_ids=submitted,
            executed_batch_count=0,
            all_work_completed=False,
            failure_code="unresolved_task",
            details={"backend_id": "phase_sync"},
        )


def _build_prepared_execution(*, executor=None):
    contexts = make_contexts_from_matrix(phase="P0", matrix=((4,),), p2_hint_mode="deterministic_stub")
    context = contexts[0]
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0,), root_rank=0))
    published = publisher.build(
        publication_slot={
            "run_id": "run",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": "0->1",
        },
        window_plan=WindowPlan(
            planner_id="barrier_criticality_joint",
            planner_family="joint",
            request_digest="0->1",
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(
                        PlannedFlow(
                            flow_id="p0_0_0",
                            phase="p0_dispatch",
                            src_rank=0,
                            dst_rank=0,
                            row_count=4,
                            release_state="ready",
                            executable=True,
                        ),
                    ),
                    estimated_duration=4.0,
                ),
            ),
            metadata={"source_layer_id": "0", "target_layer_id": "1"},
        ),
    )
    actual_context = ActualPhaseContext(
        layer_id="0",
        phase="P0",
        world_size=1,
        rank_space="global",
        layout_digest=str(context.canonical_receive_layout_id),
        metadata={"phase_ready_context": context.to_dict()},
    )
    pipeline = RuntimeExecutionPipeline(executor=executor)
    prepared = pipeline.prepare(published, actual_context)
    return context, pipeline, prepared


def _run_transport_once(*, instrumentation: RuntimeInstrumentation):
    context, pipeline, prepared = _build_prepared_execution()
    return _run_transport_once_with_prepared(
        context=context,
        pipeline=pipeline,
        prepared=prepared,
        instrumentation=instrumentation,
    )


def _run_transport_once_with_prepared(*, context, pipeline, prepared, instrumentation: RuntimeInstrumentation):
    runtime = _RecordingRuntime(instrumentation=instrumentation, target_plan_store=_RecordingTargetPlanStore())
    adapter = MegatronPhaseTransportAdapter(dispatcher_class="FakeDispatcher", dispatcher_module_sha256=None)
    adapter.activate(
        layer_name="decoder.layers.0",
        phase="P0",
        context=context,
        plan=_StubExecutionPlan(),
        prepared_execution=prepared,
        execution_pipeline=pipeline,
        runtime=runtime,
    )
    hidden_spec, probs_spec = prepared.materialized_plan.payload_specs
    hidden_tensor = torch.arange(
        hidden_spec.row_count * hidden_spec.shape_suffix[0],
        dtype=torch.float16,
    ).reshape(hidden_spec.row_count, hidden_spec.shape_suffix[0])
    probs_tensor = torch.arange(
        probs_spec.row_count * probs_spec.shape_suffix[0],
        dtype=torch.float16,
    ).reshape(probs_spec.row_count, probs_spec.shape_suffix[0])
    output_hidden = adapter.maybe_execute(
        group=None,
        input_tensor=hidden_tensor,
        output_split_sizes=context.recv_splits,
        input_split_sizes=context.send_splits,
        original_all_to_all=lambda *args, **kwargs: hidden_tensor.clone(),
    )
    output_probs = adapter.maybe_execute(
        group=None,
        input_tensor=probs_tensor,
        output_split_sizes=context.recv_splits,
        input_split_sizes=context.send_splits,
        original_all_to_all=lambda *args, **kwargs: probs_tensor.clone(),
    )
    adapter.deactivate(layer_name="decoder.layers.0", phase="P0")
    return runtime, adapter, output_hidden, output_probs


def test_merged_runtime_instrumentation_off_and_perf_light_have_no_side_effects(monkeypatch) -> None:
    calls = {
        "path_open": 0,
        "path_write_text": 0,
        "json_dump": 0,
        "json_dumps": 0,
        "cpu": 0,
        "item": 0,
        "tolist": 0,
        "broadcast": 0,
        "all_gather": 0,
    }

    original_open = Path.open
    original_write_text = Path.write_text
    original_json_dump = json.dump
    original_json_dumps = json.dumps
    original_broadcast = dist.broadcast
    original_all_gather = dist.all_gather

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: calls.__setitem__("path_open", calls["path_open"] + 1) or original_open(self, *a, **k))
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: calls.__setitem__("path_write_text", calls["path_write_text"] + 1) or original_write_text(self, *a, **k))
    monkeypatch.setattr(json, "dump", lambda *a, **k: calls.__setitem__("json_dump", calls["json_dump"] + 1) or original_json_dump(*a, **k))
    monkeypatch.setattr(json, "dumps", lambda *a, **k: calls.__setitem__("json_dumps", calls["json_dumps"] + 1) or original_json_dumps(*a, **k))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: calls.__setitem__("cpu", calls["cpu"] + 1) or self)
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **k: calls.__setitem__("item", calls["item"] + 1) or 0)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: calls.__setitem__("tolist", calls["tolist"] + 1) or [])
    monkeypatch.setattr(dist, "broadcast", lambda *a, **k: calls.__setitem__("broadcast", calls["broadcast"] + 1) or original_broadcast(*a, **k))
    monkeypatch.setattr(dist, "all_gather", lambda *a, **k: calls.__setitem__("all_gather", calls["all_gather"] + 1) or original_all_gather(*a, **k))

    off_context, off_pipeline, off_prepared = _build_prepared_execution()
    perf_context, perf_pipeline, perf_prepared = _build_prepared_execution()
    calls["json_dumps"] = 0
    off_hidden_spec, off_probs_spec = off_prepared.materialized_plan.payload_specs
    perf_hidden_spec, perf_probs_spec = perf_prepared.materialized_plan.payload_specs

    off_runtime, _, off_hidden, off_probs = _run_transport_once_with_prepared(
        context=off_context,
        pipeline=off_pipeline,
        prepared=off_prepared,
        instrumentation=RuntimeInstrumentation(
            measurement_sink=NullMeasurementSink(),
            debug_probe=NullDebugProbe(),
        )
    )
    perf_sink = PerfLightMeasurementSink(max_events=8)
    perf_runtime, adapter, perf_hidden, perf_probs = _run_transport_once_with_prepared(
        context=perf_context,
        pipeline=perf_pipeline,
        prepared=perf_prepared,
        instrumentation=RuntimeInstrumentation(
            measurement_sink=perf_sink,
            debug_probe=NullDebugProbe(),
        )
    )

    assert torch.equal(
        off_hidden,
        torch.arange(
            off_hidden_spec.row_count * off_hidden_spec.shape_suffix[0],
            dtype=torch.float16,
        ).reshape(off_hidden_spec.row_count, off_hidden_spec.shape_suffix[0]),
    )
    assert torch.equal(
        off_probs,
        torch.arange(
            off_probs_spec.row_count * off_probs_spec.shape_suffix[0],
            dtype=torch.float16,
        ).reshape(off_probs_spec.row_count, off_probs_spec.shape_suffix[0]),
    )
    assert torch.equal(
        perf_hidden,
        torch.arange(
            perf_hidden_spec.row_count * perf_hidden_spec.shape_suffix[0],
            dtype=torch.float16,
        ).reshape(perf_hidden_spec.row_count, perf_hidden_spec.shape_suffix[0]),
    )
    assert torch.equal(
        perf_probs,
        torch.arange(
            perf_probs_spec.row_count * perf_probs_spec.shape_suffix[0],
            dtype=torch.float16,
        ).reshape(perf_probs_spec.row_count, perf_probs_spec.shape_suffix[0]),
    )
    assert off_runtime.runtime_instrumentation.measurement_sink.snapshot().event_count == 0
    perf_snapshot = perf_runtime.runtime_instrumentation.measurement_sink.snapshot()
    assert perf_snapshot.event_count == 2
    assert tuple(event.event_type for event in perf_snapshot.events) == ("executor_submit", "executor_submit")
    assert all(bool(event.details.get("success")) for event in perf_snapshot.events)
    assert all(bool(event.details.get("all_work_completed")) for event in perf_snapshot.events)
    assert adapter.export_results()
    assert calls == {
        "path_open": 0,
        "path_write_text": 0,
        "json_dump": 0,
        "json_dumps": 0,
        "cpu": 0,
        "item": 0,
        "tolist": 0,
        "broadcast": 0,
        "all_gather": 0,
    }


def test_runtime_transport_failure_marks_store_failed_and_records_measurement() -> None:
    context, pipeline, prepared = _build_prepared_execution(executor=_FailingExecutor())
    sink = PerfLightMeasurementSink(max_events=4)
    store = _RecordingTargetPlanStore()
    runtime = _RecordingRuntime(
        instrumentation=RuntimeInstrumentation(
            measurement_sink=sink,
            debug_probe=NullDebugProbe(),
        ),
        target_plan_store=store,
    )
    adapter = MegatronPhaseTransportAdapter(dispatcher_class="FakeDispatcher", dispatcher_module_sha256=None)
    adapter.activate(
        layer_name="decoder.layers.0",
        phase="P0",
        context=context,
        plan=_StubExecutionPlan(),
        prepared_execution=prepared,
        execution_pipeline=pipeline,
        runtime=runtime,
    )
    spec = prepared.materialized_plan.payload_specs[0]
    tensor = torch.arange(spec.row_count * spec.shape_suffix[0], dtype=torch.float16).reshape(spec.row_count, spec.shape_suffix[0])
    with pytest.raises(HostAPIDriftError, match="formal execution pipeline failed: unresolved_task"):
        adapter.maybe_execute(
            group=None,
            input_tensor=tensor,
            output_split_sizes=context.recv_splits,
            input_split_sizes=context.send_splits,
            original_all_to_all=lambda *args, **kwargs: tensor.clone(),
        )
    assert store.failed == [(("run", 0, "decoder.layers.0"), "unresolved_task")]
    snapshot = sink.snapshot()
    assert snapshot.event_count == 1
    assert snapshot.events[0].event_type == "executor_submit"
    assert snapshot.events[0].details["success"] is False
    assert snapshot.events[0].details["all_work_completed"] is False
    assert snapshot.events[0].details["failure_code"] == "unresolved_task"


def test_instrumentation_modes_do_not_change_published_or_materialized_digests() -> None:
    _, _, prepared_a = _build_prepared_execution()
    _, _, prepared_b = _build_prepared_execution()
    assert prepared_a.published_plan.logical_plan_digest == prepared_b.published_plan.logical_plan_digest
    assert prepared_a.published_plan.published_plan_digest == prepared_b.published_plan.published_plan_digest
    assert prepared_a.materialized_plan.materialized_plan_digest == prepared_b.materialized_plan.materialized_plan_digest
