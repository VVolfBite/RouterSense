from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    GlobalJointPlanWire,
    agree_global_joint_plan,
    validate_local_schedule_against_global_plan,
    validate_pairwise_send_recv_contracts,
)


def _wire() -> GlobalJointPlanWire:
    return GlobalJointPlanWire(
        window_key="run:epoch1:mb0:layer5",
        policy_name="routersense_joint_zero_hint_async_p2p",
        safe_selected_policy="future:p012:joint:global:rscf",
        prediction_digest="pred0",
        canonical_edge_order=(("P0", 2, 3), ("P1", 3, 2)),
        wave_metadata=((0, (("P0", 2, 3),)), (1, (("P1", 3, 2),))),
        per_peer_sequence_digest="seq0",
    )


def test_global_plan_same_local_length_can_differ() -> None:
    wire = _wire()
    result = agree_global_joint_plan(wire, gathered_wires=(wire, wire))
    assert result["valid"] is True
    rank0_schedule = (
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P1", "src_rank": 3, "dst_rank": 2, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
    )
    rank1_schedule = (
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("routing_probs",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("routing_probs",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P1", "src_rank": 3, "dst_rank": 2, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
    )
    assert validate_local_schedule_against_global_plan(rank0_schedule, global_wire=wire)["valid"] is True
    assert validate_local_schedule_against_global_plan(rank1_schedule, global_wire=wire)["valid"] is True


def test_global_plan_digest_mismatch_fails_before_execution() -> None:
    first = _wire()
    second = GlobalJointPlanWire(
        window_key=first.window_key,
        policy_name=first.policy_name,
        safe_selected_policy=first.safe_selected_policy,
        prediction_digest="pred1",
        canonical_edge_order=first.canonical_edge_order,
        wave_metadata=first.wave_metadata,
        per_peer_sequence_digest=first.per_peer_sequence_digest,
    )
    result = agree_global_joint_plan(first, gathered_wires=(first, second))
    assert result["valid"] is False
    assert "global_plan_digest_mismatch" in result["errors"]


def test_pairwise_send_recv_contracts_validate_matching_roles_and_rows() -> None:
    schedules = (
        (
            {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        ),
        (
            {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        ),
    )
    assert validate_pairwise_send_recv_contracts(schedules)["valid"] is True


def test_unified_compiler_preserves_repeated_edge_logical_wave_chunks() -> None:
    from rs.runtime.online.megatron_ep.compiler_facade import (
        CompilationOptions,
        PlanCompilationRequest,
        build_phase_canonical_tasks,
        compile_schedule,
    )
    from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
    from tests.contract.megatron_ep.helpers import make_contexts_from_matrix

    matrix = ((0, 5), (0, 0))
    contexts = make_contexts_from_matrix(phase="P0", matrix=matrix)
    logical = LogicalSchedulePlan(
        policy_name="split-edge-fixture",
        waves=(
            LogicalWave(
                wave_id=7,
                flows=(
                    FlowDemand(
                        flow_id="chunk-a",
                        phase="p0_dispatch",
                        src_rank=0,
                        dst_rank=1,
                        byte_count=2,
                        release_state="ready",
                        is_executable=True,
                    ),
                ),
                duration=2.0,
            ),
            LogicalWave(
                wave_id=9,
                flows=(
                    FlowDemand(
                        flow_id="chunk-b",
                        phase="p0_dispatch",
                        src_rank=0,
                        dst_rank=1,
                        byte_count=3,
                        release_state="ready",
                        is_executable=True,
                    ),
                ),
                duration=3.0,
            ),
        ),
        diagnostics={},
    )
    result = compile_schedule(
        PlanCompilationRequest(
            logical_plan=logical,
            local_context=contexts[0],
            global_contexts=contexts,
            canonical_tasks=build_phase_canonical_tasks(
                phase="P0",
                matrix_rows=matrix,
                bucket_rows=0,
            ),
            phase="P0",
            tensor_role="dispatch_bundle",
            rank_context={"global_rank": 0, "local_rank": 0},
            compilation_options=CompilationOptions(bucket_rows=0),
            prepared_plan=object(),
        )
    )
    tasks = [task for wave in result.execution_plan.waves for task in wave.bucket_tasks]
    assert [len(wave.bucket_tasks) for wave in result.execution_plan.waves] == [1, 1]
    assert [task.row_count for task in tasks] == [2, 3]
    assert [task.sender_offset_rows for task in tasks] == [0, 2]
    assert [task.receiver_offset_rows for task in tasks] == [0, 2]
    assert result.execution_plan.metrics["direct_flow_chunk_preserved"] is True
    assert result.audit.total_rows == 5


def test_unified_compiler_preserves_rscf_flow_chunks_across_ep_sizes() -> None:
    import random

    from rs.core.contracts import (
        PlanningConstraints,
        PlanningIdentity,
        PlanningRequest,
        PlanningTopology,
        PlanningTraffic,
        PlanningWeights,
        PredictionHint,
    )
    from rs.planning import PlannerRegistry
    from rs.planning.api import to_logical_plan
    from rs.runtime.online.megatron_ep.compiler_facade import (
        CompilationOptions,
        PlanCompilationRequest,
        build_phase_canonical_tasks,
        compile_schedule,
    )
    from tests.contract.megatron_ep.helpers import make_contexts_from_matrix

    rng = random.Random(20260720)
    for world_size in (4, 8):
        for case_id in range(3):
            p0 = tuple(
                tuple(0 if src == dst else rng.randint(0, 9) for dst in range(world_size))
                for src in range(world_size)
            )
            p1 = tuple(
                tuple(int(p0[src][dst]) for src in range(world_size))
                for dst in range(world_size)
            )
            p2 = tuple(
                tuple(0 if src == dst else rng.randint(0, 9) for dst in range(world_size))
                for src in range(world_size)
            )
            request = PlanningRequest(
                identity=PlanningIdentity(
                    request_id=f"compiler-rscf-{world_size}-{case_id}",
                    source_layer_id="0",
                    target_layer_id="1",
                ),
                traffic=PlanningTraffic(p0_dispatch_rows=p0, p1_return_rows=p1),
                prediction_hint=PredictionHint(
                    predictor_id="deterministic_fixture",
                    hint_type="traffic_matrix",
                    target_dispatch_rows=p2,
                    confidence=0.75,
                    source_layer_id="0",
                    target_layer_id="1",
                ),
                topology=PlanningTopology(world_size=world_size),
                constraints=PlanningConstraints(
                    bucket_rows=0,
                    max_waves=4096,
                    expert_compute_delay=0.0,
                    phase_release_model="p1_return",
                ),
                weights=PlanningWeights(),
                information_mode="p0_p1_p2",
            )
            window_plan = PlannerRegistry.create(
                "future:p012:joint:global:rscf", None
            ).plan(request)
            logical = to_logical_plan(window_plan)
            for phase, logical_phase, matrix in (
                ("P0", "p0_dispatch", p0),
                ("P1", "p1_return", p1),
            ):
                contexts = make_contexts_from_matrix(phase=phase, matrix=matrix)
                result = compile_schedule(
                    PlanCompilationRequest(
                        logical_plan=logical,
                        local_context=contexts[0],
                        global_contexts=contexts,
                        canonical_tasks=build_phase_canonical_tasks(
                            phase=phase,
                            matrix_rows=matrix,
                            bucket_rows=0,
                        ),
                        phase=phase,
                        tensor_role="hidden_states" if phase == "P1" else "dispatch_bundle",
                        rank_context={"global_rank": 0, "local_rank": 0},
                        compilation_options=CompilationOptions(bucket_rows=0),
                        prepared_plan=object(),
                    )
                )
                expected = [
                    tuple(
                        (int(flow.src_rank), int(flow.dst_rank), int(flow.byte_count))
                        for flow in wave.flows
                        if str(flow.phase) == logical_phase and int(flow.byte_count) > 0
                    )
                    for wave in logical.waves
                ]
                expected = [wave for wave in expected if wave]
                actual = [
                    tuple(
                        (int(task.src_rank), int(task.dst_rank), int(task.row_count))
                        for task in wave.bucket_tasks
                    )
                    for wave in result.execution_plan.waves
                ]
                assert actual == expected
                consumed: dict[tuple[int, int], int] = {}
                sender_base: dict[tuple[int, int], int] = {}
                receiver_base: dict[tuple[int, int], int] = {}
                for wave in result.execution_plan.waves:
                    for task in wave.bucket_tasks:
                        edge = (int(task.src_rank), int(task.dst_rank))
                        offset = consumed.get(edge, 0)
                        sender_base.setdefault(edge, int(task.sender_offset_rows))
                        receiver_base.setdefault(edge, int(task.receiver_offset_rows))
                        assert int(task.sender_offset_rows) == sender_base[edge] + offset
                        assert int(task.receiver_offset_rows) == receiver_base[edge] + offset
                        consumed[edge] = offset + int(task.row_count)
                expected_rows = {
                    (src, dst): int(matrix[src][dst])
                    for src in range(world_size)
                    for dst in range(world_size)
                    if int(matrix[src][dst]) > 0
                }
                assert consumed == expected_rows
                assert result.execution_plan.metrics["logical_flow_chunk_preserved"] is True
