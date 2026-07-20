"""Lifecycle Prediction stage methods."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecyclePredictionMixin:
    def _online_p2_predictor_name(self) -> str:
        return str(getattr(self.config, "online_p2_predictor", "copy_current_dispatch") or "copy_current_dispatch")

    def _canonical_online_p2_predictor_id(self) -> str:
        return str(resolve_predictor_id(self._online_p2_predictor_name()))

    def _uses_faithful_fate(self) -> bool:
        return self._canonical_online_p2_predictor_id() == "fate_cross_layer_gate"

    def _fate_predictor_for_request(self, request: TargetLayerPlanningRequest):
        context = request.expert_route_context
        if context is None:
            raise RuntimeError("faithful FATE target planning request is missing ExpertRouteContext")
        second_hop_id = str(self._online_p2_predictor_config().get("second_hop_predictor_id", "bridge_copy_current"))
        second_hop_config = self._online_p2_predictor_config().get("second_hop_predictor_config")
        second_hop = PredictionRegistry.create(second_hop_id, second_hop_config, usage="runtime")
        return FateTwoHorizonRuntimePredictor(
            context_provider=None,
            fixed_context=context,
            second_hop_predictor=second_hop,
        )

    def _planning_horizon(self) -> str:
        value = str(getattr(self.config, "planning_horizon", "p012") or "p012").strip().lower().replace("-", "")
        aliases = {"p012": "p012", "p0123": "p0123"}
        if value not in aliases:
            raise ValueError(f"unsupported planning_horizon {value!r}")
        return aliases[value]

    def _planning_information_mode(self) -> str:
        return "p0_p1_p2_p3" if self._planning_horizon() == "p0123" else "p0_p1_p2"

    def _planning_timing_mode(self) -> str:
        value = str(getattr(self.config, "planning_timing", "legacy_auto") or "legacy_auto").strip().lower().replace("-", "_")
        if value not in {"legacy_auto", "on_demand", "previous_layer"}:
            raise ValueError(f"unsupported planning_timing {value!r}")
        return value

    def _online_p2_predictor_config(self) -> dict[str, Any]:
        return dict(getattr(self.config, "online_p2_predictor_config", {}) or {})

    def _build_online_predictor(self):
        config = self._online_p2_predictor_config()
        if self._canonical_online_p2_predictor_id() == "history":
            config.setdefault("alpha", 0.5)
        return PredictionRegistry.create(self._online_p2_predictor_name(), config, usage="runtime")

    def _predict_dispatch_matrix(
        self,
        *,
        layer_id: str,
        next_layer_id: str,
        current_dispatch_matrix: tuple[tuple[int, ...], ...],
        previous_dispatch_matrix: tuple[tuple[int, ...], ...] | None,
        fallback: bool = False,
    ):
        predictor = self._build_online_predictor()
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:{next_layer_id}",
                run_id=self.run_id,
                forward_id=str(self._forward_epoch),
                source_layer_id=str(layer_id),
                target_layer_id=str(next_layer_id),
            ),
            current_dispatch_rows=current_dispatch_matrix,
            current_return_rows=tuple(
                tuple(int(current_dispatch_matrix[col][row]) for col in range(len(current_dispatch_matrix)))
                for row in range(len(current_dispatch_matrix))
            ),
            history_dispatch_rows=(() if previous_dispatch_matrix is None else (previous_dispatch_matrix,)),
            world_size=len(current_dispatch_matrix),
        )
        prediction = predictor.predict(context)
        return RuntimePredictionCompatResult(
            predictor_id=str(prediction.hint.predictor_id),
            matrix=prediction.hint.target_dispatch_rows,
            matrix_digest=stable_hash([list(row) for row in prediction.hint.target_dispatch_rows]),
            predictor_version="v1",
            confidence=float(prediction.hint.confidence or 0.0),
            evaluation_eligible=not bool(prediction.hint.oracle),
            is_oracle=bool(prediction.hint.oracle),
            valid=True,
            error="",
            fallback=bool(fallback),
        )

    def _resolved_online_policy_family(self) -> str:
        if self._effective_planner_spec_cache is not None:
            return str(getattr(self._effective_planner_spec_cache, "planner_family", ""))
        resolved = resolve_online_policy_config(self.config)
        if resolved is None:
            return ""
        return str(getattr(resolved.spec, "family", ""))

    def _resolved_online_policy_capabilities(self):
        return self._resolved_policy_capabilities_cache

    def _policy_supports_runtime_prediction(self) -> bool:
        return bool(self._cross_layer_prediction_enabled_cache)

    def _policy_uses_joint_window_plan(self) -> bool:
        return bool(self._joint_window_enabled_cache)

    def _should_generate_runtime_prediction(self) -> bool:
        return self._policy_supports_runtime_prediction()

    def _policy_supports_target_layer_preplanning(self) -> bool:
        return bool(self._target_preplanning_enabled_cache)

    def _target_plan_key(self, *, layer_name: str) -> TargetPlanKey:
        return TargetPlanKey(
            run_id=str(self.run_id),
            forward_epoch=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
            target_layer_id=str(parse_layer_id(layer_name)),
        )

    def _ensure_target_planner_runtime(self) -> None:
        if not self._policy_supports_target_layer_preplanning():
            return
        if self.target_plan_store is None:
            self.target_plan_store = TargetPlanStore()
        if self.control_communication_lane is None:
            self.control_communication_lane = GlooControlCommunicationLane(
                rank=int(self.rank),
                world_size=int(len(self.ep_group_ranks) or 1),
                root_rank=int(self.ep_group_root_global_rank),
                process_group=self.target_plan_control_group,
                group_ranks=tuple(int(v) for v in self.ep_group_ranks),
            )
        if self.target_planner_service is None:
            request_factory = self._fate_predictor_for_request if self._uses_faithful_fate() else None
            if request_factory is not None and self.expert_route_context_provider is None:
                raise RuntimeError(
                    "faithful FATE requires an ExpertRouteContext provider at runtime attach"
                )
            from rs.runtime.online.megatron_ep.target_planning.p012_planner_factory import (
                make_target_p012_planner_factory,
            )

            planner_config = dict(getattr(self.config, "planner_config", {}) or {})
            planner_factory = make_target_p012_planner_factory(
                config_provider=lambda _planner_id: planner_config,
            )
            self.target_planner_service = TargetLayerPlannerService(
                store=self.target_plan_store,
                planner_factory=planner_factory,
                two_horizon_predictor_request_factory=request_factory,
                two_horizon_predictor_config=self._online_p2_predictor_config(),
            )
            self.target_planner_service.start()

    def _cleanup_target_plan_runtime(self) -> None:
        if self.target_planner_service is not None:
            self.target_planner_service.shutdown()
        if self.target_plan_store is not None:
            self.target_plan_store.shutdown()
        self.control_communication_lane = None
        self.target_plan_control_group = None
        self.target_plan_control_group_handle = None
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        self._execution_plan_cache().clear()
        self._prepared_execution_cache().clear()

    @staticmethod
    def _target_plan_key_from_slot(slot: Any) -> TargetPlanKey:
        return TargetPlanKey(
            run_id=str(slot.run_id),
            forward_epoch=int(slot.forward_generation),
            microbatch_id=str(slot.microbatch_id),
            target_layer_id=str(slot.target_layer_id),
        )

    def _pump_target_planner_publications(self) -> None:
        if self.target_planner_service is None:
            return
        for ready in self.target_planner_service.drain_ready_publications():
            candidate = self.target_planner_service.local_publication_candidate(ready)
            if candidate is None:
                continue
            self._ready_target_plan_candidates[str(candidate.slot.semantic_digest())] = (ready, candidate)
            if len(self.ep_group_ranks) <= 1 or not dist.is_available() or not dist.is_initialized():
                self._poll_target_plan_slot(target_layer_id=str(ready.request.target_layer_id), safe_point="single_rank_autopublish")

    def _poll_target_plan_slot(self, *, target_layer_id: str, safe_point: str | None = None) -> None:
        if self.control_communication_lane is None or self.target_plan_store is None:
            return
        slot_key = (str(self.run_id), int(self._forward_epoch), str(self.microbatch_id), str(target_layer_id))
        slot = self._expected_publication_slots.get(slot_key)
        if slot is None:
            return
        slot_digest = str(slot.semantic_digest())
        if slot_digest in self._terminal_publication_slots or slot_digest in self._published_publication_slots:
            return
        if safe_point is not None and (slot_digest, str(safe_point)) in self._poll_attempts:
            return
        if safe_point is not None:
            self._poll_attempts.add((slot_digest, str(safe_point)))
        ready_pair = self._ready_target_plan_candidates.get(slot_digest)
        local_candidate = None if ready_pair is None else ready_pair[1]
        if local_candidate is None and self.target_planner_service is not None:
            local_candidate = self.target_planner_service.publication_state_for_slot(slot)
        poll_result = self.control_communication_lane.poll(slot, local_candidate)
        if poll_result.status is PublicationPollStatus.NOT_READY:
            return
        if poll_result.status in {PublicationPollStatus.CANCELLED, PublicationPollStatus.EXPIRED, PublicationPollStatus.FAILED, PublicationPollStatus.SLOT_MISMATCH}:
            self._terminal_publication_slots.add(slot_digest)
            target_key = self._target_plan_key_from_slot(slot)
            if self.target_planner_service is not None:
                self.target_planner_service.cancel_slot(
                    slot,
                    final_status=str(poll_result.status.value).upper(),
                )
            self.target_plan_store.clear_expected_publication(target_key)
            self.target_plan_store.close_key_if_unclaimed(
                target_key,
                final_status="FAILED" if poll_result.status in {PublicationPollStatus.FAILED, PublicationPollStatus.SLOT_MISMATCH} else "CANCELLED",
                execution_origin=f"lane:{poll_result.status.value}",
            )
            self._ready_target_plan_candidates.pop(slot_digest, None)
            self._published_publication_slots.discard(slot_digest)
            return
        if local_candidate is None:
            return
        ready = None if ready_pair is None else ready_pair[0]
        publish_token = PreparationToken(
            service_session_id=int(local_candidate.token.service_session_id),
            forward_generation=int(local_candidate.token.forward_generation),
            target_key=self._target_plan_key_from_slot(slot),
            task_version=int(local_candidate.token.task_version),
            publish_sequence=int(dict(local_candidate.metadata).get("publish_sequence", 0)),
        )
        canonical_payload = dict(poll_result.canonical_payload)
        metadata_payload = dict(canonical_payload.get("metadata") or {})
        plan_payload = dict(canonical_payload.get("plan") or metadata_payload.get("plan") or {})
        if not plan_payload:
            self.target_plan_store.close_key_if_unclaimed(
                publish_token.target_key,
                final_status="FAILED",
                execution_origin="lane:missing_plan_payload",
            )
            return
        published = TargetLayerPreparedJointPlan.from_dict(plan_payload)
        published_plan = None
        if getattr(self, "plan_publisher", None) is not None and published.window_plan is not None:
            published_plan = self.plan_publisher.build(
                publication_slot=slot.semantic_payload(),
                window_plan=published.window_plan,
            )
        publish_result = self.target_plan_store.publish_if_current(token=publish_token, plan=published)
        if publish_result.status not in {"PUBLISHED", "ALREADY_PUBLISHED_SAME"}:
            self._timeline(
                "target_plan_publish_rejected",
                target_layer_id=str(publish_token.target_key.target_layer_id),
                status=str(publish_result.status),
                logical_plan_digest=str(published.logical_plan_digest),
            )
            return
        if published_plan is not None:
            self._execution_plan_cache()[self.target_plan_store._key(publish_token.target_key)] = published_plan
        self._ready_target_plan_candidates.pop(slot_digest, None)
        self._published_publication_slots.add(slot_digest)
        self._store_target_planner_predictions_from_canonical(plan=published)
        self._timeline(
            "target_plan_ready",
            layer_name=str(publish_token.target_key.target_layer_id),
            target_layer_id=str(publish_token.target_key.target_layer_id),
            logical_plan_digest=str(published.logical_plan_digest),
            h1_digest=str(published.h1_prediction_digest),
            h2_digest=str(published.h2_prediction_digest),
            planner_wall_us=float(dict(local_candidate.metadata).get("planner_wall_us", 0.0) or 0.0),
            publish_status=str(publish_result.status),
        )

    def _store_target_planner_predictions_from_canonical(self, *, plan: TargetLayerPreparedJointPlan) -> None:
        from rs.runtime.online.megatron_ep.prediction.contracts import ActiveNextDispatchPrediction, PredictedTrafficMatrix

        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})

        def _to_predicted(
            *,
            predictor_name: str,
            source_layer_id: str,
            target_layer_id: str,
            matrix_rows: tuple[tuple[int, ...], ...],
            matrix_digest: str,
            confidence: float,
        ) -> PredictedTrafficMatrix:
            matrix = tuple(tuple(int(value) for value in row) for row in matrix_rows)
            return PredictedTrafficMatrix(
                predictor_name=str(predictor_name),
                predictor_version="v1",
                source_layer_id=str(source_layer_id),
                predicted_layer_id=str(target_layer_id),
                matrix=matrix,
                matrix_digest=str(matrix_digest),
                total_bytes=int(matrix_remote_bytes(matrix)),
                nonzero_edge_count=int(matrix_nonzero_remote_edge_count(matrix)),
                confidence=float(confidence),
                is_oracle=False,
                evaluation_eligible=True,
                created_at_phase="P0",
                valid=True,
                error="",
            )

        h1_prediction = _to_predicted(
            predictor_name=str(self._online_p2_predictor_name()),
            source_layer_id=str(plan.source_layer_id),
            target_layer_id=str(plan.target_layer_id),
            matrix_rows=plan.h1_rows,
            matrix_digest=str(plan.h1_prediction_digest),
            confidence=1.0,
        )
        predicted_dispatch_by_layer[str(plan.target_layer_id)] = h1_prediction.to_dict()
        next_target_layer_id = (
            str(int(plan.target_layer_id) + 1)
            if str(plan.target_layer_id).isdigit()
            else str(plan.target_layer_id)
        )
        h2_prediction = _to_predicted(
            predictor_name=str(self._online_p2_predictor_name()),
            source_layer_id=str(plan.target_layer_id),
            target_layer_id=next_target_layer_id,
            matrix_rows=plan.h2_rows,
            matrix_digest=str(plan.h2_prediction_digest),
            confidence=1.0,
        )
        predicted_dispatch_by_layer[str(next_target_layer_id)] = h2_prediction.to_dict()
        self._runtime_state.write("predicted_dispatch_by_layer", predicted_dispatch_by_layer)
        self._increment_state_counter_map("predict_count_by_layer", str(plan.source_layer_id))

        active_prediction = ActiveNextDispatchPrediction(
            source_layer_id=str(plan.source_layer_id),
            target_layer_id=str(plan.target_layer_id),
            forecast_matrix=h1_prediction.matrix,
            matrix_digest=str(h1_prediction.matrix_digest),
            predictor_name=str(h1_prediction.predictor_name),
            predictor_version=str(h1_prediction.predictor_version),
            confidence=float(h1_prediction.confidence),
            evaluation_eligible=bool(h1_prediction.evaluation_eligible),
            is_oracle=bool(h1_prediction.is_oracle),
            created_at_phase="P0",
            created_at_stage="target_planner_worker",
            prediction_time_us=0.0,
            valid=True,
            error="",
        )
        self._runtime_state.write("active_next_dispatch_prediction", active_prediction.to_dict())
        self._runtime_state.write("latest_predictor_name", str(h1_prediction.predictor_name))
        self._runtime_state.write("latest_prediction_digest", str(h1_prediction.matrix_digest))
        self._runtime_state.write("latest_prediction_target_layer_id", str(plan.target_layer_id))
        self._runtime_state.write("latest_prediction_matrix_source", "target_planner_worker_h1")
        self._runtime_state.write("latest_prediction_row_sums", [int(sum(row)) for row in h1_prediction.matrix])
        self._runtime_state.write(
            "latest_prediction_col_sums",
            [
                int(sum(h1_prediction.matrix[row_idx][col_idx] for row_idx in range(len(h1_prediction.matrix))))
                for col_idx in range(len(h1_prediction.matrix[0]) if h1_prediction.matrix else 0)
            ],
        )

    def _agree_target_plan_payload(self, payload: dict[str, Any]) -> str:
        digest = str(payload.get("logical_plan_digest", ""))
        if not dist.is_available() or not dist.is_initialized() or len(self.ep_group_ranks) <= 1:
            return digest
        group = self.target_plan_control_group if self.target_plan_control_group is not None else self.ep_process_group
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        device = torch.device("cpu")
        local_len = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
        world_size = int(len(self.ep_group_ranks) or dist.get_world_size(group=group))
        gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
        dist.all_gather(gathered_lens, local_len, group=group)
        max_len = max(int(item.item()) for item in gathered_lens)
        padded = torch.zeros(max_len, dtype=torch.uint8, device=device)
        if encoded:
            padded[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        gathered_payloads = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered_payloads, padded, group=group)
        decoded = []
        for length_tensor, bytes_tensor in zip(gathered_lens, gathered_payloads, strict=True):
            length = int(length_tensor.item())
            decoded.append(bytes(bytes_tensor[:length].tolist()).decode("utf-8"))
        if len(set(decoded)) != 1:
            raise RuntimeError(
                f"target plan agreement mismatch rank={self.rank} payloads={decoded}"
            )
        return digest

    def _build_release_batch_tasks_from_plan(
        self,
        *,
        plan: PhaseExecutionPlan,
        tensor_role: str,
    ) -> list[ReleaseBatchTask]:
        tasks: list[ReleaseBatchTask] = []
        previous_task_id = ""
        peer_sequence = 0
        for wave in plan.waves:
            for task in wave.bucket_tasks:
                payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
                if payload is None or int(payload.row_count) <= 0 or int(task.src_rank) == int(task.dst_rank):
                    continue
                deps = (previous_task_id,) if previous_task_id else ()
                tasks.append(
                    ReleaseBatchTask(
                        task_id=str(task.task_id),
                        phase=str(plan.phase),
                        src_rank=int(task.src_rank),
                        dst_rank=int(task.dst_rank),
                        row_count=int(payload.row_count),
                        sender_offset=int(payload.sender_offset_rows),
                        receiver_offset=int(payload.receiver_offset_rows),
                        tensor_role=str(tensor_role),
                        peer_sequence=int(peer_sequence),
                        dependency_ids=deps,
                        plan_digest=str(plan.plan_hash),
                        plan_version=int((plan.metrics or {}).get("plan_version", 1) or 1),
                    )
                )
                previous_task_id = str(task.task_id)
                peer_sequence += 1
        return tasks

    @staticmethod
    def _residualize_suffix_tasks(
        *,
        candidate_tasks: list[ReleaseBatchTask],
        frozen_tasks: tuple[ReleaseBatchTask, ...],
    ) -> list[ReleaseBatchTask]:
        frozen_ends: dict[tuple[int, int], int] = {}
        for task in frozen_tasks:
            edge = (int(task.src_rank), int(task.dst_rank))
            frozen_ends[edge] = max(
                int(frozen_ends.get(edge, 0)),
                int(task.sender_offset) + int(task.row_count),
            )
        residual: list[ReleaseBatchTask] = []
        for task in candidate_tasks:
            edge = (int(task.src_rank), int(task.dst_rank))
            frozen_end = int(frozen_ends.get(edge, 0))
            start = int(task.sender_offset)
            end = int(task.sender_offset) + int(task.row_count)
            if end <= frozen_end:
                continue
            if start < frozen_end:
                shrink = int(frozen_end - start)
                task = replace(
                    task,
                    sender_offset=int(task.sender_offset) + shrink,
                    receiver_offset=int(task.receiver_offset) + shrink,
                    row_count=int(task.row_count) - shrink,
                )
            if int(task.row_count) > 0:
                residual.append(task)
        return residual

    def handle(self, event: RuntimeEvent) -> RuntimeDecision:
        if isinstance(event, ForwardBeginEvent):
            self.begin_forward(forward_epoch=event.forward_epoch)
            return RuntimeDecision(action="forward_begin")
        if isinstance(event, DispatchReadyEvent):
            if event.layer_role == "prediction_source":
                self.before_prediction_source_dispatch(
                    layer_name=event.layer_name,
                    dispatcher=event.dispatcher,
                    packed_hidden_states=event.packed_hidden_states,
                    packed_probs=event.packed_probs,
                )
            else:
                self.before_token_dispatch(
                    layer_name=event.layer_name,
                    dispatcher=event.dispatcher,
                    packed_hidden_states=event.packed_hidden_states,
                    packed_probs=event.packed_probs,
                )
                self.mark_token_dispatch_committed(layer_name=event.layer_name)
            return RuntimeDecision(action="dispatch_ready", details={"layer_role": event.layer_role})
        if isinstance(event, DispatchCompleteEvent):
            if event.layer_role != "prediction_source":
                self.capture_phase_transport_output(
                    layer_name=event.layer_name,
                    phase="P0",
                    result=event.result,
                    dispatcher=event.dispatcher,
                )
                self.after_token_dispatch(layer_name=event.layer_name)
            return RuntimeDecision(action="dispatch_complete", details={"layer_role": event.layer_role})
        if isinstance(event, DispatchFailedEvent):
            adapter = getattr(self, "transport_adapter", None)
            if adapter is not None and hasattr(adapter, "abort"):
                adapter.abort(
                    layer_name=str(event.layer_name),
                    phase="P0",
                    reason=f"dispatch_failed:{type(event.error).__name__}",
                )
            self._active_transport = None
            self.evidence_counters.execution_failure_count += 1
            self._runtime_failure_reason = f"dispatch_failed:{type(event.error).__name__}"
            self.release_state_ledger.reset(
                run_id=str(self.run_id),
                forward_generation=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
            return RuntimeDecision(action="dispatch_failed", details={"layer_role": event.layer_role, "error": type(event.error).__name__})
        if isinstance(event, CombineReadyEvent):
            self.before_token_combine(
                layer_name=event.layer_name,
                dispatcher=event.dispatcher,
                packed_hidden_states=event.packed_hidden_states,
            )
            return RuntimeDecision(action="combine_ready")
        if isinstance(event, CombineCompleteEvent):
            self.capture_phase_transport_output(
                layer_name=event.layer_name,
                phase="P1",
                result=event.result,
                dispatcher=event.dispatcher,
            )
            self.after_token_combine(layer_name=event.layer_name)
            return RuntimeDecision(action="combine_complete")
        if isinstance(event, CombineFailedEvent):
            adapter = getattr(self, "transport_adapter", None)
            if adapter is not None and hasattr(adapter, "abort"):
                adapter.abort(
                    layer_name=str(event.layer_name),
                    phase="P1",
                    reason=f"combine_failed:{type(event.error).__name__}",
                )
            self._active_transport = None
            self.evidence_counters.execution_failure_count += 1
            self._runtime_failure_reason = f"combine_failed:{type(event.error).__name__}"
            self.release_state_ledger.reset(
                run_id=str(self.run_id),
                forward_generation=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
            return RuntimeDecision(action="combine_failed", details={"error": type(event.error).__name__})
        if isinstance(event, ForwardEndEvent):
            self._finalize_result_bundle()
            self.end_forward()
            return RuntimeDecision(action="forward_end")
        if isinstance(event, ForwardFailedEvent):
            adapter = getattr(self, "transport_adapter", None)
            if adapter is not None and hasattr(adapter, "abort"):
                adapter.abort(reason=f"forward_failed:{type(event.error).__name__}")
            self.evidence_counters.execution_failure_count += 1
            self._runtime_failure_reason = f"forward_failed:{type(event.error).__name__}"
            self._finalize_result_bundle()
            self.end_forward()
            return RuntimeDecision(action="forward_failed", details={"error": type(event.error).__name__})
        raise TypeError(f"unsupported runtime event: {type(event).__name__}")

    def _agree_late_suffix(
        self,
        *,
        key: TargetPlanKey,
        frontier: Any,
        residual_digest: str,
        replacement_tasks: list[ReleaseBatchTask],
        new_plan_digest: str,
        release_epoch: int,
    ) -> dict[str, Any]:
        payload = {
            "key": key.to_dict(),
            "release_epoch": int(release_epoch),
            "frontier_digest": str(frontier.frontier_digest()),
            "residual_digest": str(residual_digest),
            "replacement_suffix_digest": stable_hash(
                [
                    (
                        str(task.task_id),
                        int(task.src_rank),
                        int(task.dst_rank),
                        int(task.row_count),
                        int(task.sender_offset),
                        int(task.receiver_offset),
                        int(task.peer_sequence),
                    )
                    for task in replacement_tasks
                ]
            ),
            "new_plan_digest": str(new_plan_digest),
        }
        self._agree_target_plan_payload(payload)
        return {"agreed": True, "payload": payload}

    def _compile_async_phase_from_logical_plan(
        self,
        *,
        logical_plan: Any,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
        matrix: tuple[tuple[int, ...], ...],
        plan_origin: str,
        plan_version: int,
    ) -> PhaseExecutionPlan:
        global_contexts = reconstruct_global_phase_contexts_from_byte_matrix(
            local_context=local_context,
            matrix=matrix,
            matrix_unit="rows",
        )
        compiled_local_context = next(
            (context for context in global_contexts if int(context.global_rank) == int(local_context.global_rank)),
            local_context,
        )
        canonical_tasks = build_phase_canonical_tasks(
            phase=str(phase),
            matrix_rows=matrix,
            bucket_rows=int(self.config.bucket_rows),
        )
        prepared_wrapper = PreparedWindowPlan(
            window_key=stable_hash({"layer_name": str(layer_name), "phase": str(phase), "plan_origin": str(plan_origin)})[:16],
            forecast_digest="",
            logical_plan=logical_plan,
            created_at_layer_id=str(parse_layer_id(layer_name)),
            applies_from_layer_id=str(parse_layer_id(layer_name)),
            execution_capability_required="joint_window_async_p2p",
            forecast_matrix=(),
        )
        compilation = compile_schedule(
            PlanCompilationRequest(
                logical_plan=logical_plan,
                local_context=compiled_local_context,
                global_contexts=global_contexts,
                canonical_tasks=canonical_tasks,
                phase=str(phase),
                tensor_role="hidden_states" if str(phase) == "P1" else "dispatch_bundle",
                rank_context={
                    "global_rank": int(compiled_local_context.global_rank),
                    "local_rank": int(compiled_local_context.local_rank),
                },
                compilation_options=CompilationOptions(
                    bucket_rows=int(self.config.bucket_rows),
                    p0_weight=float(self.config.p0_weight),
                    p1_reservation_weight=float(self.config.p1_reservation_weight),
                    p2_hint_weight=float(self.config.p2_hint_weight),
                    debug_trace=not self._is_perf_profile(),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    diagnostic_compiler_fallback=bool(getattr(self.config, "diagnostic_compiler_fallback", False)),
                ),
                prepared_plan=prepared_wrapper,
            )
        )
        compiled = compilation.execution_plan
        self._runtime_state.write("compiler_id", str(compilation.audit.compiler_id))
        self._runtime_state.write("logical_plan_digest", str(compilation.audit.logical_plan_digest))
        self._runtime_state.write("compiled_plan_digest", str(compilation.audit.compiled_plan_digest))
        self._runtime_state.write("canonical_task_digest", str(compilation.audit.task_digest))
        self._runtime_state.write("canonical_task_count", int(compilation.audit.task_count))
        self._runtime_state.write("canonical_task_total_rows", int(compilation.audit.total_rows))
        return replace(
            compiled,
            execution_mode="joint_window_async_p2p",
            metrics={
                **compiled.metrics,
                "plan_origin": str(plan_origin),
                "plan_version": int(plan_version),
                "requested_bucket_mode": str(self._requested_bucket_mode()),
                "effective_bucket_mode": str(self._effective_bucket_mode()),
                "requested_bucket_rows": int(self.config.bucket_rows),
                "effective_bucket_rows": int(self.config.bucket_rows),
                "max_inflight_release_batches": int(getattr(self.config, "max_inflight_release_batches", 1) or 1),
            },
        )

    def _record_prediction_for_dispatch(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
        device: torch.device,
    ) -> None:
        stage_start_ns = time.monotonic_ns()
        layer_id = parse_layer_id(layer_name)
        next_layer_id = self._next_layer_id(layer_name)
        world_size = int(len(self.ep_group_ranks) or len(observation.local_p0_row) or 1)
        full_matrix = tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
        remote_matrix = canonicalize_remote_matrix(full_matrix)
        actual_dispatch_by_layer = dict(self._runtime_state.read("actual_dispatch_by_layer", {}) or {})
        actual_dispatch_by_layer[str(layer_id)] = {
            "matrix": [list(row) for row in remote_matrix],
            "full_matrix": [list(row) for row in full_matrix],
            "matrix_digest": matrix_digest_remote(remote_matrix),
            "matrix_source": "pre_transport_phase_ready_context",
            "row_sums": list(matrix_row_sums_remote(remote_matrix)),
            "col_sums": list(matrix_col_sums_remote(remote_matrix)),
            "total_bytes": int(matrix_remote_bytes(remote_matrix)),
            "nonzero_edge_count": int(matrix_nonzero_remote_edge_count(remote_matrix)),
        }
        self._runtime_state.write("actual_dispatch_by_layer", actual_dispatch_by_layer)

        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
        existing_prediction = predicted_dispatch_by_layer.get(str(layer_id))
        audit_start_ns = time.monotonic_ns()
        if isinstance(existing_prediction, dict) and existing_prediction:
            from rs.runtime.online.megatron_ep.prediction.contracts import PredictedTrafficMatrix

            predicted = PredictedTrafficMatrix(
                predictor_name=str(existing_prediction.get("predictor_name", "")),
                predictor_version=str(existing_prediction.get("predictor_version", "")),
                source_layer_id=str(existing_prediction.get("source_layer_id", "")),
                predicted_layer_id=str(existing_prediction.get("predicted_layer_id", "")),
                matrix=tuple(tuple(int(value) for value in row) for row in existing_prediction.get("matrix", [])),
                matrix_digest=str(existing_prediction.get("matrix_digest", "")),
                total_bytes=int(existing_prediction.get("total_bytes", 0) or 0),
                nonzero_edge_count=int(existing_prediction.get("nonzero_edge_count", 0) or 0),
                confidence=float(existing_prediction.get("confidence", 0.0) or 0.0),
                is_oracle=bool(existing_prediction.get("is_oracle", False)),
                evaluation_eligible=bool(existing_prediction.get("evaluation_eligible", False)),
                created_at_phase=str(existing_prediction.get("created_at_phase", "")),
            )
            audit = compare_predicted_to_actual(predicted, remote_matrix)
            audit_row = audit.to_dict()
            if str(audit_row.get("predictor_name", "")) == "copy_current" and self._online_p2_predictor_name() == "copy_current_dispatch":
                audit_row["predictor_name"] = "copy_current_dispatch"
            self.prediction_audits.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "actual_matrix_source": "pre_transport_phase_ready_context",
                    **audit_row,
                }
            )
            predicted_dispatch_by_layer.pop(str(layer_id), None)
        audit_end_ns = time.monotonic_ns()

        stage_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="predict_next_dispatch",
            start_ns=stage_start_ns,
            end_ns=stage_end_ns,
            matrix_source="pre_transport_phase_ready_context",
            matrix_total_bytes=int(matrix_remote_bytes(remote_matrix)),
            matrix_nonzero_edge_count=int(matrix_nonzero_remote_edge_count(remote_matrix)),
            p2_matrix_gather_time_us=0.0,
            p2_matrix_gather_call_count=0,
            predictor_name="target_planner_worker",
            predicted_layer_id=str(next_layer_id),
            prediction_confidence=0.0,
            prediction_valid=True,
            prediction_error="",
            prediction_time_us=0.0,
            audit_time_us=max(0.0, float(audit_end_ns - audit_start_ns) / 1000.0),
            prediction_audit_emitted=bool(existing_prediction is not None),
        )
        if self._policy_supports_target_layer_preplanning() and self._layer_id_selected(str(next_layer_id)):
            self._ensure_target_planner_runtime()
            previous_matrix = None
            previous_record = actual_dispatch_by_layer.get(str(int(layer_id) - 1)) if str(layer_id).isdigit() else None
            if isinstance(previous_record, dict):
                previous_matrix = tuple(tuple(int(value) for value in row) for row in previous_record.get("matrix", []))
            if self.target_planner_service is not None:
                target_planner_id = str(self._target_window_planner_id() or "")
                joint_planner_name = target_planner_id
                local_planner_name = ""
                if str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select") == "host_select":
                    joint_planner_name, local_planner_name = self._runtime_safe_scope_pair(target_planner_id)
                expert_route_context = None
                if self._uses_faithful_fate():
                    if self.expert_route_context_provider is None:
                        raise RuntimeError("faithful FATE context provider disappeared after attach")
                    expert_route_context = self.expert_route_context_provider(
                        source_layer_id=str(layer_id),
                        target_layer_id=str(next_layer_id),
                    )
                result = self.target_planner_service.submit(
                    TargetLayerPlanningRequest(
                        run_id=str(self.run_id),
                        forward_epoch=int(self._forward_epoch),
                        microbatch_id=str(self.microbatch_id),
                        source_layer_id=str(layer_id),
                        target_layer_id=str(next_layer_id),
                        current_p0_rows=remote_matrix,
                        previous_p0_rows=previous_matrix,
                        predictor_name=str(self._online_p2_predictor_name()),
                        policy_id=target_planner_id,
                        group_size=int(world_size),
                        bucket_rows=int(self.config.bucket_rows),
                        policy_options=PlannerPolicyConfig(
                            p0_weight=float(getattr(self.config, "p0_weight", 1.0)),
                            p1_weight=float(getattr(self.config, "p1_reservation_weight", 1.0)),
                            p2_hint_weight=float(getattr(self.config, "p2_hint_weight", 1.0)),
                            p3_return_weight=0.0,
                            residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
                            barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
                            age_weight=float(getattr(self.config, "age_weight", 0.15)),
                            prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
                        ),
                        topology_digest=digest_text(stable_hash({"ep_group_ranks": list(int(v) for v in self.ep_group_ranks)})),
                        bucket_contract_digest=str(self._effective_bucket_mode()),
                        joint_planner_id=str(joint_planner_name),
                        local_planner_id=str(local_planner_name),
                        safe_projection_mode=str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select"),
                        information_mode="p0_p1_p2",
                        planning_track="runtime_lookahead",
                        planning_timing="previous_layer",
                        expert_route_context=expert_route_context,
                    )
                )
                slot = slot_from_request(
                    run_id=str(self.run_id),
                    forward_generation=int(self._forward_epoch),
                    microbatch_id=str(self.microbatch_id),
                    source_layer_id=str(layer_id),
                    target_layer_id=str(next_layer_id),
                )
                self._expected_publication_slots[
                    (str(self.run_id), int(self._forward_epoch), str(self.microbatch_id), str(next_layer_id))
                ] = slot
                self._runtime_state.write("latest_target_plan_submit_status", str(result.status.value))
                self._runtime_state.write("latest_target_plan_submit_task_key", str(result.task_key))
                self._increment_state_counter_map("target_plan_submit_count_by_source_target", f"{layer_id}->{next_layer_id}")
                if result.status in {PreparationSubmitStatus.ACCEPTED, PreparationSubmitStatus.REPLACED_STALE}:
                    self._increment_state_counter_map(
                        "target_plan_enqueue_count_by_source_target",
                        f"{layer_id}->{next_layer_id}",
                    )
                elif result.status is PreparationSubmitStatus.DROPPED_OVERLOAD:
                    self._runtime_state.write("latest_target_plan_preparation_state", "MISSED_OVERLOAD")
                    self._terminal_publication_slots.add(str(slot.semantic_digest()))
                elif result.status is PreparationSubmitStatus.REJECTED_EXPIRED:
                    self._runtime_state.write("latest_target_plan_preparation_state", "EXPIRED")
                    self._terminal_publication_slots.add(str(slot.semantic_digest()))
                elif result.status is PreparationSubmitStatus.REJECTED_CLOSED:
                    raise RuntimeError(f"target_planner_submit_failed:{result.status.value}:{result.task_key}")


__all__ = ["LifecyclePredictionMixin"]
