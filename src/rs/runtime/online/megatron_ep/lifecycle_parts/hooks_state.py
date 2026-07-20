"""Split lifecycle responsibility: LifecycleHookStateMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleHookStateMixin:
    def capture_phase_transport_output(
        self,
        *,
        layer_name: str,
        phase: str,
        result: Any,
        dispatcher: Any,
    ) -> None:
        recorder = self.observation_recorder
        if recorder is None:
            return
        layer_id = parse_layer_id(layer_name)
        if not recorder.should_capture_tensor(layer_id=layer_id, phase=phase):
            return
        tensors: list[tuple[str, torch.Tensor]] = []
        if isinstance(result, torch.Tensor):
            tensors.append(("hidden_states", result))
        elif isinstance(result, (list, tuple)):
            roles = ["hidden_states", "routing_probs"]
            for index, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    role = roles[index] if index < len(roles) else f"output_{index}"
                    tensors.append((role, item))
        input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        for role, tensor in tensors:
            checksum = hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
            row_digest = hashlib.sha256(
                tensor.detach().float().cpu().reshape(tensor.shape[0], -1).numpy().tobytes()
            ).hexdigest() if tensor.ndim >= 1 else checksum
            recorder.record_captured_tensor(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "phase": phase,
                    "rank": self.rank,
                    "tensor_role": role,
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "input_splits": list(input_splits),
                    "output_splits": list(output_splits),
                    "row_order_digest": row_digest,
                    "tensor_checksum": checksum,
                    "tensor": tensor.detach().cpu(),
                }
            )
    def _record_observer(self, **payload: Any) -> None:
        if self.observer is None:
            return
        try:
            self.observer.record(**payload)
        except Exception:
            pass
    def _context(self, layer_name: str) -> PolicyContext:
        layer_id = parse_layer_id(layer_name)
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        return PolicyContext(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            layer_id=layer_id,
            run_id_digest=digest_text(self.run_id),
            step_id_digest=digest_text(self.step_id),
            microbatch_id_digest=digest_text(self.microbatch_id),
            request_table_hash=self.request_table_hash,
            model_revision_hash=self.model_revision_hash,
            expert_placement_hash="unknown",
            ep_group_ranks=self.ep_group_ranks,
            ep_group_size=len(self.ep_group_ranks),
            ep_group_hash=ep_group_hash,
            future_hint_mode=self.config.future_hint_mode,
            control_mode=self.config.control_mode,
        )
    def _plan_key(self, layer_name: str, phase: str) -> dict[str, Any]:
        return {
            "run_id_digest": digest_text(self.run_id),
            "forward_epoch": int(self._forward_epoch),
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "ep_group_hash": compute_ep_group_hash(self.ep_group_ranks),
            "ep_group_epoch": 0,
            "model_revision_hash": self.model_revision_hash,
            "expert_placement_hash": "unknown",
            "request_table_hash": self.request_table_hash,
        }
    def begin_forward(self, *, forward_epoch: int | None = None) -> None:
        previous_epoch = int(self._forward_epoch)
        if forward_epoch is None:
            self._forward_epoch += 1
        else:
            self._forward_epoch = int(forward_epoch)
        self._current_plan_build_keys.clear()
        self._selected_layer_active_ns.clear()
        self._expert_module_active_ns.clear()
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._selected_layer_active_ns.clear()
        self._expert_module_active_ns.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        self._runtime_state.write("forward_start_ns", int(time.monotonic_ns()))
        self._runtime_state.write("forward_end_ns", 0)
        self._target_plan_reconciled_keys.clear()
        self._latest_execution_outcomes.clear()
        self._latest_result_bundle = None
        self.evidence_counters = RuntimeEvidenceCounters()
        self.expected_evidence.reset(
            claim_scope="formal",
            selected_layers=set(self._selected_layer_id_set),
            measurement_required=str(getattr(self, "_instrumentation_mode", "off") or "off") != "off",
            performance_claim_requested=False,
            prediction_claim_requested=False,
            offline_claim_requested=False,
        )
        self._runtime_failure_reason = ""
        self._cleanup_execution_caches_for_generation(before_generation=int(self._forward_epoch))
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
        )
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        if self.target_planner_service is not None:
            self.target_planner_service.cancel_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )
        if self.target_plan_store is not None:
            self.target_plan_store.cleanup_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )
        if self.control_communication_lane is not None and hasattr(self.control_communication_lane, "cancel_before_generation"):
            self.control_communication_lane.cancel_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )
    def end_forward(self) -> dict[str, Any]:
        exact_generation = int(self._forward_epoch)
        active_transport = self._active_transport is not None
        has_active_prediction = bool(self._runtime_state.read("active_next_dispatch_prediction"))
        self._runtime_state.write("forward_end_ns", int(time.monotonic_ns()))
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        if self.target_planner_service is not None:
            self.target_planner_service.cancel_generation(
                run_id=str(self.run_id),
                forward_epoch=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
        if self.target_plan_store is not None:
            self.target_plan_store.cleanup_epoch(
                run_id=str(self.run_id),
                forward_epoch=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=exact_generation,
            microbatch_id=str(self.microbatch_id),
        )
        self._cleanup_execution_caches_for_generation(exact_generation=exact_generation)
        return {
            "forward_epoch": exact_generation,
            "active_transport_cleared": bool(active_transport),
            "stale_prediction_cleared": bool(has_active_prediction),
            "valid": not active_transport,
        }
    def _append_runtime_state_record(self, key: str, record: dict[str, Any]) -> None:
        rows = list(self._runtime_state.read(key, []) or [])
        rows.append(dict(record))
        self._runtime_state.write(key, rows)
    def record_selected_layer_enter(self, *, layer_name: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        self._selected_layer_active_ns[(int(self._forward_epoch), layer_id)] = int(time.perf_counter_ns())
    def record_selected_layer_exit(self, *, layer_name: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        end_ns = int(time.perf_counter_ns())
        layer_id = str(parse_layer_id(layer_name))
        key = (int(self._forward_epoch), layer_id)
        start_ns = self._selected_layer_active_ns.pop(key, 0)
        if start_ns <= 0:
            self._append_runtime_state_record("selected_layer_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "measurement_status": "unavailable", "reason": "missing_selected_layer_enter"})
            return
        self._append_runtime_state_record("selected_layer_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "selected_layer_enter_ns": int(start_ns), "selected_layer_exit_ns": int(end_ns), "selected_layer_total_us": max(0.0, float(end_ns - start_ns) / 1000.0), "measurement_status": "measured"})
    def record_expert_module_enter(self, *, layer_name: str, expert_module_name: str = "") -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        self._expert_module_active_ns[(int(self._forward_epoch), layer_id)] = int(time.perf_counter_ns())
        status = dict(self._runtime_state.read("attribution_boundary_status", {}) or {})
        status[layer_id] = {**dict(status.get(layer_id, {}) or {}), "expert_boundary_status": "hook_registered", "expert_module_name": str(expert_module_name)}
        self._runtime_state.write("attribution_boundary_status", status)
    def record_expert_module_exit(self, *, layer_name: str, expert_module_name: str = "") -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        end_ns = int(time.perf_counter_ns())
        layer_id = str(parse_layer_id(layer_name))
        key = (int(self._forward_epoch), layer_id)
        start_ns = self._expert_module_active_ns.pop(key, 0)
        if start_ns <= 0:
            self._append_runtime_state_record("expert_module_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "expert_module_name": str(expert_module_name), "measurement_status": "unavailable", "reason": "missing_expert_module_enter"})
            return
        self._append_runtime_state_record("expert_module_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "expert_module_name": str(expert_module_name), "expert_module_enter_ns": int(start_ns), "expert_module_exit_ns": int(end_ns), "expert_module_wall_us": max(0.0, float(end_ns - start_ns) / 1000.0), "measurement_status": "measured"})
    def record_expert_boundary_unavailable(self, *, layer_name: str, reason: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        status = dict(self._runtime_state.read("attribution_boundary_status", {}) or {})
        status[layer_id] = {**dict(status.get(layer_id, {}) or {}), "expert_boundary_status": "unavailable", "expert_boundary_reason": str(reason)}
        self._runtime_state.write("attribution_boundary_status", status)
