"""Split lifecycle responsibility: LifecycleAsyncPlanningMixin."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403 - internal lifecycle dependency surface


class LifecycleAsyncPlanningMixin:
    def _build_provisional_async_plan(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation_p0: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
    ) -> PhaseExecutionPlan:
        self._store_runtime_joint_plan_from_p0(
            layer_name=layer_name,
            phase_ctx=phase_ctx,
            observation_p0=observation_p0,
            actual_p0_full_row_matrix=actual_p0_full_row_matrix,
            plan_origin="provisional_current_plan",
        )
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            raise RuntimeError(f"missing provisional prepared plan for {layer_name}")
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=getattr(prepared_plan, "logical_plan"),
            layer_name=layer_name,
            phase="P0",
            local_context=phase_ctx,
            matrix=tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix),
            plan_origin="provisional",
            plan_version=0,
        )
        self._runtime_state.write("execution_origin", "provisional_only")
        self.evidence_counters.provisional_execution_count += 1
        self.evidence_counters.fallback_count += 1
        self._runtime_state.write("provisional_plan_digest", str(compiled.plan_hash))
        return compiled
    def _late_suffix_provider(
        self,
        *,
        context: PhaseReadyContext,
        plan: PhaseExecutionPlan,
        tensor_role: str,
        frontier: Any,
        release_epoch: int,
    ) -> dict[str, Any] | None:
        if not self._policy_supports_target_layer_preplanning() or self.target_plan_store is None:
            return None
        if str(tensor_role) != "hidden_states":
            return None
        layer_name = str(context.layer_name)
        key = self._target_plan_key(layer_name=layer_name)
        if self.target_plan_store.peek(key) is None:
            return None
        prepared_plan = self.target_plan_store.claim_for_reconciliation(key)
        self._runtime_state.write("prepared_target_selected_variant", str(getattr(prepared_plan, "selected_variant", "")))
        self._runtime_state.write(
            "prepared_target_safe_projection_mode",
            str(getattr(prepared_plan, "safe_projection_mode", "disabled") or "disabled"),
        )
        if getattr(frontier, "pending_count", lambda: 0)() <= 0:
            self.target_plan_store.expire_key(key, execution_origin="too_late_no_effect")
            return None
        actual_rows = tuple(
            tuple(int(value) for value in row)
            for row in (
                ((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix"))
                or []
            )
        )
        if not actual_rows:
            return None
        outcome = _reconcile_once_compat(
            prepared_plan=prepared_plan,
            actual_p0_rows=canonicalize_remote_matrix(actual_rows),
            frozen_frontier=set(frontier.immutable_prefix_ids()),
        )
        if outcome.status == "rejected" or outcome.logical_plan is None:
            self.target_plan_store.reject(key, execution_origin="late_rejected")
            return None
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=outcome.logical_plan,
            layer_name=layer_name,
            phase=str(context.phase),
            local_context=context,
            matrix=actual_rows,
            plan_origin="late_spliced",
            plan_version=2,
        )
        compiled_tasks = self._build_release_batch_tasks_from_plan(plan=compiled, tensor_role=tensor_role)
        suffix_tasks = self._residualize_suffix_tasks(
            candidate_tasks=compiled_tasks,
            frozen_tasks=tuple(frontier.immutable_prefix()),
        )
        if not suffix_tasks:
            self.target_plan_store.expire_key(key, execution_origin="too_late_no_effect")
            return None
        agreement_token = self._agree_late_suffix(
            key=key,
            frontier=frontier,
            residual_digest=stable_hash(
                [
                    (int(task.src_rank), int(task.dst_rank), int(task.row_count), int(task.sender_offset), int(task.receiver_offset))
                    for task in suffix_tasks
                ]
            ),
            replacement_tasks=suffix_tasks,
            new_plan_digest=str(outcome.logical_plan_digest or compiled.plan_hash),
            release_epoch=int(release_epoch),
        )
        self.target_plan_store.consume_once(key, execution_origin="provisional_then_late_suffix")
        self._runtime_state.write("execution_origin", "provisional_then_late_suffix")
        self._runtime_state.write("suffix_splice_count", 1)
        return {
            "apply_suffix": True,
            "suffix_tasks": suffix_tasks,
            "new_plan_version": 2,
            "parent_plan_version": int((plan.metrics or {}).get("plan_version", 0) or 0),
            "agreement_token": agreement_token,
        }
    def _compile_async_local_phase_plan(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
    ) -> PhaseExecutionPlan:
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            raise RuntimeError(f"missing prepared runtime joint plan for {layer_name} {phase}")
        if str(phase) == "P0":
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
            )
            matrix_unit = "rows"
        else:
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix")) or [])
            )
            matrix_unit = "rows"
            if not matrix:
                matrix = tuple(
                    tuple(int(value) for value in row)
                    for row in (self._runtime_state.read("p1_inferred_from_p0") or [])
                )
        if not matrix:
            raise RuntimeError(f"missing global row matrix for async local materialization {layer_name} {phase}")
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
        bucket_summary = summarize_bucket_tasks(canonical_tasks)
        compilation = compile_schedule(
            PlanCompilationRequest(
                logical_plan=getattr(prepared_plan, "logical_plan"),
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
                prepared_plan=prepared_plan,
                prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                phase_policy_name=str(self._effective_phase_policy_name() or "prepared_priority"),
            )
        )
        compiled = compilation.execution_plan
        self._runtime_state.write("compiler_id", str(compilation.audit.compiler_id))
        self._runtime_state.write("logical_plan_digest", str(compilation.audit.logical_plan_digest))
        self._runtime_state.write("compiled_plan_digest", str(compilation.audit.compiled_plan_digest))
        self._runtime_state.write("canonical_task_digest", str(compilation.audit.task_digest))
        self._runtime_state.write("canonical_task_count", int(compilation.audit.task_count))
        self._runtime_state.write("canonical_task_total_rows", int(compilation.audit.total_rows))
        self._runtime_state.write(
            "secondary_policy_invocation_count",
            int(compilation.audit.metrics.get("secondary_policy_invocation_count", 0) or 0),
        )
        self._runtime_state.write(
            "secondary_policy_call_count",
            int(compilation.audit.metrics.get("secondary_policy_call_count", 0) or 0),
        )
        self._runtime_state.write(
            "direct_compiler_selected_count",
            int(compilation.audit.metrics.get("direct_compiler_selected_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_compare_count",
            int(compilation.audit.metrics.get("compiler_shadow_compare_count", 0) or 0),
        )
        self._runtime_state.write("compiler_shadow_status", str(compilation.audit.metrics.get("shadow_status", "")))
        self._runtime_state.write(
            "compiler_shadow_plan_hash_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_plan_hash_matches_legacy", False)),
        )
        self._runtime_state.write("compiler_shadow_plan_hash", str(compilation.audit.metrics.get("shadow_plan_hash", "")))
        self._runtime_state.write(
            "compiler_shadow_missing_task_count",
            int(compilation.audit.metrics.get("shadow_missing_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_extra_task_count",
            int(compilation.audit.metrics.get("shadow_extra_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_execution_order_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_execution_order_matches_legacy", False)),
        )
        return replace(
            compiled,
            execution_mode="joint_window_async_p2p",
            metrics={
                **compiled.metrics,
                "requested_bucket_mode": str(self._requested_bucket_mode()),
                "effective_bucket_mode": str(self._effective_bucket_mode()),
                "requested_bucket_rows": int(self.config.bucket_rows),
                "effective_bucket_rows": int(self.config.bucket_rows),
                "canonical_bucket_task_summary": bucket_summary,
                "joint_window_async_local_materialization": True,
                "p1_planning_collective_count": 0 if str(phase) == "P1" else int(compiled.metrics.get("p1_planning_collective_count", 0) or 0),
                "prediction_extra_collective_count": 0,
                "preflight_mode": str(getattr(self.config, "preflight_mode", "full")),
                "emit_detailed_task_artifacts": not self._is_perf_profile(),
            },
        )
