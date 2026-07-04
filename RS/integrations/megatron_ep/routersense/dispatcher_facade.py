from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import torch

from integrations.megatron_ep.routersense.contracts import (
    InjectionDecision,
    PlanAgreement,
    PolicyContext,
    RankTopologyRecord,
    RouterSenseInjectionConfig,
    RouterSensePlan,
    RuntimeObservation,
)
from integrations.megatron_ep.routersense.execution.fifo_policy import run_phase_plan_agreement
from integrations.megatron_ep.routersense.observer import RouterSenseObserver
from integrations.megatron_ep.routersense.p2 import P2HintRequest, build_p2_hint_provider
from integrations.megatron_ep.routersense.phase import PhaseExecutionPlan, PhaseReadyContext, build_phase_ready_context
from integrations.megatron_ep.routersense.policy.agreement import compute_ep_group_hash, run_policy_agreement
from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy, supported_phase_policies
from integrations.megatron_ep.routersense.policy.joint_shadow import JointShadowP0P1Policy
from integrations.megatron_ep.routersense.policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from integrations.megatron_ep.routersense.policy.native_order import NativeOrderPolicy
from integrations.megatron_ep.routersense.policy.validation import stable_hash


class UnsupportedSchedulerMode(ValueError):
    pass


class SelectedLayerStop(RuntimeError):
    pass


def _parse_layer_id(layer_name: str) -> str:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return match.group(1)
    return "unknown"


def _extract_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flattened.extend(int(sub_item) for sub_item in item)
            else:
                flattened.append(int(item))
        return tuple(flattened)
    if hasattr(value, "tolist"):
        try:
            return _extract_int_tuple(value.tolist())
        except Exception:
            return ()
    return ()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build_runtime_observation(
    *,
    run_id: str,
    step_id: str,
    microbatch_id: str,
    model_revision_hash: str,
    request_table_hash: str,
    hostname: str,
    layer_name: str,
    rank: int,
    local_rank: int,
    ep_group_ranks: tuple[int, ...],
    ep_group_hash: str,
    dispatcher: Any,
    phase: str,
    hidden_states: Any,
) -> RuntimeObservation:
    peer_count = len(ep_group_ranks)
    rank_index = ep_group_ranks.index(rank) if rank in ep_group_ranks else 0
    split_attr = "input_splits" if phase == "P0" else "output_splits"
    splits = list(_extract_int_tuple(getattr(dispatcher, split_attr, None))[:peer_count])
    splits.extend([0] * max(0, peer_count - len(splits)))
    per_peer_rows = tuple(int(v) for v in splits)
    elem_size = int(hidden_states.element_size()) if isinstance(hidden_states, torch.Tensor) else 0
    hidden_dim = int(hidden_states.shape[-1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim >= 2 else 0
    per_peer_bytes = tuple(int(rows * hidden_dim * elem_size) for rows in per_peer_rows)
    local_rows = int(per_peer_rows[rank_index]) if rank_index < len(per_peer_rows) else 0
    remote_rows = sum(int(value) for idx, value in enumerate(per_peer_rows) if idx != rank_index)
    local_expert_indices = _extract_int_tuple(getattr(dispatcher, "local_expert_indices", None))
    tokens_per_expert = _extract_int_tuple(getattr(dispatcher, "num_global_tokens_per_local_expert", None))
    input_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "input_splits", None))[:peer_count])
    output_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "output_splits", None))[:peer_count])
    run_id_digest = _digest(run_id)
    step_id_digest = _digest(step_id)
    microbatch_id_digest = _digest(microbatch_id)
    dispatcher_hash = _digest(type(dispatcher).__name__)
    hostname_digest = _digest(hostname)
    availability = {
        "step_id": "unknown" if step_id == "unknown" else "available",
        "microbatch_id": "unknown" if microbatch_id == "unknown" else "available",
        "layer_id": "unknown" if _parse_layer_id(layer_name) == "unknown" else "available",
        "tokens_per_expert": "available" if tokens_per_expert else "unknown",
    }
    expert_placement_hash = _digest(
        stable_hash(
            {
                "placement_mode": "megatron_native_ep",
                "ep_group_ranks": list(ep_group_ranks),
                "ep_group_size": len(ep_group_ranks),
                "dispatcher_class": type(dispatcher).__name__,
            }
        )
    )
    digest_payload = {
        "run_id_digest": run_id_digest,
        "step_id_digest": step_id_digest,
        "microbatch_id_digest": microbatch_id_digest,
        "layer_id": _parse_layer_id(layer_name),
        "rank": rank,
        "phase": phase,
        "per_peer_rows": list(per_peer_rows),
        "per_peer_bytes": list(per_peer_bytes),
        "local_rows": local_rows,
        "remote_rows": remote_rows,
        "expert_placement_hash": expert_placement_hash,
        "model_revision_hash": model_revision_hash,
        "request_table_hash": request_table_hash,
        "hostname_digest": hostname_digest,
    }
    topology = RankTopologyRecord(
        global_rank=rank,
        local_rank=local_rank,
        node_index=-1,
        hostname_digest=hostname_digest,
        device_index=local_rank,
        ep_group_rank=rank_index,
    )
    return RuntimeObservation(
        run_id=run_id,
        step_id=step_id,
        microbatch_id=microbatch_id,
        layer_id=_parse_layer_id(layer_name),
        layer_name=layer_name,
        global_rank=rank,
        local_rank=local_rank,
        node_id=hostname,
        device=f"cuda:{local_rank}",
        ep_group_ranks=ep_group_ranks,
        ep_group_size=len(ep_group_ranks),
        dispatcher_class=type(dispatcher).__name__,
        expert_placement_hash=expert_placement_hash,
        model_revision_hash=model_revision_hash,
        dispatcher_hash=dispatcher_hash,
        ep_group_hash=ep_group_hash,
        request_table_hash=request_table_hash,
        run_id_digest=run_id_digest,
        step_id_digest=step_id_digest,
        microbatch_id_digest=microbatch_id_digest,
        phase=phase,
        per_peer_rows=per_peer_rows,
        per_peer_bytes=per_peer_bytes,
        local_rows=local_rows,
        remote_rows=remote_rows,
        topology=topology,
        tokens_per_expert=tokens_per_expert,
        input_splits=input_splits,
        output_splits=output_splits,
        observation_digest=stable_hash(digest_payload),
        availability=availability,
    )


@dataclass
class RouterSenseDispatcherFacade:
    native_dispatcher: Callable[..., Any]
    facade_mode: str = "no_op_native_passthrough"
    scheduler_mode: str = "disabled"
    future_hint_mode: str = "none"
    control_mode: str = "default_continue"

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self.native_dispatcher(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        *,
        native_dispatcher: Callable[..., Any],
        config: RouterSenseInjectionConfig,
    ) -> "RouterSenseDispatcherFacade":
        supported_scheduler_modes = {
            "disabled",
            "native_order",
            "joint_shadow_p0p1",
            "native_passthrough_identity",
            *supported_phase_policies(),
        }
        if config.scheduler_mode not in supported_scheduler_modes:
            raise UnsupportedSchedulerMode(
                "Unsupported scheduler_mode="
                f"{config.scheduler_mode!r}; only {sorted(supported_scheduler_modes)!r} are implemented"
            )
        if config.future_hint_mode != "none":
            raise UnsupportedSchedulerMode(
                f"Unsupported future_hint_mode={config.future_hint_mode!r}; only 'none' is implemented"
            )
        if config.control_mode not in {"default_continue", "sync_before_phase"}:
            raise UnsupportedSchedulerMode(
                f"Unsupported control_mode={config.control_mode!r}; only 'default_continue' and 'sync_before_phase' are implemented"
            )
        return cls(
            native_dispatcher=native_dispatcher,
            scheduler_mode=config.scheduler_mode,
            future_hint_mode=config.future_hint_mode,
            control_mode=config.control_mode,
        )


@dataclass
class PolicyRuntimeRecord:
    layer_name: str
    context: PolicyContext
    local_observations: tuple[RuntimeObservation, ...]
    plan: RouterSensePlan
    agreement: PlanAgreement
    decision: InjectionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "context": self.context.to_dict(),
            "local_observations": [item.to_dict() for item in self.local_observations],
            "plan": self.plan.to_dict(),
            "agreement": self.agreement.to_dict(),
            "decision": self.decision.to_dict(),
        }


@dataclass
class RouterSenseInjectionRuntime:
    config: RouterSenseInjectionConfig
    rank: int
    local_rank: int
    run_id: str
    step_id: str
    microbatch_id: str
    model_revision_hash: str
    request_table_hash: str
    hostname: str
    observer: RouterSenseObserver | None = None
    ep_group_ranks: tuple[int, ...] = ()
    ep_group_root_global_rank: int = 0
    ep_process_group: Any | None = None
    completed: list[PolicyRuntimeRecord] = field(default_factory=list)
    _pending_p0: dict[str, RuntimeObservation] = field(default_factory=dict)
    control_timeline: list[dict[str, Any]] = field(default_factory=list)
    control_commands: list[dict[str, Any]] = field(default_factory=list)
    assertion_state: dict[str, Any] = field(default_factory=dict)
    _active_plan_versions: dict[str, int] = field(default_factory=dict)
    _active_plan_hashes: dict[str, str] = field(default_factory=dict)
    phase_contexts: list[dict[str, Any]] = field(default_factory=list)
    transport_bundles: list[dict[str, Any]] = field(default_factory=list)
    scheduled_phase_plans: list[dict[str, Any]] = field(default_factory=list)
    transport_execution_results: list[dict[str, Any]] = field(default_factory=list)
    captured_phase_tensors: list[dict[str, Any]] = field(default_factory=list)
    _active_transport: dict[str, Any] | None = None

    def _effective_phase_policy_name(self) -> str:
        if self.config.policy:
            return str(self.config.policy)
        if self.config.scheduler_mode in set(supported_phase_policies()):
            return str(self.config.scheduler_mode)
        return ""

    def _policy(self):
        phase_policy_name = self._effective_phase_policy_name()
        if phase_policy_name:
            return resolve_phase_policy(
                policy_name=phase_policy_name,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                p2_hint_artifact=self.config.p2_hint_artifact,
            )
        if self.config.scheduler_mode == "native_passthrough_identity":
            return NativePassthroughIdentityPolicy()
        if self.config.scheduler_mode == "native_order":
            return NativeOrderPolicy()
        if self.config.scheduler_mode == "joint_shadow_p0p1":
            return JointShadowP0P1Policy()
        raise UnsupportedSchedulerMode(f"Unsupported scheduler_mode={self.config.scheduler_mode!r}")

    def _layer_selected(self, layer_name: str) -> bool:
        selector = str(self.config.schedule_layer_selector)
        if selector in {"", "all"}:
            return True
        selected = {item.strip() for item in selector.split(",") if item.strip()}
        return _parse_layer_id(layer_name) in selected

    def _phase_selected(self, phase: str) -> bool:
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return True
        return selector == str(phase).lower()

    def _should_schedule_phase(self, *, layer_name: str, phase: str) -> bool:
        return (
            bool(self._effective_phase_policy_name())
            and self.config.execution_mode == "phase_sync_wave"
            and self.config.control_mode == "sync_before_phase"
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
        )

    def _should_stop_after_layer(self, *, layer_name: str, phase: str) -> bool:
        if not (
            self.config.stop_after_selected_layer
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
        ):
            return False
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return str(phase).upper() == "P1"
        return True

    def _activate_transport(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        self._active_transport = {
            "layer_name": layer_name,
            "phase": phase,
            "context": context,
            "plan": plan,
        }
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            adapter.activate(layer_name=layer_name, phase=phase, context=context, plan=plan)

    def current_transport(self) -> dict[str, Any] | None:
        return self._active_transport

    def clear_transport(self, *, layer_name: str, phase: str) -> None:
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            adapter.deactivate(layer_name=layer_name, phase=phase)
        if self._active_transport is None:
            return
        if self._active_transport.get("layer_name") == layer_name and self._active_transport.get("phase") == phase:
            self._active_transport = None

    def record_transport_execution(self, payload: dict[str, Any]) -> None:
        self.transport_execution_results.append(dict(payload))

    def _append_heartbeat(self, payload: dict[str, Any]) -> None:
        if not self.config.executor_heartbeat_path:
            return
        heartbeat_dir = Path(self.config.executor_heartbeat_path)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        target = heartbeat_dir / f"heartbeat-rank{self.rank}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()

    def _timeline(self, event: str, *, layer_name: str, **detail: Any) -> None:
        row = {
            "ts_us": int(time.time() * 1e6),
            "monotonic_ns": time.monotonic_ns(),
            "event_seq": len(self.control_timeline) + 1,
            "event": event,
            "run_id": self.run_id,
            "forward_epoch": 0,
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": _parse_layer_id(layer_name),
            "phase": "P0" if ("p0" in event or "dispatch" in event) else "P1" if ("p1" in event or "combine" in event) else "control",
            "layer": layer_name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "control_mode": self.config.control_mode,
            "scheduler_mode": self.config.scheduler_mode,
            **detail,
        }
        self.control_timeline.append(row)
        if event in {
            "before_phase_plan",
            "after_phase_plan",
            "before_wave",
            "after_wave",
            "before_payload_collective",
            "after_payload_collective",
            "after_phase",
            "p0_pre_transport_observation_ready",
            "p1_pre_transport_observation_ready",
            "p0_native_dispatch_committed",
        }:
            self._append_heartbeat(row)

    def _build_p2_hint(self, *, layer_name: str, phase: str):
        provider = build_p2_hint_provider(self.config.p2_hint_mode)
        return provider.build_hint(
            P2HintRequest(
                plan_key=self._plan_key(layer_name, phase),
                layer_id=_parse_layer_id(layer_name),
                phase=phase,
                global_rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
            )
        )

    def capture_phase_transport_output(
        self,
        *,
        layer_name: str,
        phase: str,
        result: Any,
        dispatcher: Any,
    ) -> None:
        if not self.config.capture_phase_tensors:
            return
        if not self._layer_selected(layer_name) or not self._phase_selected(phase):
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
        input_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        output_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        for role, tensor in tensors:
            checksum = hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
            row_digest = hashlib.sha256(
                tensor.detach().float().cpu().reshape(tensor.shape[0], -1).numpy().tobytes()
            ).hexdigest() if tensor.ndim >= 1 else checksum
            self.captured_phase_tensors.append(
                {
                    "layer_name": layer_name,
                    "layer_id": _parse_layer_id(layer_name),
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
        layer_id = _parse_layer_id(layer_name)
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        return PolicyContext(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            layer_id=layer_id,
            run_id_digest=_digest(self.run_id),
            step_id_digest=_digest(self.step_id),
            microbatch_id_digest=_digest(self.microbatch_id),
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
            "run_id_digest": _digest(self.run_id),
            "forward_epoch": 0,
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": _parse_layer_id(layer_name),
            "phase": phase,
            "ep_group_hash": compute_ep_group_hash(self.ep_group_ranks),
            "ep_group_epoch": 0,
            "model_revision_hash": self.model_revision_hash,
            "expert_placement_hash": "unknown",
            "request_table_hash": self.request_table_hash,
        }

    def before_token_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        observation = _build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=ep_group_hash,
            dispatcher=dispatcher,
            phase="P0",
            hidden_states=packed_hidden_states,
        )
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P0")
        self._pending_p0[layer_name] = observation
        phase_ctx = build_phase_ready_context(
            plan_key=self._plan_key(layer_name, "P0"),
            phase="P0",
            control_mode=self.config.control_mode,
            forward_epoch=0,
            layer_id=_parse_layer_id(layer_name),
            layer_name=layer_name,
            global_rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_root_rank=self.ep_group_root_global_rank,
            topology=observation.topology.to_dict(),
            dispatcher_class=type(dispatcher).__name__,
            dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
            expert_placement_hash=observation.expert_placement_hash,
            input_splits=observation.input_splits,
            output_splits=observation.output_splits,
            packed_tensors=tuple(
                tensor
                for tensor in (packed_hidden_states, packed_probs)
                if isinstance(tensor, torch.Tensor)
            ),
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
            p2_hint=p2_hint,
        )
        self.phase_contexts.append(phase_ctx.to_dict())
        self.transport_bundles.extend(bundle.to_dict() for bundle in phase_ctx.transport_bundles)
        pre_input_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        pre_output_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        hidden_ptr = int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1
        probs_ptr = int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1
        self._timeline(
            "p0_pre_transport_observation_ready",
            layer_name=layer_name,
            input_splits=list(pre_input_splits),
            output_splits=list(pre_output_splits),
            hidden_shape=list(packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
            probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P0")
        if self._should_schedule_phase(layer_name=layer_name, phase="P0"):
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=self._policy(), group=self.ep_process_group)
            self.scheduled_phase_plans.append(plan.to_dict())
            self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P0",
                plan_hash=plan.plan_hash,
                wave_count=len(plan.waves),
                bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                execution_mode=plan.execution_mode,
            )
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P0", plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            return
        context = replace(self._context(layer_name), expert_placement_hash=observation.expert_placement_hash)
        local_observations = (observation,)
        plan, agreement = run_policy_agreement(
            local_observations=local_observations,
            context=context,
            policy=self._policy(),
            device=torch.device(f"cuda:{self.local_rank}"),
            group=self.ep_process_group,
        )
        post_input_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        post_output_splits = tuple(int(v) for v in _extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        self.assertion_state["native_splits_unchanged"] = pre_input_splits == post_input_splits and pre_output_splits == post_output_splits
        self.assertion_state["native_buffers_unchanged"] = (
            hidden_ptr == (int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1)
            and probs_ptr == (int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1)
        )
        current_version = self._active_plan_versions.get(layer_name, 0)
        self._active_plan_versions[layer_name] = current_version
        self._active_plan_hashes[layer_name] = plan.plan_hash
        decision = InjectionDecision(
            accepted=True,
            fallback="native",
            plan_hash=plan.plan_hash,
            reason="identity_pre_transport_passthrough",
            policy_name=plan.policy_name,
            control_mode=self.config.control_mode,
        )
        self.completed.append(
            PolicyRuntimeRecord(
                layer_name=layer_name,
                context=context,
                local_observations=local_observations,
                plan=plan,
                agreement=agreement,
                decision=decision,
            )
        )
        self._timeline(
            "root_plan_broadcast_received",
            layer_name=layer_name,
            root_wire_hash=agreement.root_wire_hash,
        )
        self._timeline(
            "root_plan_decoded",
            layer_name=layer_name,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "plan_agreement_verified",
            layer_name=layer_name,
            agreement_status=agreement.agreement_status,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "identity_plan_agreed",
            layer_name=layer_name,
            root_wire_hash=agreement.root_wire_hash,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
            agreement_status=agreement.agreement_status,
            version=current_version,
        )
        self._record_observer(
            phase="policy_plan",
            layer=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            policy_name=plan.policy_name,
            scheduler_mode=self.config.scheduler_mode,
            control_mode=self.config.control_mode,
            plan_hash=plan.plan_hash,
            execution_mode=plan.execution_mode,
            wave_count=len(plan.waves),
            agreement=agreement.to_dict(),
            decision=decision.to_dict(),
        )
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "before_commit":
            self._active_plan_versions[layer_name] = current_version + 1
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current_version,
                    "new_version": current_version + 1,
                    "status": "applied",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_replaced_active",
                layer_name=layer_name,
                old_version=current_version,
                new_version=current_version + 1,
                transport_mutation=False,
            )

    def mark_token_dispatch_committed(self, *, layer_name: str) -> None:
        if self.config.scheduler_mode != "native_passthrough_identity" and not bool(self._effective_phase_policy_name()):
            return
        self._timeline(
            "p0_native_dispatch_committed",
            layer_name=layer_name,
            active_version=self._active_plan_versions.get(layer_name, 0),
        )

    def after_token_dispatch(self, *, layer_name: str) -> None:
        if bool(self._effective_phase_policy_name()):
            self.clear_transport(layer_name=layer_name, phase="P0")
            if str(self.config.schedule_phase_selector).lower() == "p0" and self._should_stop_after_layer(layer_name=layer_name, phase="P0"):
                raise SelectedLayerStop(f"Stopped after selected P0 layer {layer_name}")
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            return
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "after_commit":
            current = self._active_plan_versions.get(layer_name, 0)
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current,
                    "new_version": current + 1,
                    "status": "expired_late",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_expired_late",
                layer_name=layer_name,
                old_version=current,
                attempted_version=current + 1,
                transport_mutation=False,
            )

    def after_token_combine(self, *, layer_name: str) -> None:
        if bool(self._effective_phase_policy_name()):
            self.clear_transport(layer_name=layer_name, phase="P1")
            if self._should_stop_after_layer(layer_name=layer_name, phase="P1"):
                raise SelectedLayerStop(f"Stopped after selected P1 layer {layer_name}")
            return
        if self.config.scheduler_mode == "native_passthrough_identity":
            self._timeline("native_p1_observed", layer_name=layer_name)

    def before_token_combine(self, *, layer_name: str, dispatcher: Any, packed_hidden_states: Any) -> None:
        observation = _build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
            dispatcher=dispatcher,
            phase="P1",
            hidden_states=packed_hidden_states,
        )
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P1")
        phase_ctx = build_phase_ready_context(
            plan_key=self._plan_key(layer_name, "P1"),
            phase="P1",
            control_mode=self.config.control_mode,
            forward_epoch=0,
            layer_id=_parse_layer_id(layer_name),
            layer_name=layer_name,
            global_rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_root_rank=self.ep_group_root_global_rank,
            topology=observation.topology.to_dict(),
            dispatcher_class=type(dispatcher).__name__,
            dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
            expert_placement_hash=observation.expert_placement_hash,
            input_splits=observation.input_splits,
            output_splits=observation.output_splits,
            packed_tensors=(packed_hidden_states,) if isinstance(packed_hidden_states, torch.Tensor) else (),
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
            p2_hint=p2_hint,
        )
        self.phase_contexts.append(phase_ctx.to_dict())
        self.transport_bundles.extend(bundle.to_dict() for bundle in phase_ctx.transport_bundles)
        self._timeline(
            "p1_pre_transport_observation_ready",
            layer_name=layer_name,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P1")
        if self._should_schedule_phase(layer_name=layer_name, phase="P1"):
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=self._policy(), group=self.ep_process_group)
            self.scheduled_phase_plans.append(plan.to_dict())
            self._activate_transport(layer_name=layer_name, phase="P1", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P1",
                plan_hash=plan.plan_hash,
                wave_count=len(plan.waves),
                bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                execution_mode=plan.execution_mode,
            )
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P1", plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            return

    def on_dispatch(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        if self.config.scheduler_mode in {"disabled", "native_passthrough_identity"} or bool(self._effective_phase_policy_name()):
            return
        observation = _build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
            dispatcher=dispatcher,
            phase="P0",
            hidden_states=hidden_states,
        )
        self._pending_p0[layer_name] = observation

    def on_combine(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        if self.config.scheduler_mode in {"disabled", "native_passthrough_identity"} or bool(self._effective_phase_policy_name()):
            return
        if layer_name not in self._pending_p0:
            return
        p0_observation = self._pending_p0.pop(layer_name)
        p1_observation = _build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
            dispatcher=dispatcher,
            phase="P1",
            hidden_states=hidden_states,
        )
        context = replace(self._context(layer_name), expert_placement_hash=p0_observation.expert_placement_hash)
        local_observations = (p0_observation, p1_observation)
        policy = self._policy()
        plan, agreement = run_policy_agreement(
            local_observations=local_observations,
            context=context,
            policy=policy,
            device=torch.device(f"cuda:{self.local_rank}"),
            group=self.ep_process_group,
        )
        decision = InjectionDecision(
            accepted=True,
            fallback="native",
            plan_hash=plan.plan_hash,
            reason="native_order_passthrough" if plan.policy_name == "native_order" else "shadow_only_passthrough",
            policy_name=plan.policy_name,
            control_mode=self.config.control_mode,
        )
        self.completed.append(
            PolicyRuntimeRecord(
                layer_name=layer_name,
                context=context,
                local_observations=local_observations,
                plan=plan,
                agreement=agreement,
                decision=decision,
            )
        )
        self._record_observer(
            phase="policy_plan",
            layer=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            policy_name=plan.policy_name,
            scheduler_mode=self.config.scheduler_mode,
            control_mode=self.config.control_mode,
            plan_hash=plan.plan_hash,
            execution_mode=plan.execution_mode,
            wave_count=len(plan.waves),
            ready_wave_count=len(plan.ready_waves),
            blocked_future_wave_count=len(plan.blocked_future_waves),
            agreement=agreement.to_dict(),
            decision=decision.to_dict(),
        )

    def export_records(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.completed]

    def export_control_timeline(self) -> list[dict[str, Any]]:
        return list(self.control_timeline)

    def export_control_commands(self) -> list[dict[str, Any]]:
        return list(self.control_commands)

    def export_assertions(self) -> dict[str, Any]:
        return dict(self.assertion_state)

    def export_phase_contexts(self) -> list[dict[str, Any]]:
        return list(self.phase_contexts)

    def export_transport_bundles(self) -> list[dict[str, Any]]:
        return list(self.transport_bundles)

    def export_scheduled_phase_plans(self) -> list[dict[str, Any]]:
        return list(self.scheduled_phase_plans)

    def export_transport_execution_results(self) -> list[dict[str, Any]]:
        return list(self.transport_execution_results)

    def export_captured_phase_tensors(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.captured_phase_tensors:
            rows.append({key: value for key, value in item.items() if key != "tensor"})
        return rows
