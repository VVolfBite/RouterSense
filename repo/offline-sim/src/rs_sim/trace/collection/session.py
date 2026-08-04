"""Capture session used by automatic and explicit instrumentation."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .extract import RoutingExtractionError, extract_routing_counts
from .fate_artifacts import canonical_fate_record_digest
from .fate_online import OnlineFateError, SampledGateInput, capture_sampled_gate_input, predict_rank_row
from .writer import RankArtifactWriter


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _env_int(*names: str, default: int = 0) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return int(default)


class _EventInterval:
    def __init__(self, *, kind: str, layer_id: int, request_id: str, decode_step: int, rank: int) -> None:
        self.kind = kind
        self.layer_id = int(layer_id)
        self.request_id = request_id
        self.decode_step = int(decode_step)
        self.rank = int(rank)
        self.start_cpu_ns = time.perf_counter_ns()
        self.end_cpu_ns: int | None = None
        self.start_event = None
        self.end_event = None
        self.cuda = False
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
                self.cuda = True
        except Exception:
            self.cuda = False

    def close(self) -> None:
        if self.end_cpu_ns is not None:
            return
        self.end_cpu_ns = time.perf_counter_ns()
        if self.cuda and self.end_event is not None:
            self.end_event.record()

    def elapsed_ns(self) -> int:
        self.close()
        if self.cuda and self.start_event is not None and self.end_event is not None:
            self.end_event.synchronize()
            return max(0, int(round(float(self.start_event.elapsed_time(self.end_event)) * 1_000_000.0)))
        assert self.end_cpu_ns is not None
        return max(0, int(self.end_cpu_ns - self.start_cpu_ns))


class CaptureSession:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        capture = config["capture"]
        self.output_dir = Path(config["output_dir"])
        self.global_rank = _env_int("RANK", "SLURM_PROCID", default=0)
        self.global_world_size = _env_int("WORLD_SIZE", default=1)
        mapping = dict(capture.get("global_rank_to_source_rank", {}))
        self.source_rank = int(mapping.get(str(self.global_rank), self.global_rank))
        explicit_nodes = capture.get("rank_to_node")
        self.rank_to_node = None if explicit_nodes is None else tuple(int(v) for v in explicit_nodes)
        self.request_id = str(capture["request_id"])
        self.sample_prefix = str(capture["sample_id_prefix"])
        self._explicit_decode_step: int | None = None
        self._layer_call_count: dict[int, int] = {}
        self._object_layer_ids: dict[int, int] = {}
        self._next_fallback_layer_id = int(capture.get("layer_id_offset", 0))
        self._intervals: list[_EventInterval] = []
        self._active_intervals: dict[tuple[str, int, int], _EventInterval] = {}
        self._dispatch_postprocess_ns: dict[tuple[int, int], int] = {}
        self._pending_moe_output: tuple[int, int, _EventInterval] | None = None
        self._fate_inputs: dict[tuple[int, int], SampledGateInput] = {}
        self._lock = threading.RLock()
        self._flushed = False
        self._capture_enabled = True
        self._performance_eligible = False
        self._qualification_evidence: dict[str, Any] = {}
        self.writer = RankArtifactWriter(
            self.output_dir, global_rank=self.global_rank, source_rank=self.source_rank
        )
        self._write_manifest(status="ACTIVE")
        atexit.register(self.flush)

    @property
    def capture_enabled(self) -> bool:
        return bool(self._capture_enabled)

    def set_performance_qualification(self, *, eligible: bool, evidence: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._performance_eligible = bool(eligible)
            self._qualification_evidence = dict(evidence or {})

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            enabled = bool(enabled)
            if self._capture_enabled == enabled:
                return
            self._capture_enabled = enabled
            if not enabled:
                # Warmup must leave no partial timing or cross-sample state.
                for interval in self._active_intervals.values():
                    interval.close()
                self._active_intervals.clear()
                if self._pending_moe_output is not None:
                    _, _, interval = self._pending_moe_output
                    try:
                        self._intervals.remove(interval)
                    except ValueError:
                        pass
                    self._pending_moe_output = None

    def finish_sample(self, *, decode_step: int) -> None:
        """Close a forward boundary without inventing a next-router interval."""
        step = int(decode_step)
        with self._lock:
            active = [key for key in self._active_intervals if key[2] == step]
            if active:
                raise RuntimeError(f"sample {step} ended with active capture intervals: {active}")
            if self._pending_moe_output is not None:
                _, pending_step, interval = self._pending_moe_output
                if int(pending_step) == step:
                    # Final MoE layer has no next router in this sample.  Drop the
                    # open interval so the following sample cannot close it.
                    try:
                        self._intervals.remove(interval)
                    except ValueError:
                        pass
                    self._pending_moe_output = None
            for key in [key for key in self._fate_inputs if key[0] == step]:
                self._fate_inputs.pop(key, None)

    @property
    def strict(self) -> bool:
        return bool(self.config["capture"].get("strict", True))

    @property
    def fate_enabled(self) -> bool:
        prediction = self.config.get("prediction", {})
        return (
            str(prediction.get("mode", "")).upper() == "FATE_P2"
            and str(prediction.get("provider", "")).upper() == "MEGATRON_SAMPLED_FATE"
        )

    def set_context(self, *, request_id: str | None = None, decode_step: int | None = None) -> None:
        with self._lock:
            if request_id is not None:
                self.request_id = str(request_id)
            self._explicit_decode_step = None if decode_step is None else int(decode_step)

    def layer_id_for(self, module: Any) -> int:
        for owner in (module, getattr(module, "router", None), getattr(module, "config", None)):
            if owner is None:
                continue
            for name in ("layer_number", "layer_id", "layer_idx"):
                value = getattr(owner, name, None)
                if value is not None:
                    try:
                        return int(value) + int(self.config["capture"].get("layer_id_offset", 0))
                    except (TypeError, ValueError):
                        pass
        key = id(module)
        with self._lock:
            if key not in self._object_layer_ids:
                self._object_layer_ids[key] = self._next_fallback_layer_id
                self._next_fallback_layer_id += 1
            return self._object_layer_ids[key]

    def invocation_step(self, layer_id: int, *, advance: bool = False) -> int:
        if self._explicit_decode_step is not None:
            return int(self._explicit_decode_step)
        with self._lock:
            current = self._layer_call_count.get(int(layer_id), 0)
            if advance:
                self._layer_call_count[int(layer_id)] = current + 1
            return current

    def completed_invocation_step(self, layer_id: int) -> int:
        """Return the step owned by post-router/compute callbacks.

        Explicit runner contexts already identify the current sample and must
        never be decremented.  Legacy implicit counting advances at preprocess,
        so downstream callbacks use the previous counter value.
        """
        if self._explicit_decode_step is not None:
            return int(self._explicit_decode_step)
        return max(0, int(self.invocation_step(layer_id)) - 1)

    def sample_id(self, decode_step: int) -> str:
        return f"{self.sample_prefix}:step{int(decode_step)}"

    def capture_fate_gate_input(
        self, *, layer_id: int, hidden_states: Any, decode_step: int
    ) -> None:
        if not self.capture_enabled or not self.fate_enabled:
            return
        try:
            sample = capture_sampled_gate_input(
                hidden_states,
                layer_id=int(layer_id),
                decode_step=int(decode_step),
                max_sample_tokens=int(self.config["prediction"].get("max_sample_tokens", 2048)),
            )
            with self._lock:
                self._fate_inputs[(int(decode_step), int(layer_id))] = sample
        except Exception as exc:
            self.warning(
                "FATE_GATE_INPUT_CAPTURE_FAILED", str(exc), layer_id=layer_id,
                decode_step=decode_step, exception_type=type(exc).__name__,
            )
            if self.strict:
                raise

    def resolve_fate_for_target(
        self, *, target_module: Any, target_layer_id: int, decode_step: int
    ) -> None:
        if not self.capture_enabled or not self.fate_enabled:
            return
        source_layer_id = int(target_layer_id) - 1
        if source_layer_id < 0:
            return
        key = (int(decode_step), source_layer_id)
        with self._lock:
            sample = self._fate_inputs.pop(key, None)
        if sample is None:
            return
        try:
            router = getattr(target_module, "router", None)
            weight = None if router is None else getattr(router, "weight", None)
            if weight is None and router is not None:
                for owner_name in ("gate", "linear", "router"):
                    owner = getattr(router, owner_name, None)
                    value = getattr(owner, "weight", None) if owner is not None else None
                    if value is not None:
                        weight = value
                        break
            if weight is None or getattr(weight, "ndim", None) != 2:
                raise OnlineFateError("target router has no inferable expert dimension")
            num_experts = int(weight.shape[0])
            dispatcher = getattr(target_module, "token_dispatcher", None)
            expert_to_rank = tuple(self._expert_to_rank(
                num_experts=num_experts,
                local_expert_indices=getattr(dispatcher, "local_expert_indices", None),
            ))
            world_size = len(self.rank_to_node) if self.rank_to_node is not None else self.global_world_size
            row, evidence = predict_rank_row(
                sample, target_module=target_module, expert_to_rank=expert_to_rank,
                world_size=int(world_size),
            )
            payload = {
                "schema_version": "RS_SIM_CAPTURE_FATE_P2_ROW",
                "capture_id": self.config["capture"]["capture_id"],
                "model_id": self.config["capture"]["model_id"],
                "sample_id": self.sample_id(int(decode_step)),
                "request_id": self.request_id,
                "decode_step": int(decode_step),
                "source_layer_id": int(source_layer_id),
                "target_layer_id": int(target_layer_id),
                "source_rank": int(self.source_rank),
                "world_size": int(world_size),
                "routing_rows_by_destination": list(row),
                "confidence_ppm": int(self.config["prediction"].get("confidence_ppm", 750000)),
                **evidence,
            }
            payload["record_digest"] = canonical_fate_record_digest(payload)
            self.writer.append_fate(payload)
        except Exception as exc:
            self.warning(
                "FATE_P2_RESOLUTION_FAILED", str(exc), layer_id=target_layer_id,
                decode_step=decode_step, exception_type=type(exc).__name__,
            )
            if self.strict:
                raise

    def record_routing(
        self,
        *,
        layer_id: int,
        routing_map: Any,
        probs: Any | None = None,
        raw_routing_map: Any | None = None,
        explicit_padding_rows: Any | None = None,
        local_expert_indices: Any | None = None,
        drop_and_pad: bool = False,
        metadata: dict[str, Any] | None = None,
        decode_step: int | None = None,
    ) -> None:
        if not self.capture_enabled:
            return
        step = self.invocation_step(layer_id) if decode_step is None else int(decode_step)
        try:
            result = extract_routing_counts(
                routing_map=routing_map,
                probs=probs,
                raw_routing_map=raw_routing_map,
                explicit_padding_rows=explicit_padding_rows,
                drop_and_pad=bool(drop_and_pad),
                infer_padding_from_zero_prob=bool(
                    self.config["capture"].get("infer_padding_from_zero_prob", False)
                ),
            )
            expert_to_rank = self._expert_to_rank(
                num_experts=result.num_experts,
                local_expert_indices=local_expert_indices,
            )
            world_size = len(self.rank_to_node) if self.rank_to_node is not None else max(
                self.global_world_size, max(expert_to_rank, default=0) + 1
            )
            rank_to_node = list(self.rank_to_node) if self.rank_to_node is not None else self._infer_rank_to_node(world_size)
            payload = {
                "schema_version": "RS_SIM_CAPTURE_SOURCE_EXPERT_COUNTS",
                "capture_id": self.config["capture"]["capture_id"],
                "collector_version": self.config["capture"]["collector_version"],
                "model_id": self.config["capture"]["model_id"],
                "sample_id": self.sample_id(step),
                "request_id": self.request_id,
                "decode_step": step,
                "layer_id": int(layer_id),
                "global_rank": self.global_rank,
                "source_rank": self.source_rank,
                "world_size": int(world_size),
                "num_experts": int(result.num_experts),
                "raw_selected_rows": list(result.raw_selected_rows),
                "kept_rows": list(result.kept_rows),
                "dropped_rows": list(result.dropped_rows),
                "padding_rows": list(result.padding_rows),
                "source_expert_counts": [
                    int(result.kept_rows[index]) + int(result.padding_rows[index])
                    for index in range(result.num_experts)
                ],
                "expert_to_rank_map": expert_to_rank,
                "rank_to_node": rank_to_node,
                "capture_quality": result.extraction_mode,
                "source_shape": list(result.source_shape),
                "drop_and_pad": bool(drop_and_pad),
                "metadata": dict(metadata or {}),
            }
            payload["record_digest"] = canonical_fate_record_digest(payload)
            self.writer.append_routing(payload)
        except Exception as exc:
            self.warning(
                "ROUTING_CAPTURE_FAILED",
                str(exc),
                layer_id=layer_id,
                decode_step=step,
                exception_type=type(exc).__name__,
            )
            if self.strict:
                raise

    def _expert_to_rank(self, *, num_experts: int, local_expert_indices: Any | None) -> list[int]:
        explicit = self.config["capture"].get("expert_to_rank")
        if explicit is not None:
            result = [int(v) for v in explicit]
            if len(result) != num_experts:
                raise RoutingExtractionError(
                    f"configured expert_to_rank length {len(result)} != captured num_experts {num_experts}"
                )
            return result
        # Standard Megatron EP partitions global experts contiguously.  This is
        # only inferred when the captured routing columns are global experts and
        # divide exactly over the configured/source world size.
        world_size = len(self.rank_to_node) if self.rank_to_node is not None else self.global_world_size
        if world_size > 0 and num_experts % world_size == 0:
            per_rank = num_experts // world_size
            return [min(world_size - 1, expert // per_rank) for expert in range(num_experts)]
        if local_expert_indices is not None:
            try:
                indices = [int(v) for v in local_expert_indices]
                if len(indices) == num_experts:
                    return [self.source_rank for _ in indices]
            except Exception:
                pass
        raise RoutingExtractionError(
            "expert_to_rank cannot be inferred safely; set capture.expert_to_rank explicitly"
        )

    def _infer_rank_to_node(self, world_size: int) -> list[int]:
        local_world = _env_int("LOCAL_WORLD_SIZE", default=world_size)
        if local_world <= 0:
            local_world = world_size
        return [rank // local_world for rank in range(world_size)]

    def start_interval(self, kind: str, *, layer_id: int, decode_step: int | None = None) -> None:
        if not self.capture_enabled or not bool(self.config["capture"].get("capture_compute", True)):
            return
        step = self.invocation_step(layer_id) if decode_step is None else int(decode_step)
        key = (str(kind), int(layer_id), step)
        with self._lock:
            if key in self._active_intervals:
                self.warning("DUPLICATE_INTERVAL_START", f"interval already active: {key}", layer_id=layer_id, decode_step=step)
                return
            interval = _EventInterval(
                kind=str(kind), layer_id=int(layer_id), request_id=self.request_id, decode_step=step, rank=self.source_rank
            )
            self._active_intervals[key] = interval
            self._intervals.append(interval)

    def end_interval(self, kind: str, *, layer_id: int, decode_step: int | None = None) -> None:
        if not self.capture_enabled or not bool(self.config["capture"].get("capture_compute", True)):
            return
        step = self.invocation_step(layer_id) if decode_step is None else int(decode_step)
        key = (str(kind), int(layer_id), step)
        with self._lock:
            interval = self._active_intervals.pop(key, None)
            if interval is None:
                self.warning("MISSING_INTERVAL_START", f"no active interval: {key}", layer_id=layer_id, decode_step=step)
                return
            interval.close()

    def mark_moe_output_ready(self, *, layer_id: int, decode_step: int) -> None:
        if not self.capture_enabled or not bool(self.config["capture"].get("capture_compute", True)):
            return
        interval = _EventInterval(
            kind="combine_release_to_router_ready_ns",
            layer_id=layer_id,
            request_id=self.request_id,
            decode_step=decode_step,
            rank=self.source_rank,
        )
        with self._lock:
            self._pending_moe_output = (int(layer_id), int(decode_step), interval)
            self._intervals.append(interval)

    def close_previous_moe_to_router(self, *, next_layer_id: int, decode_step: int) -> None:
        if not self.capture_enabled:
            return
        with self._lock:
            pending = self._pending_moe_output
            if pending is None:
                return
            previous_layer, previous_step, interval = pending
            if previous_step == int(decode_step) and previous_layer == int(next_layer_id):
                return
            interval.close()
            self._pending_moe_output = None

    def warning(self, code: str, message: str, **context: Any) -> None:
        payload = {
            "schema_version": "RS_SIM_CAPTURE_WARNING",
            "capture_id": self.config["capture"]["capture_id"],
            "global_rank": self.global_rank,
            "source_rank": self.source_rank,
            "code": str(code),
            "message": str(message),
            "context": context,
            "time_unix_ns": time.time_ns(),
        }
        self.writer.append_warning(payload)

    def flush(self) -> None:
        with self._lock:
            if self._flushed:
                return
            self._flushed = True
            for interval in self._active_intervals.values():
                interval.close()
            self._active_intervals.clear()
            if self._pending_moe_output is not None:
                # No next router exists for the final layer.  Keep the field
                # absent rather than treating process-exit time as pure compute.
                self._pending_moe_output = None

        grouped: dict[tuple[str, int, int], dict[str, list[int]]] = {}
        for interval in self._intervals:
            if interval.end_cpu_ns is None:
                continue
            value = interval.elapsed_ns()
            key = (interval.request_id, interval.decode_step, interval.layer_id)
            grouped.setdefault(key, {}).setdefault(interval.kind, []).append(value)
        for (request_id, step, layer_id), values in sorted(grouped.items()):
            reduced = {name: int(round(sum(rows) / len(rows))) for name, rows in values.items() if rows}
            routed_total = reduced.pop("routed_experts_compute_total_ns", None)
            dispatch_post = reduced.get("dispatch_local_postprocess_ns")
            if routed_total is not None:
                reduced["dispatch_release_to_combine_source_ready_ns"] = max(
                    0, int(routed_total) - int(dispatch_post or 0)
                )
            payload = {
                "schema_version": "RS_SIM_CAPTURE_LOCAL_COMPUTE",
                "capture_id": self.config["capture"]["capture_id"],
                "collector_version": self.config["capture"]["collector_version"],
                "sample_id": self.sample_id(step),
                "request_id": request_id,
                "decode_step": step,
                "layer_id": layer_id,
                "global_rank": self.global_rank,
                "source_rank": self.source_rank,
                "measurement_method": "cuda_event_auto_megatron" if any(i.cuda for i in self._intervals) else "cpu_perf_counter_auto",
                "performance_eligible": bool(self._performance_eligible),
                "performance_qualification": dict(self._qualification_evidence),
                "field_values_ns": reduced,
                "field_quality": {
                    "router_and_pack_ns": "AUTO_MEGATRON_ROUTE_TO_PREPROCESS",
                    "dispatch_local_postprocess_ns": "AUTO_DISPATCH_POSTPROCESS",
                    "dispatch_release_to_combine_source_ready_ns": "AUTO_ROUTED_EXPERTS_MINUS_POSTPROCESS",
                    "combine_release_to_router_ready_ns": "AUTO_MOE_OUTPUT_TO_NEXT_ROUTER",
                    "bootstrap_router_and_pack_ns": "FALLBACK_UNLESS_EXPLICIT_MARKER",
                },
            }
            payload["record_digest"] = canonical_fate_record_digest(payload)
            self.writer.append_compute(payload)
        self._write_manifest(status="FLUSHED")

    def _write_manifest(self, *, status: str) -> None:
        capture = self.config["capture"]
        payload = {
            "schema_version": "RS_SIM_CAPTURE_RANK_MANIFEST",
            "status": status,
            "capture_id": capture["capture_id"],
            "collector_version": capture["collector_version"],
            "backend": capture["backend"],
            "model_id": capture["model_id"],
            "model_path": capture.get("model_path"),
            "global_rank": self.global_rank,
            "global_world_size": self.global_world_size,
            "source_rank": self.source_rank,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank_to_node": list(self.rank_to_node) if self.rank_to_node is not None else None,
            "capture_compute": bool(capture.get("capture_compute", True)),
            "strict": self.strict,
            "performance_eligible": bool(self._performance_eligible),
            "performance_qualification": dict(self._qualification_evidence),
        }
        payload["manifest_digest"] = _sha256_json(payload)
        self.writer.write_manifest(payload)
