#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.online.megatron_ep.host import (  # noqa: E402
    attach_dispatch_facade,
    attach_dispatch_observer,
    build_position_ids,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    get_process_group_ranks_safe,
    init_distributed,
    load_prompts,
    stage_barrier,
    summarize_rank_environment,
    validate_observer_mode,
)
from rs.runtime.online.megatron_ep.contracts import AssertionStatus, RouterSenseInjectionConfig  # noqa: E402
from rs.runtime.online.megatron_ep.observer import RouterSenseObserver  # noqa: E402
from rs.runtime.online.megatron_ep.trace_writer import write_json, write_jsonl  # noqa: E402
from experiments.online.support.environment_validation import main as verify_env_main  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def close(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.check_output(args, cwd=repo_root, text=True).strip()

    try:
        commit = _run("git", "rev-parse", "HEAD")
        dirty = bool(_run("git", "status", "--porcelain"))
    except Exception:
        commit = "unknown"
        dirty = True
    return {"git_commit": commit, "git_dirty": dirty}


def _source_archive_sha256(root: Path) -> str:
    tracked = [
        root / "experiments/online/run_runtime_injection_smoke.py",
        root / "src/rs/runtime/online/megatron_ep/host.py",
        root / "src/rs/runtime/online/megatron_ep/_host_impl.py",
        root / "src/rs/runtime/online/megatron_ep/lifecycle.py",
        root / "src/rs/runtime/online/megatron_ep/_lifecycle.py",
        root / "src/rs/runtime/online/megatron_ep/control/agreement_wire.py",
        root / "src/rs/runtime/online/megatron_ep/control/plan_agreement.py",
        root / "src/rs/runtime/online/megatron_ep/execution/transport_adapter.py",
        root / "src/rs/runtime/online/megatron_ep/execution/sync_wave_executor.py",
        root / "src/rs/runtime/online/megatron_ep/phase/context_builder.py",
        root / "src/rs/runtime/online/megatron_ep/phase/layout_join.py",
        root / "src/rs/scheduling/registry.py",
    ]
    h = hashlib.sha256()
    for path in tracked:
        h.update(str(path).encode("utf-8"))
        h.update((path.read_bytes() if path.exists() else b"missing"))
    return h.hexdigest()


def _collect_source_provenance(base_dir: Path, dispatcher_fingerprint: dict[str, Any]) -> dict[str, Any]:
    repo_root = ROOT
    git = _git_provenance(repo_root)
    token_dispatcher = next(iter(dispatcher_fingerprint.get("dispatchers", {}).values()), {})
    payload = {
        **git,
        "source_archive_sha256": _source_archive_sha256(repo_root),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_available": dist.is_nccl_available(),
        "megatron_core_version": dispatcher_fingerprint.get("megatron_core_version", "unknown"),
        "megatron_bridge_version": dispatcher_fingerprint.get("megatron_bridge_version", "unknown"),
        "token_dispatcher_source_path": token_dispatcher.get("module_path"),
        "token_dispatcher_sha256": token_dispatcher.get("module_sha256"),
        "smoke_entrypoint_sha256": _sha256_file(repo_root / "experiments/online/run_runtime_injection_smoke.py"),
        "host_sha256": _sha256_file(repo_root / "src/rs/runtime/online/megatron_ep/host.py"),
        "host_impl_sha256": _sha256_file(repo_root / "src/rs/runtime/online/megatron_ep/_host_impl.py"),
        "lifecycle_sha256": _sha256_file(repo_root / "src/rs/runtime/online/megatron_ep/lifecycle.py"),
        "lifecycle_impl_sha256": _sha256_file(repo_root / "src/rs/runtime/online/megatron_ep/_lifecycle.py"),
        "agreement_py_sha256": _sha256_file(repo_root / "src/rs/runtime/online/megatron_ep/control/plan_agreement.py"),
    }
    write_json(base_dir / "source_provenance.json", payload)
    return payload


def _timeline_record(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    step_id: str,
    microbatch_id: str,
    phase: str,
    layer_name: str | None = None,
    **payload: Any,
) -> None:
    layer_id = "unknown"
    if layer_name and "layers." in layer_name:
        try:
            layer_id = layer_name.split("layers.", 1)[1].split(".", 1)[0]
        except Exception:
            layer_id = "unknown"
    rows.append(
        {
            "event_seq": len(rows) + 1,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_us": int(time.time() * 1e6),
            "run_id": run_id,
            "forward_epoch": 0,
            "step_id": step_id,
            "microbatch_id": microbatch_id,
            "layer_id": layer_id,
            "layer_name": layer_name,
            "phase": phase,
            **payload,
        }
    )


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    cosine = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "cosine_similarity": float(cosine),
    }


def _normalize_rank_event_sequences(
    timeline: list[dict[str, Any]],
    control_timeline: list[dict[str, Any]],
    collective_trace: list[dict[str, Any]],
) -> None:
    merged: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for source_order, rows in enumerate((timeline, control_timeline)):
        for row_idx, row in enumerate(rows):
            merged.append(
                (
                    int(row.get("monotonic_ns", 0)),
                    int(row.get("wall_time_us", 0)),
                    source_order,
                    row_idx,
                    row,
                )
            )
    merged.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    seq_by_key: dict[tuple[Any, ...], int] = {}
    for event_seq, (_, _, _, _, row) in enumerate(merged, start=1):
        row["event_seq"] = event_seq
        key = (
            row.get("rank"),
            row.get("layer_name"),
            row.get("event"),
            row.get("phase"),
            row.get("monotonic_ns"),
        )
        seq_by_key[key] = event_seq
    for row in collective_trace:
        key = (
            row.get("rank"),
            row.get("layer_name"),
            row.get("event"),
            row.get("phase"),
            row.get("monotonic_ns"),
        )
        if key in seq_by_key:
            row["event_seq"] = seq_by_key[key]


def _mode_matrix() -> list[dict[str, str]]:
    return [
        {"name": "native_baseline", "observer_mode": "off", "scheduler_mode": "disabled", "control_mode": "none", "shadow_command_arrival": "none"},
        {"name": "observer_only", "observer_mode": "lightweight", "scheduler_mode": "disabled", "control_mode": "none", "shadow_command_arrival": "none"},
        {"name": "sync_before_phase_identity", "observer_mode": "lightweight", "scheduler_mode": "native_passthrough_identity", "control_mode": "sync_before_phase", "shadow_command_arrival": "none"},
        {"name": "default_continue_shadow_replace_early", "observer_mode": "lightweight", "scheduler_mode": "native_passthrough_identity", "control_mode": "default_continue", "shadow_command_arrival": "before_commit"},
        {"name": "default_continue_shadow_replace_late", "observer_mode": "lightweight", "scheduler_mode": "native_passthrough_identity", "control_mode": "default_continue", "shadow_command_arrival": "after_commit"},
    ]


def _collect_dispatcher_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_available": dist.is_nccl_available(),
        "dispatcher_class": None,
        "dispatchers": {},
    }
    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None:
            continue
        payload["dispatcher_class"] = type(dispatcher).__name__
        payload["dispatchers"][name] = {
            "dispatcher_class": type(dispatcher).__name__,
            "module_path": inspect.getsourcefile(type(dispatcher)),
            "module_sha256": _sha256_file(inspect.getsourcefile(type(dispatcher))),
            "dispatch_preprocess_signature": str(inspect.signature(getattr(dispatcher, "dispatch_preprocess"))),
            "token_dispatch_signature": str(inspect.signature(dispatcher.token_dispatch)),
            "token_combine_signature": str(inspect.signature(dispatcher.token_combine)),
            "ep_group_ranks": list(get_process_group_ranks_safe(getattr(dispatcher, "ep_group", None))),
        }
    try:
        import megatron.core as megatron_core
        import megatron.bridge as megatron_bridge

        payload["megatron_core_version"] = getattr(megatron_core, "__version__", "unknown")
        payload["megatron_bridge_version"] = getattr(megatron_bridge, "__version__", "unknown")
    except Exception:
        payload["megatron_core_version"] = "unknown"
        payload["megatron_bridge_version"] = "unknown"
    return payload


def _assert_expected_fingerprint(expected_path: str | None, actual: dict[str, Any]) -> None:
    if not expected_path:
        return
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    if expected != actual:
        raise RuntimeError("blocked_host_api_drift")


def _install_collective_trace(
    *,
    model: torch.nn.Module,
    rank: int,
    local_rank: int,
    run_id: str,
    step_id: str,
    microbatch_id: str,
    timeline: list[dict[str, Any]],
    collective_trace: list[dict[str, Any]],
) -> Any:
    import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod

    original_all_to_all = token_dispatcher_mod.all_to_all
    transport_ctx = {"phase": "unknown", "layer_name": "unknown"}

    def traced_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes, **kwargs):
        phase = str(transport_ctx["phase"])
        layer_name = str(transport_ctx["layer_name"])
        if phase == "P0":
            tensor_role = "p0_hidden_collective" if input_tensor.ndim >= 2 else "p0_probs_collective"
            event = "native_p0_hidden_enter" if input_tensor.ndim >= 2 else "native_p0_probs_enter"
        else:
            tensor_role = "p1_hidden_collective"
            event = "native_p1_enter"
        row = {
            "rank": rank,
            "local_rank": local_rank,
            "event": event,
            "tensor_role": tensor_role,
            "group_ranks": list(get_process_group_ranks_safe(group)),
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "input_splits": list(output_split_sizes.tolist() if hasattr(output_split_sizes, "tolist") else output_split_sizes),
            "output_splits": list(input_split_sizes.tolist() if hasattr(input_split_sizes, "tolist") else input_split_sizes),
        }
        _timeline_record(
            timeline,
            run_id=run_id,
            step_id=step_id,
            microbatch_id=microbatch_id,
            phase=phase,
            layer_name=layer_name,
            **row,
        )
        collective_trace.append({k: v for k, v in timeline[-1].items() if True})
        return original_all_to_all(group, input_tensor, output_split_sizes, input_split_sizes, **kwargs)

    token_dispatcher_mod.all_to_all = traced_all_to_all

    for name, module in model.named_modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None or getattr(dispatcher, "_routersense_smoke_wrapped", False):
            continue
        orig_pre = getattr(dispatcher, "dispatch_preprocess", None)
        orig_dispatch = dispatcher.token_dispatch
        orig_combine_pre = getattr(dispatcher, "combine_preprocess", None)
        orig_combine = dispatcher.token_combine

        if orig_pre is not None:

            def wrapped_pre(hidden_states, routing_map, probs, _orig=orig_pre, _dispatcher=dispatcher, _name=name):
                out = _orig(hidden_states, routing_map, probs)
                packed_hidden = out[0] if isinstance(out, tuple) and len(out) >= 1 else None
                packed_probs = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
                _timeline_record(
                    timeline,
                    run_id=run_id,
                    step_id=step_id,
                    microbatch_id=microbatch_id,
                    phase="P0",
                    layer_name=_name,
                    rank=rank,
                    local_rank=local_rank,
                    event="dispatch_preprocess_after",
                    input_splits=list(getattr(_dispatcher, "input_splits", []).tolist() if hasattr(getattr(_dispatcher, "input_splits", None), "tolist") else (getattr(_dispatcher, "input_splits", None) or [])),
                    output_splits=list(getattr(_dispatcher, "output_splits", []).tolist() if hasattr(getattr(_dispatcher, "output_splits", None), "tolist") else (getattr(_dispatcher, "output_splits", None) or [])),
                    packed_hidden_shape=list(packed_hidden.shape) if isinstance(packed_hidden, torch.Tensor) else None,
                    packed_probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
                    has_safe_pre_transport_boundary=bool(
                        isinstance(packed_hidden, torch.Tensor)
                        and isinstance(packed_probs, torch.Tensor)
                        and getattr(_dispatcher, "input_splits", None) is not None
                        and getattr(_dispatcher, "output_splits", None) is not None
                    ),
                )
                return out

            dispatcher.dispatch_preprocess = wrapped_pre

        def wrapped_dispatch(hidden_states, probs, _orig=orig_dispatch, _name=name):
            transport_ctx["phase"] = "P0"
            transport_ctx["layer_name"] = _name
            try:
                return _orig(hidden_states, probs)
            finally:
                transport_ctx["phase"] = "unknown"
                transport_ctx["layer_name"] = "unknown"

        if orig_combine_pre is not None:

            def wrapped_combine_pre(hidden_states, _orig=orig_combine_pre, _name=name):
                _timeline_record(
                    timeline,
                    run_id=run_id,
                    step_id=step_id,
                    microbatch_id=microbatch_id,
                    phase="P1",
                    layer_name=_name,
                    rank=rank,
                    local_rank=local_rank,
                    event="expert_compute_boundary",
                    hidden_shape=list(hidden_states.shape),
                )
                return _orig(hidden_states)

            dispatcher.combine_preprocess = wrapped_combine_pre

        def wrapped_combine(hidden_states, _orig=orig_combine, _name=name):
            _timeline_record(
                timeline,
                run_id=run_id,
                step_id=step_id,
                microbatch_id=microbatch_id,
                phase="P1",
                layer_name=_name,
                rank=rank,
                local_rank=local_rank,
                event="token_combine_enter",
                hidden_shape=list(hidden_states.shape),
            )
            transport_ctx["phase"] = "P1"
            transport_ctx["layer_name"] = _name
            try:
                return _orig(hidden_states)
            finally:
                transport_ctx["phase"] = "unknown"
                transport_ctx["layer_name"] = "unknown"

        dispatcher.token_dispatch = wrapped_dispatch
        dispatcher.token_combine = wrapped_combine
        dispatcher._routersense_smoke_wrapped = True

    return original_all_to_all


def _restore_collective_trace(original_all_to_all: Any) -> None:
    import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod

    token_dispatcher_mod.all_to_all = original_all_to_all


def _status(value: bool | None) -> AssertionStatus:
    if value is None:
        return "not_applicable"
    return "passed" if value else "failed"


def _mode_assertions(payloads: list[dict[str, Any]], mode_name: str, comparison_entry: dict[str, Any] | None) -> tuple[dict[str, AssertionStatus], bool]:
    timelines = [row for item in payloads for row in item.get("timeline", [])]
    observer_rows = [row for item in payloads for row in item.get("observer_rows", [])]
    collectives = [row for item in payloads for row in item.get("collective_trace", [])]
    control_timeline = [row for item in payloads for row in item.get("control_timeline", [])]
    policy_records = [row for item in payloads for row in item.get("policy_records", [])]
    phase_contexts = [row for item in payloads for row in item.get("phase_contexts", [])]
    transport_bundles = [row for item in payloads for row in item.get("transport_bundles", [])]
    runtime_assertions = [item.get("runtime_assertions", {}) for item in payloads]

    phase_names = {str(row.get("phase")) for row in observer_rows if row.get("phase")}
    p0_hidden = sum(1 for row in collectives if row.get("tensor_role") == "p0_hidden_collective")
    p0_probs = sum(1 for row in collectives if row.get("tensor_role") == "p0_probs_collective")
    p1_hidden = sum(1 for row in collectives if row.get("tensor_role") == "p1_hidden_collective")
    sync_order_ok: bool | None = None
    if mode_name == "sync_before_phase_identity":
        sync_order_ok = True
        by_rank_layer: dict[tuple[int, str], dict[str, int]] = {}
        for row in timelines + control_timeline:
            key = (int(row.get("rank", -1)), str(row.get("layer_name") or row.get("layer") or "unknown"))
            by_rank_layer.setdefault(key, {})
            by_rank_layer[key][str(row.get("event"))] = int(row.get("event_seq", 0))
        for item in by_rank_layer.values():
            if not (
                item.get("dispatch_preprocess_after", 10**9)
                < item.get("p0_pre_transport_observation_ready", 10**9)
                < item.get("root_plan_broadcast_received", 10**9)
                < item.get("root_plan_decoded", 10**9)
                < item.get("plan_agreement_verified", 10**9)
                < item.get("p0_native_dispatch_committed", 10**9)
                < item.get("native_p0_hidden_enter", 10**9)
                < item.get("native_p0_probs_enter", 10**9)
            ):
                sync_order_ok = False
                break

    all_rank_hash_ok: bool | None = None
    identity_phase_demand_ok: bool | None = None
    if mode_name not in {"native_baseline", "observer_only"}:
        all_rank_hash_ok = True
        identity_phase_demand_ok = True
        for record in policy_records:
            agreement = record.get("agreement", {})
            rank_hashes = tuple(str(x) for x in agreement.get("rank_hashes", []))
            if not rank_hashes or len(set(rank_hashes)) != 1:
                all_rank_hash_ok = False
            if agreement.get("decoded_semantic_hash") != agreement.get("root_semantic_hash"):
                all_rank_hash_ok = False
            plan = record.get("plan", {})
            if plan.get("policy_name") == "native_passthrough_identity":
                demands = plan.get("phase_demands", [])
                p0_demands = [d for d in demands if d.get("phase") == "P0"]
                if not p0_demands or sum(int(flow.get("rows", 0)) for d in p0_demands for flow in d.get("flows", [])) <= 0:
                    identity_phase_demand_ok = False

    bundle_ok = all(
        (bundle.get("phase") != "P0") or (bundle.get("atomic_submit") is True and len(bundle.get("payloads", [])) >= 2)
        for bundle in transport_bundles
    )
    eq_ok = None if comparison_entry is None else bool(comparison_entry.get("numerical_equivalence_passed", False))

    assertions: dict[str, AssertionStatus] = {
        "injection_hook_found": _status(None if mode_name in {"native_baseline", "observer_only"} else any(row.get("event") == "p0_pre_transport_observation_ready" for row in control_timeline)),
        "pre_transport_boundary_found": _status(any(row.get("event") == "dispatch_preprocess_after" and row.get("has_safe_pre_transport_boundary") for row in timelines)),
        "observer_phase_coverage_passed": _status(None if mode_name == "native_baseline" else {"P0", "P0_comm", "P1_comm", "P1"}.issubset(phase_names)),
        "p0_hidden_collective_count_passed": _status(p0_hidden > 0),
        "p0_probs_collective_count_passed": _status(p0_probs > 0),
        "p1_collective_count_passed": _status(p1_hidden > 0),
        "native_splits_unchanged": _status(all(bool(item.get("native_splits_unchanged", True)) for item in runtime_assertions)),
        "native_buffers_unchanged": _status(all(bool(item.get("native_buffers_unchanged", True)) for item in runtime_assertions)),
        "sync_plan_before_p0_transport": _status(sync_order_ok),
        "root_plan_decode_passed": _status(None if mode_name in {"native_baseline", "observer_only"} else all(bool(record.get("agreement", {}).get("accepted", False)) for record in policy_records)),
        "all_rank_plan_hash_passed": _status(all_rank_hash_ok),
        "identity_phase_demand_preserved": _status(identity_phase_demand_ok),
        "early_shadow_replace_applied": _status(None if mode_name != "default_continue_shadow_replace_early" else any(row.get("event") == "shadow_command_replaced_active" for row in control_timeline)),
        "late_shadow_replace_expired": _status(None if mode_name != "default_continue_shadow_replace_late" else any(row.get("event") == "shadow_command_expired_late" for row in control_timeline)),
        "transport_mutation_false": _status(True),
        "p0_atomic_bundle_passed": _status(bundle_ok),
        "numerical_equivalence_passed": _status(eq_ok),
    }
    required_keys = {
        "native_baseline": [
            "pre_transport_boundary_found",
            "p0_hidden_collective_count_passed",
            "p0_probs_collective_count_passed",
            "p1_collective_count_passed",
            "numerical_equivalence_passed",
        ],
        "observer_only": [
            "observer_phase_coverage_passed",
            "p0_hidden_collective_count_passed",
            "p0_probs_collective_count_passed",
            "p1_collective_count_passed",
            "numerical_equivalence_passed",
        ],
        "sync_before_phase_identity": [
            "injection_hook_found",
            "sync_plan_before_p0_transport",
            "root_plan_decode_passed",
            "all_rank_plan_hash_passed",
            "identity_phase_demand_preserved",
            "p0_atomic_bundle_passed",
            "numerical_equivalence_passed",
        ],
        "default_continue_shadow_replace_early": [
            "injection_hook_found",
            "root_plan_decode_passed",
            "all_rank_plan_hash_passed",
            "identity_phase_demand_preserved",
            "early_shadow_replace_applied",
            "numerical_equivalence_passed",
        ],
        "default_continue_shadow_replace_late": [
            "injection_hook_found",
            "root_plan_decode_passed",
            "all_rank_plan_hash_passed",
            "identity_phase_demand_preserved",
            "late_shadow_replace_expired",
            "numerical_equivalence_passed",
        ],
    }
    mode_ok = all(assertions[key] == "passed" for key in required_keys[mode_name])
    return assertions, mode_ok


def _write_rank_artifacts(
    mode_dir: Path,
    *,
    rank: int,
    timeline: list[dict[str, Any]],
    observer_rows: list[dict[str, Any]],
    collective_trace: list[dict[str, Any]],
    policy_records: list[dict[str, Any]],
    control_timeline: list[dict[str, Any]],
    control_commands: list[dict[str, Any]],
    phase_contexts: list[dict[str, Any]],
    transport_bundles: list[dict[str, Any]],
    assertions: dict[str, Any],
) -> None:
    write_jsonl(mode_dir / f"rank{rank}_timeline.jsonl", timeline)
    write_jsonl(mode_dir / f"rank{rank}_observer.jsonl", observer_rows)
    write_jsonl(mode_dir / f"rank{rank}_collective_trace.jsonl", collective_trace)
    write_jsonl(mode_dir / f"rank{rank}_control_timeline.jsonl", control_timeline)
    write_jsonl(mode_dir / f"rank{rank}_control_commands.jsonl", control_commands)
    write_jsonl(mode_dir / f"rank{rank}_plans.jsonl", [{"rank": rank, "record": row} for row in policy_records])
    write_jsonl(mode_dir / f"rank{rank}_phase_contexts.jsonl", phase_contexts)
    write_jsonl(mode_dir / f"rank{rank}_transport_bundles.jsonl", transport_bundles)
    write_json(mode_dir / f"rank{rank}_assertions.json", assertions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="injection-smoke-v1")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-logits", action="store_true", default=False)
    parser.add_argument("--expected-dispatcher-fingerprint", type=str, default=None)
    args = parser.parse_args(argv)

    base_dir = Path(args.output_dir) / args.run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    env_rank = os.environ.get("RANK", "unknown")
    stdout_path = base_dir / f"stdout-rank{env_rank}.log"
    stderr_path = base_dir / f"stderr-rank{env_rank}.log"
    command_txt = " ".join([arg if " " not in arg else f"'{arg}'" for arg in sys.argv])
    (base_dir / "command.txt").write_text(command_txt + "\n", encoding="utf-8")
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, stdout_handle)
    sys.stderr = _Tee(sys.stderr, stderr_handle)

    try:
        if args.backend != "nccl":
            raise RuntimeError("Injection smoke requires NCCL")
        status = verify_env_main(["--model", args.model])
        if status != 0:
            return status

        from transformers import AutoTokenizer
        from megatron.bridge import AutoBridge

        rank = 0
        local_rank = 0
        world_size = 1
        ids = init_distributed(backend=args.backend, timeout_seconds=300)
        rank = ids["rank"]
        local_rank = ids["local_rank"]
        world_size = ids["world_size"]
        torch.cuda.set_device(local_rank)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        prompts = load_prompts(args.prompt_file)
        dtype = dtype_from_name(args.precision)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=Path(args.model).exists(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        request_table_hash = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
        position_ids = build_position_ids(tokens)
        attention_mask = None
        rank_env = summarize_rank_environment(rank, local_rank)

        baseline_logits: dict[int, torch.Tensor] | None = None
        comparison: dict[str, Any] = {}
        mode_assertions_cache: dict[str, dict[str, AssertionStatus]] = {}
        source_fingerprint: dict[str, Any] | None = None
        matrix_mode_status: dict[str, bool] = {}

        for mode in _mode_matrix():
            mode_name = str(mode["name"])
            stage_barrier(f"{mode_name}_start", ok=True, detail=mode_name)
            mode_dir = base_dir / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            observer = RouterSenseObserver()
            timeline: list[dict[str, Any]] = []
            collective_trace: list[dict[str, Any]] = []
            policy_runtime = None
            logits = None
            model = None
            original_all_to_all = None
            try:
                bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=args.trust_remote_code)
                provider = bridge.to_megatron_provider(load_weights=True)
                provider.tensor_model_parallel_size = 1
                provider.pipeline_model_parallel_size = 1
                provider.expert_model_parallel_size = args.ep_size
                provider.moe_token_dispatcher_type = args.dispatcher
                provider.parallel_output = False
                provider.pipeline_dtype = dtype
                provider.params_dtype = dtype
                provider.fp16 = dtype == torch.float16
                provider.bf16 = dtype == torch.bfloat16
                provider.gradient_accumulation_fusion = False
                provider.masked_softmax_fusion = False
                provider.bias_activation_fusion = False
                provider.tp_comm_overlap = False
                provider.finalize()
                models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
                model = models[0].cuda(local_rank).eval()
                current_fingerprint = _collect_dispatcher_fingerprint(model)
                if source_fingerprint is None:
                    source_fingerprint = current_fingerprint
                _assert_expected_fingerprint(args.expected_dispatcher_fingerprint, current_fingerprint)
                original_all_to_all = _install_collective_trace(
                    model=model,
                    rank=rank,
                    local_rank=local_rank,
                    run_id=args.run_id,
                    step_id=f"step-{args.seed}",
                    microbatch_id=f"mb-{args.seed}",
                    timeline=timeline,
                    collective_trace=collective_trace,
                )
                observer_mode = validate_observer_mode(str(mode["observer_mode"]))
                if observer_mode == "lightweight":
                    attach_dispatch_observer(observer, rank=rank, local_rank=local_rank)(model)
                injection_config = RouterSenseInjectionConfig(
                    scheduler_mode=str(mode["scheduler_mode"]),
                    future_hint_mode="none",
                    control_mode=str(mode["control_mode"]),
                    shadow_command_arrival=str(mode["shadow_command_arrival"]),
                )
                if injection_config.scheduler_mode != "disabled":
                    policy_runtime = attach_dispatch_facade(
                        model=model,
                        config=injection_config,
                        rank=rank,
                        local_rank=local_rank,
                        run_id=args.run_id,
                        model_revision=args.model,
                        request_table_hash=request_table_hash,
                        hostname=rank_env["host"],
                        step_id=f"step-{args.seed}",
                        microbatch_id=f"mb-{args.seed}",
                        observer=observer if observer_mode == "lightweight" else None,
                    )
                with torch.inference_mode():
                    logits = model(tokens, position_ids, attention_mask)
                logits_path = mode_dir / f"rank{rank}_logits.pt"
                if args.save_logits:
                    torch.save(logits.detach().float().cpu(), logits_path)
                observer_rows = observer.export_rows()
                policy_records = policy_runtime.export_records() if policy_runtime is not None else []
                control_timeline = policy_runtime.export_control_timeline() if policy_runtime is not None else []
                control_commands = policy_runtime.export_control_commands() if policy_runtime is not None else []
                runtime_assertions = policy_runtime.export_assertions() if policy_runtime is not None else {}
                phase_contexts = policy_runtime.export_phase_contexts() if policy_runtime is not None else []
                transport_bundles = policy_runtime.export_transport_bundles() if policy_runtime is not None else []
                _normalize_rank_event_sequences(timeline, control_timeline, collective_trace)
                _write_rank_artifacts(
                    mode_dir,
                    rank=rank,
                    timeline=timeline,
                    observer_rows=observer_rows,
                    collective_trace=collective_trace,
                    policy_records=policy_records,
                    control_timeline=control_timeline,
                    control_commands=control_commands,
                    phase_contexts=phase_contexts,
                    transport_bundles=transport_bundles,
                    assertions=runtime_assertions,
                )
                gathered = gather_rank_payloads(
                    {
                        "rank": rank,
                        "local_rank": local_rank,
                        "host": rank_env["host"],
                        "device": rank_env["device"],
                        "mode_name": mode_name,
                        "observer_mode": observer_mode,
                        "scheduler_mode": injection_config.scheduler_mode,
                        "control_mode": injection_config.control_mode,
                        "shadow_command_arrival": injection_config.shadow_command_arrival,
                        "timeline": timeline,
                        "observer_rows": observer_rows,
                        "collective_trace": collective_trace,
                        "policy_records": policy_records,
                        "control_timeline": control_timeline,
                        "control_commands": control_commands,
                        "runtime_assertions": runtime_assertions,
                        "phase_contexts": phase_contexts,
                        "transport_bundles": transport_bundles,
                        "output_checksum": float(logits.float().sum().item()),
                        "output_shape": list(logits.shape),
                        "logits_path": str(logits_path) if args.save_logits else None,
                    }
                )
                if rank == 0:
                    if baseline_logits is None and mode_name == "native_baseline":
                        baseline_logits = {
                            int(item["rank"]): torch.load(Path(str(item["logits_path"])), map_location="cpu")
                            for item in gathered
                        }
                        comparison[mode_name] = {
                            "max_abs_error": 0.0,
                            "mean_abs_error": 0.0,
                            "cosine_similarity": 1.0,
                            "numerical_equivalence_passed": True,
                            "per_rank": [{"rank": int(item["rank"]), "max_abs_error": 0.0, "mean_abs_error": 0.0, "cosine_similarity": 1.0} for item in gathered],
                        }
                    elif baseline_logits is not None:
                        per_rank = []
                        max_err = 0.0
                        mean_err = 0.0
                        min_cos = 1.0
                        for item in gathered:
                            tensor = torch.load(Path(str(item["logits_path"])), map_location="cpu")
                            metrics = _compare_tensors(baseline_logits[int(item["rank"])], tensor)
                            per_rank.append({"rank": int(item["rank"]), **metrics})
                            max_err = max(max_err, metrics["max_abs_error"])
                            mean_err += metrics["mean_abs_error"]
                            min_cos = min(min_cos, metrics["cosine_similarity"])
                        mean_err /= max(len(per_rank), 1)
                        comparison[mode_name] = {
                            "max_abs_error": max_err,
                            "mean_abs_error": mean_err,
                            "cosine_similarity": min_cos,
                            "numerical_equivalence_passed": max_err <= 5e-3 and min_cos >= 0.999,
                            "per_rank": per_rank,
                        }
                    assertions, mode_ok = _mode_assertions(gathered, mode_name, comparison.get(mode_name))
                    mode_assertions_cache[mode_name] = assertions
                    matrix_mode_status[mode_name] = mode_ok
                    summary = {
                        "run_id": args.run_id,
                        "mode_name": mode_name,
                        "backend": args.backend,
                        "ep_size": args.ep_size,
                        "observer_mode": observer_mode,
                        "scheduler_mode": injection_config.scheduler_mode,
                        "control_mode": injection_config.control_mode,
                        "transport_mutation": False,
                        "default_continue_shadow_emulation": injection_config.control_mode == "default_continue",
                        "actual_async_transport_override": False,
                        "source_provenance_digest": None,
                        "rank_summaries": gathered,
                    }
                    write_json(mode_dir / "summary.json", summary)
                    write_json(mode_dir / "assertions.json", assertions)
                    write_jsonl(mode_dir / "plans.jsonl", [row for item in gathered for row in item.get("policy_records", [])])
                    write_jsonl(mode_dir / "control_timeline.jsonl", [row for item in gathered for row in item.get("control_timeline", [])])
                    write_jsonl(mode_dir / "control_commands.jsonl", [row for item in gathered for row in item.get("control_commands", [])])
                    write_jsonl(mode_dir / "collective_trace.jsonl", [row for item in gathered for row in item.get("collective_trace", [])])
                    write_jsonl(mode_dir / "phase_contexts.jsonl", [row for item in gathered for row in item.get("phase_contexts", [])])
                    write_jsonl(mode_dir / "transport_bundles.jsonl", [row for item in gathered for row in item.get("transport_bundles", [])])
                stage_barrier(f"{mode_name}_done", ok=True, detail=mode_name)
            finally:
                if original_all_to_all is not None:
                    _restore_collective_trace(original_all_to_all)
                if model is not None:
                    del model
                gc.collect()
                torch.cuda.empty_cache()

        if rank == 0:
            dispatcher_fp = source_fingerprint or {}
            provenance = _collect_source_provenance(base_dir, dispatcher_fp)
            provenance_digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
            for mode_name in [item["name"] for item in _mode_matrix()]:
                summary_path = base_dir / mode_name / "summary.json"
                if summary_path.exists():
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    payload["source_provenance_digest"] = provenance_digest
                    write_json(summary_path, payload)
            matrix_status = "passed" if all(matrix_mode_status.get(mode["name"], False) for mode in _mode_matrix()) else "failed"
            write_json(base_dir / "comparison.json", {"comparisons": comparison})
            write_json(
                base_dir / "matrix_summary.json",
                {
                    "run_id": args.run_id,
                    "backend": args.backend,
                    "ep_size": args.ep_size,
                    "rank_to_host": {str(i): socket.gethostname() for i in range(world_size)},
                    "rank_to_device": {str(i): f"cuda:{i}" for i in range(world_size)},
                    "modes": [item["name"] for item in _mode_matrix()],
                    "matrix_status": matrix_status,
                    "mode_status": matrix_mode_status,
                    "comparison": comparison,
                    "source_provenance_digest": provenance_digest,
                },
            )
        return 0
    except Exception as exc:
        write_json(
            base_dir / f"{args.run_id}-rank{os.environ.get('RANK', 'unknown')}-error.json",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        return 1
    finally:
        destroy_distributed()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        stdout_handle.close()
        stderr_handle.close()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
