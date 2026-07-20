from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8")


def main() -> None:
    host = ROOT / "src/rs/runtime/online/megatron_ep/host.py"
    lifecycle = ROOT / "src/rs/runtime/online/megatron_ep/lifecycle.py"
    transport_adapter = ROOT / "src/rs/runtime/online/megatron_ep/execution/transport_adapter.py"
    sync_exec = ROOT / "src/rs/runtime/online/megatron_ep/execution/sync_wave_executor.py"
    async_exec = ROOT / "src/rs/runtime/online/megatron_ep/execution/async_p2p_executor.py"
    p2_provider = ROOT / "src/rs/runtime/online/megatron_ep/control/p2_provider.py"
    async_projection = ROOT / "src/rs/runtime/online/megatron_ep/async_release/runtime_projection.py"
    async_joint = ROOT / "src/rs/runtime/online/megatron_ep/async_release/joint_plan_agreement.py"
    async_p2p = ROOT / "src/rs/runtime/online/megatron_ep/async_release/p2p_executor.py"
    correctness_runner = ROOT / "experiments/online/run_policy_correctness.py"
    observe_runner = ROOT / "experiments/online/collect_native_ep_trace.py"

    audit = {
        "megatron_hook_entry": {
            "file": str(host.relative_to(ROOT)),
            "attached": _contains(host, "wrapped_dispatch") and _contains(host, "wrapped_combine"),
        },
        "host_attach_point": {
            "file": str(host.relative_to(ROOT)),
            "attach_formal_online_runtime": _contains(host, "def attach_formal_online_runtime"),
        },
        "phase_sync_transport_adapter": {
            "file": str(transport_adapter.relative_to(ROOT)),
            "reachable": _contains(transport_adapter, "execute_scheduled_phase_tensor"),
        },
        "joint_async_execution_mode_reachable": {
            "file": str(host.relative_to(ROOT)),
            "reachable": _contains(host, "joint_window_async_p2p"),
        },
        "prediction_call": {
            "file": str(lifecycle.relative_to(ROOT)),
            "record_prediction_for_dispatch": _contains(lifecycle, "_record_prediction_for_dispatch"),
            "active_prediction_state": _contains(lifecycle, "active_next_dispatch_prediction"),
        },
        "host_projection_call": {
            "file": str(async_projection.relative_to(ROOT)),
            "implemented": async_projection.exists(),
            "referenced_from_runtime": _contains(lifecycle, "host_project_safe_selection")
            or _contains(lifecycle, "RuntimeHostFeasibilityProjector"),
        },
        "global_plan_agreement_call": {
            "file": str(async_joint.relative_to(ROOT)),
            "implemented": async_joint.exists(),
            "referenced_from_runtime": _contains(lifecycle, "GlobalJointPlanWire") or _contains(host, "GlobalJointPlanWire"),
        },
        "p2p_executor_call": {
            "file": str(async_exec.relative_to(ROOT)),
            "implemented": async_exec.exists(),
            "referenced_from_transport_adapter": _contains(transport_adapter, "execute_async_phase_tensor"),
        },
        "forward_begin_end_call": {
            "file": str(lifecycle.relative_to(ROOT)),
            "implemented": _contains(lifecycle, "def begin_forward") and _contains(lifecycle, "def end_forward"),
            "referenced_from_host": (
                _contains(host, "begin_forward(")
                or _contains(host, "end_forward(")
                or _contains(correctness_runner, ".begin_forward(")
                or _contains(correctness_runner, ".end_forward(")
                or _contains(observe_runner, ".begin_forward(")
                or _contains(observe_runner, ".end_forward(")
            ),
        },
        "current_real_executor": {
            "file": str(async_exec.relative_to(ROOT)),
            "phase_sync_only": False,
            "async_p2p_available": async_exec.exists(),
        },
    }

    out_path = ROOT / "outputs/runtime_callgraph_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
