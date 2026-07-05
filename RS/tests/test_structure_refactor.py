from __future__ import annotations

from pathlib import Path


def test_refactor_structure_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src/rs/core/contracts/flow.py",
        root / "src/rs/core/contracts/topology.py",
        root / "src/rs/core/contracts/trace.py",
        root / "src/rs/core/contracts/result.py",
        root / "src/rs/core/contracts/provenance.py",
        root / "src/rs/scheduling/contracts.py",
        root / "src/rs/scheduling/phase_local/fifo.py",
        root / "src/rs/scheduling/phase_local/aurora_fixed.py",
        root / "src/rs/scheduling/phase_local/fast_bvn_fixed.py",
        root / "src/rs/scheduling/multiphase/global_ready_set.py",
        root / "src/rs/scheduling/multiphase/routersense_p0p1.py",
        root / "src/rs/scheduling/multiphase/routersense_p0p1p2.py",
        root / "src/rs/runtime/offline/trace/olmoe.py",
        root / "src/rs/runtime/offline/trace/qwen.py",
        root / "src/rs/runtime/offline/traffic/matrix_builder.py",
        root / "src/rs/runtime/offline/prediction/cross_layer.py",
        root / "src/rs/runtime/online/megatron_ep/host.py",
        root / "src/rs/runtime/online/megatron_ep/runtime.py",
        root / "src/rs/runtime/online/megatron_ep/lifecycle.py",
        root / "src/rs/runtime/online/megatron_ep/artifact_recorder.py",
        root / "src/rs/runtime/online/megatron_ep/phase/contracts.py",
        root / "src/rs/runtime/online/megatron_ep/phase/layout_join.py",
        root / "src/rs/runtime/online/megatron_ep/control/plan_agreement.py",
        root / "src/rs/runtime/online/megatron_ep/execution/sync_wave_executor.py",
        root / "scripts",
        root / "experiments/offline/collect_router_trace.py",
        root / "experiments/offline/analyze_cross_layer_prediction.py",
        root / "experiments/offline/run_multiphase_reference.py",
        root / "experiments/offline/run_scheduler_ablation.py",
        root / "experiments/offline/compare_prediction_inputs.py",
        root / "experiments/online/collect_native_ep_trace.py",
        root / "experiments/online/run_policy_correctness.py",
        root / "experiments/online/run_policy_benchmark.py",
        root / "experiments/online/run_injection_smoke.py",
        root / "experiments/online/run_host_api_probe.py",
        root / "legacy/hf_olmoe_ep_harness/README.md",
        root / "legacy/historical_poc/README.md",
        root / "archive/README.md",
    ]
    for path in required:
        assert path.exists(), path


def test_legacy_paths_remain_outside_formal_mainline() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy_runtime_paths = [
        root / "src/rs/online/olmoe_ep",
        root / "src/rs/runtime/distributed_ep",
        root / "experiments/distributed",
        root / "experiments/online/bench_native_ep.py",
        root / "experiments/online/bench_scheduled_ep.py",
    ]
    for path in legacy_runtime_paths:
        assert path.exists(), path
