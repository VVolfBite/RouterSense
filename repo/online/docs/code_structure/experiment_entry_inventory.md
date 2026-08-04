# Experiment Entry Inventory

## Official

- `experiments/run_offline_replay.py`
  Purpose: unified offline replay entrypoint.
  Status: direct CLI, official.
- `experiments/run_online_phase_sync.py`
  Purpose: public phase-sync online entrypoint.
  Status: direct CLI, official.
- `experiments/run_online_async_release.py`
  Purpose: public async-release online entrypoint.
  Status: direct CLI, official.
- `experiments/dev/run_validation.py`
  Purpose: single public validation suite entrypoint.
  Status: direct CLI, validation.
- `experiments/reporting/build_report.py`
  Purpose: single public report-generation entrypoint.
  Status: direct CLI, official reporting.

## Compatibility Wrappers

- `experiments/dev/run_gpu_validation.py`
  Replacement: `experiments/dev/run_validation.py`
  Status: compatibility forwarder.

## Internal Validation / Diagnostic Workflows

- `experiments/distributed/run_gpu_b2_lifecycle.py`
- `experiments/distributed/run_gpu_c2_async_correctness.py`
- `experiments/distributed/run_gpu_a2_strategy_compare.py`
- `experiments/distributed/run_stage1_gloo_e2e_gate.py`
- `experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py`
- `experiments/distributed/run_stage3_runtime_integrated_gloo_gate_lowmem.py`
- `experiments/online/run_strategy_comparison.py`
- `experiments/online/run_policy_correctness.py`
- `experiments/online/collect_native_ep_trace.py`

These remain executable, but they are internal workflow implementations rather than public entrypoints.

## Historical / Analysis Scripts

The large collections under `experiments/offline/` and `experiments/online/` are treated as historical analysis or research scripts unless promoted through the official wrappers above.

