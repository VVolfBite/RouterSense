# RS Refactor Status - 2026-07-02

## Scope
Completed the no-GPU refactor pass requested by `rs_channel/ins`, focused on structure cleanup and import compatibility without changing scheduling logic.

## Completed
- Split POC-line1 logic into:
  - `RS/src/routesense/scheduler/oracle.py`
  - `RS/src/routesense/scheduler/greedy.py`
  - `RS/src/routesense/evaluation/cross_layer.py`
  - `RS/src/routesense/evaluation/traffic_matrix.py`
  - `RS/src/routesense/evaluation/analysis.py`
- Kept `RS/src/routesense/evaluation/poc_line1.py` as a compatibility re-export layer.
- Added `RS/src/routesense/scheduler/__init__.py` and restored scheduler exports through `routesense.evaluation`.
- Removed old empty dirs:
  - `RS/src/routesense/oracle/`
  - `RS/experiments/baseline/`
  - `RS/experiments/paper/`
  - `RS/experiments/stress/`
  - `RS/src/routesense/runtime/local_test/`
- Moved analysis placeholders to the correct top-level path `RS/analysis/`.
- Added direct-execution wrappers:
  - `RS/experiments/poc_line1/exp_trace.py`
  - `RS/experiments/poc_line1/exp_cross_layer.py`
  - `RS/experiments/poc_line1/exp_oracle.py`
  - `RS/experiments/poc_line1/exp_pairwise.py`
  - `RS/experiments/distributed/exp_nccl_smoke.py`
  - `RS/experiments/distributed/exp_olmoe_ep.py`
- Restored a minimal real runtime compatibility layer in:
  - `RS/src/routesense/runtime/single_gpu.py`
  - `RS/src/routesense/runtime/__init__.py`
  so existing scripts importing `load_model_and_tokenizer`, `run_single_gpu_text_inference`, and `gpu_environment_snapshot` no longer point to deleted code.

## Validation
- `cd RS && pytest -q tests`
  - result: `31 passed`
- `cd RS && python experiments/poc_line1/exp_pairwise.py --help`
  - direct script execution works
- `cd RS && python experiments/distributed/exp_nccl_smoke.py --help`
  - direct script execution works

## Notes
- This pass was structural only. No scheduler algorithm semantics were changed.
- Active experiment outputs were left uncommitted and not included in this refactor step.
- Relevant code lives in `RS/src/routesense/`.
