Qwen trace scaffold is in place.

Implemented:

- `RS/src/routesense/trace/qwen_moe_router_trace.py`
  - `collect_qwen_moe_architecture_probe`
  - `collect_qwen_moe_router_trace`
  - `collect_qwen_moe_full_sequence_trace`
- `RS/experiments/poc_line1/full_sequence_trace_qwen.py`
- `RS/tests/test_qwen_trace.py`

Validation:

- `cd RS && pytest -q tests/test_qwen_trace.py tests/test_poc_line1.py`
- result: `23 passed`

Smoke status:

- Real local Qwen smoke was attempted with
  `experiments/poc_line1/full_sequence_trace_qwen.py`
- current single-GPU loader OOMs on local
  `Qwen1.5-MoE-A2.7B` under the existing whole-model-on-one-GPU load path
- the failure is in model loading, not in the new trace schema logic

Implication:

- The trace/eval framework is ready for Qwen.
- To run a real Qwen trace on this host, next step is either:
  1. multi-GPU placement, or
  2. a controlled CPU-offload / device-map load path

No scheduler or analysis logic was changed in this step.
