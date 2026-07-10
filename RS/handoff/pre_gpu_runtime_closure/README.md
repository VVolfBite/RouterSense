# Pre-GPU Runtime Closure Handoff

This directory is the tracked handoff entrypoint for the current pre-GPU closure.

Start here:

- [Stage1 Single-Node Final Status](../../docs/stage1_single_node_final_status.md)
- [Final Pre-GPU Closure Status](../../docs/codex_status/2026-07-11-final-pre-gpu-closure.md)
- [Runtime Joint Async Design](../../docs/runtime_joint_async_design.md)
- [Prediction Final Design](../../docs/prediction_final_design.md)
- [Control Overhead Optimization](../../docs/control_overhead_optimization.md)
- [GPU Run B2/C2/A2 Commands](../../docs/gpu_run_b2_c2_a2_commands.md)

Key runtime validation artifacts:

- [Runtime Callgraph Audit](../../outputs/runtime_callgraph_audit.json)
- [Gloo Executor Gate Summary](../../outputs/distributed/run_stage1_gloo_e2e_gate/summary.json)
- [Runtime-Integrated Gloo Gate Summary](../../outputs/distributed/run_stage1_runtime_integrated_gloo_gate/summary.json)

Current environment note:

- only one CUDA device is visible in this session
- 4GPU runs are prepared but not executed here
