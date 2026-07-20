# Async-release runtime contract

`runtime.line=async_release` is the deployable online path. It materializes a
canonical logical plan into rank-local P0/P1 tasks and executes them with the
Megatron transport adapter and ordered point-to-point operations.

## Supported semantics

- actual P0 is executable immediately after the plan is accepted;
- P1 remains blocked until the matching P0 inbound completion and compute gate;
- predicted P2/P3 is advisory only and is never compiled as current-layer bytes;
- Future-P012 planning may run in the previous layer and bind actual P0/P1 at
  the target layer using exact/repairable/reject reconciliation;
- Safe selection compares Joint and same-engine Local plans under one cost
  profile before transport begins.

## Required preflight

Before the first P2P operation the runtime validates plan identity, canonical
coverage, peer offsets, send/receive rows and bytes, dtype/shape suffixes,
process-group identity and receive-buffer coverage. A post-execution audit
rechecks task IDs, rows, bytes and completion events.

## Non-claims before server validation

CPU/Gloo contracts do not prove physical NCCL performance. CUDA/NCCL correctness,
real P0/P1 tensor replacement and end-to-end latency become eligible only after
the deployment pipeline returns a passing collected result summary.
