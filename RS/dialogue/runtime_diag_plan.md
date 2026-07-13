# Runtime Diagnostic Plan

The next GPU run should execute `configs/official/gpu_runtime_diag.yaml` once with 4 GPUs, selected layers `0,1`, workload `8x16`, `timeline_light`, and `preflight_mode=compact`.

Required next-run checks:

- Preflight: requested, effective, and executor modes are all `compact`.
- Preflight: actual collective count equals compact expected count.
- Expert gap: report `p0_all_requests_completed_ns`, expert boundary fields when available, `p1_hook_enter_ns`, and residual gap.
- Hook attribution: report selected hook total, non-overlapping component totals, derived unattributed time, and explained ratio.
- Control collectives: separate traffic matrix, plan agreement, broadcast, preflight, barrier, and other control communication.
- Rank 3: identify whether its excess is planning, preflight, submit, wait, hook, or expert/inter-phase gap.
- Task granularity: report task payload buckets, `batch_isend_irecv` calls, tasks per call, bytes per call, and payload type counts.

No algorithm or executor policy changes should be mixed into that GPU measurement.
