# Stage1 GPU Experiment Report

Source of truth:

- `outputs/distributed/stage1_gpu_blocker_summary.json`
- `outputs/distributed/stage1_gpu_blocker_summary.md`
- `outputs/distributed/run_b2_20260711_144200`
- `outputs/distributed/run_b2_20260711_145000`

## Current status

- `Run B2`: not passed
- `Run C2`: not executed because B2 has not passed
- `Run A2 initial/final`: not executed because B2 has not passed

## What has been demonstrated on 4GPU

- The async transport path is reachable from the real Megatron hook.
- `P0` executes real nonzero wave payloads on 4 GPU.
- `input_splits` observed at layer 2 are nonzero and consistent with real distributed dispatch.
- The failure is no longer “async executor never runs”.

## Current blocker

- The runtime still enters `before_token_combine` with a zero stored `actual_p0_full_row_matrix` and zero `inferred_p1_row_matrix`.
- That causes a false-positive local `P1` invariant mismatch and stops B2 before `P1` async transport can be validated.

## Conclusion

- The transport executor path has crossed the earlier reachability barrier.
- The remaining blocker is lifecycle/materialization state propagation into the stored `P1` abstract plan and its invariant gate.
- No performance conclusion should be drawn yet from the 4GPU runtime path.
