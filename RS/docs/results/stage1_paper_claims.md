# Stage1 Paper Claims

## Supported

- Joint scheduling opportunity exists offline.
- The strongest safe-U families still show positive offline gain relative to their paired B baselines.
- `P0` async transport is reachable from the real Megatron runtime on 4GPU.
- The offline predictor-selection procedure is reproducible and currently selects `zero_hint` under held-out schedule regret.

## Partially supported

- Host-projected joint runtime planning is wired into the online path.
- Real 4GPU async execution has progressed through nonzero `P0` payload execution.

## Not yet supported

- End-to-end `B2` lifecycle success on real 4GPU.
- `C2` correctness parity for async vs sync.
- `A2` performance claims against `Birkhoff async`, `Birkhoff sync`, or `native`.

## Claims that should be narrowed now

- “Prediction already improves runtime scheduling” should be narrowed.
  - Current offline held-out result selects `zero_hint`.
  - Oracle traffic does not produce a positive held-out scheduling gain under the present scheduler-core summary.
- “Async runtime is performance-ready” should be narrowed.
  - Real 4GPU evidence only supports async `P0` reachability so far, not full `P1` correctness or performance.

## Next claim boundary once the blocker is fixed

- If `B2` passes and `C2` shows parity, the next valid claim would be:
  - “real async P0/P1 execution is correct on the tested 4GPU single-node setup”
- Only after `A2` should any performance claim be made.
