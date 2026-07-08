## Dependency Graph

### Intended formal direction

```text
experiments
    ↓
runtime / evaluation
    ↓
scheduling
    ↓
core
```

Equivalent restrictions:

- `core` must not import `scheduling`, `runtime`, `experiments`, or `legacy`
- `scheduling` must not import `runtime`, `experiments`, `torch`, `Megatron`, or `NCCL`
- `runtime` must not import `experiments`
- `experiments` may import `runtime`, `scheduling`, and `core`
- `legacy` must not be imported by formal code

### Current formal packages

- `src/rs/core`
- `src/rs/scheduling`
- `src/rs/runtime/offline`
- `src/rs/runtime/online/megatron_ep`

### Current enforced non-violations

- formal `src/rs` imports only the current canonical namespaces
- online contract tests now import `rs.runtime.online.megatron_ep.*` and `rs.scheduling.*`
- formal `experiments/*` import the canonical runtime and scheduling surfaces only
- formal baseline/reference shims now fail closed with `unsupported` metadata instead of advertising placeholder optimality
- default `pytest -q` excludes `legacy`, `gpu`, `nccl`, `multinode`, and `slow`

### Remaining structural debt

- legacy trees remain preserved under `legacy/historical_poc/*` for historical audit only and are intentionally excluded from default validation
- online multiphase joint execution is still not implemented:
  - prepared-window plans currently feed calibrated P2 hint and shadow analysis
  - actual online execution remains phase-local under the frozen executor contract
- artifact-side diagnostics now include prepared-window shadow alignment analysis under `experiments/online/support/*` and `scripts/diagnostics/*`
