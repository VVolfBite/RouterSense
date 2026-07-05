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

- formal `src/rs` no longer imports `integrations.*`
- formal `src/rs` no longer imports `rs.online`, `rs.runtime.distributed_ep`, `rs.evaluation`, `rs.trace`, or `rs.scheduler`
- online contract tests now import `rs.runtime.online.megatron_ep.*` and `rs.scheduling.*`
- formal `experiments/*` no longer import `integrations.*`, `rs.evaluation`, `rs.scheduler`, `rs.trace`, `rs.online`, `rs.offline`, or `rs.runtime.distributed_ep`
- formal baseline/reference shims now fail closed with `unsupported` metadata instead of advertising placeholder optimality
- default `pytest -q` excludes `legacy`, `gpu`, `nccl`, `multinode`, and `slow`

### Remaining structural debt

- oversized formal modules such as `src/rs/runtime/online/megatron_ep/_host_impl.py`, `_lifecycle.py`, and `src/rs/scheduling/multiphase/global_ready_set_impl.py` still need responsibility-based splitting
- historical research docs in `docs/` still reference old paths and should be migrated into archive-oriented documentation over time
- legacy trees remain preserved under `legacy/historical_poc/*` for historical audit only and are intentionally excluded from default validation

### Legacy parking already completed

- `integrations/*` moved to `legacy/historical_poc/integrations`
- `experiments/poc_line1/*` moved to `legacy/historical_poc/experiments_poc_line1`
- `experiments/distributed/*` moved to `legacy/historical_poc/experiments_distributed`
- historical offline experiment entrypoints moved to `legacy/historical_poc/experiments_offline`
- `experiments/legacy/*` moved to `legacy/historical_poc/experiments_legacy`
- old benchmark entrypoints moved to `legacy/historical_poc/experiments_online`
- `src/rs/evaluation`, `src/rs/scheduler`, `src/rs/trace`, and `src/rs/legacy` moved to `legacy/historical_poc/src_rs_legacy`
- analysis helpers moved into `scripts/metrics` and `scripts/plot`
- ablation config YAML moved into `configs/experiment/ablation`
- archive scripts moved into `scripts/maintenance/archive`
