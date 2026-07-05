## Dependency Graph

### Target dependency direction

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

- `core` must not import `scheduling`, `runtime`, `experiments`, `integrations`, or `legacy`
- `scheduling` must not import `runtime`, `experiments`, `integrations`, or `legacy`
- `runtime` must not import `experiments`
- `experiments` may import `runtime`, `scheduling`, and `core`
- `legacy` must not be imported by formal code

### Current formal packages

- `src/rs/core`
- `src/rs/scheduling`
- `src/rs/runtime/offline`
- `src/rs/runtime/online/megatron_ep`

### Current known violations

1. `src/rs/scheduling/policy/agreement.py`
   - imports `torch` and `torch.distributed`
   - functionally belongs under `runtime/online/megatron_ep/control`

2. `experiments/offline/*`
   - several wrappers still call `experiments/poc_line1/*`
   - this is not a stable formal end state

3. `experiments/online/bench_native_ep.py` and `collect_native_ep_trace.py`
   - still import `rs.online.olmoe_ep`

4. `experiments/distributed/*`
   - still import `rs.runtime.distributed_ep`

5. old namespaces still remain in the repository:
   - `src/rs/scheduler`
   - `src/rs/evaluation`
   - `src/rs/trace`
   - `src/rs/online/olmoe_ep`
   - `src/rs/runtime/distributed_ep`

### Current non-violations already enforced

- `src/rs/runtime/*` does not import `experiments`
- `src/rs/scheduling` no longer imports `rs.runtime.online.megatron_ep.phase` or `execution` contracts directly
- formal `host` import works with `PYTHONPATH=src`

### Required next moves

- move `policy/agreement.py` to `runtime/online/megatron_ep/control`
- replace remaining `experiments/offline -> experiments/poc_line1` delegation with direct formal runtime/scheduling calls
- migrate or legacy-park `rs.online.olmoe_ep`
- migrate or legacy-park `rs.runtime.distributed_ep`
