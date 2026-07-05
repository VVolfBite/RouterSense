## Pre-Evaluation Handoff

This repository is not yet at the fully cut-over post-migration state.
It is at a pre-evaluation consolidation state with a formal mainline already established under `src/rs/`.

### What is already true

- `src/rs/runtime/online/megatron_ep` is the canonical home for the current Megatron online runtime implementation
- `src/rs/runtime/offline` and `src/rs/scheduling` exist and are importable
- `PYTHONPATH=src pytest -q` passes on the current CPU suite
- source archive packaging self-check passes for the mainline scope
- online experiment wrappers under `experiments/online` are callable with `PYTHONPATH=src:.`

### What is not yet fully closed

- old namespaces still exist:
  - `src/rs/online/olmoe_ep`
  - `src/rs/runtime/distributed_ep`
  - `src/rs/scheduler`
  - `src/rs/evaluation`
  - `src/rs/trace`
- `experiments/distributed` and `experiments/poc_line1` still remain
- several formal wrappers still point through historical experiment modules
- `src/rs/scheduling/policy/agreement.py` still contains runtime-distributed logic

### Immediate next cleanup targets

1. move distributed agreement helpers out of `src/rs/scheduling/policy/agreement.py`
2. replace `experiments/offline/* -> experiments/poc_line1/*` delegation
3. migrate or legacy-park `rs.online.olmoe_ep`
4. migrate or legacy-park `rs.runtime.distributed_ep`
5. add AST dependency tests to keep formal layers from regressing

### Current safe editing zones

- `src/rs/scheduling/*`
- `src/rs/runtime/offline/*`
- `src/rs/runtime/online/megatron_ep/*`
- `experiments/offline/*`
- `experiments/online/*`
- `configs/*`
- `docs/architecture/*`

### Frozen zones unless correctness bug is proven

- P0/P1 hook timing
- TransferLayout semantics
- P0 hidden/probs atomicity
- P1 bundle contract
- transport adapter semantics
- sync wave executor semantics
