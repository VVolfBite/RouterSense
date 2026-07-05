## Source Of Truth Inventory

This inventory reflects the current pre-evaluation mainline after the runtime
and experiment cutover pass. It records only the formal source-of-truth
surfaces that remain active in the mainline tree.

### Formal online runtime

Canonical online Megatron EP implementation lives in:

- `src/rs/runtime/online/megatron_ep/_host_impl.py`
- `src/rs/runtime/online/megatron_ep/_lifecycle.py`
- `src/rs/runtime/online/megatron_ep/_observation.py`
- `src/rs/runtime/online/megatron_ep/contracts.py`
- `src/rs/runtime/online/megatron_ep/observer.py`
- `src/rs/runtime/online/megatron_ep/trace_writer.py`
- `src/rs/runtime/online/megatron_ep/p2_provider.py`
- `src/rs/runtime/online/megatron_ep/phase/*`
- `src/rs/runtime/online/megatron_ep/control/*`
- `src/rs/runtime/online/megatron_ep/execution/*`
- `src/rs/runtime/online/megatron_ep/policy_adapter.py`

Stable public entrypoints:

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/runtime.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/artifact_recorder.py`

Status:

- canonical implementation: `src/rs/runtime/online/megatron_ep`
- formal tests: now import `rs.runtime.online.megatron_ep.*`

### Formal offline runtime

Canonical offline trace / traffic / prediction implementation lives in:

- `src/rs/runtime/offline/trace/*`
- `src/rs/runtime/offline/traffic/*`
- `src/rs/runtime/offline/prediction/*`
- `src/rs/runtime/offline/runner.py`

Status:

- trace collection and prediction helpers are canonical under `runtime/offline`
- `experiments/offline/collect_router_trace.py` and `analyze_next_layer_traffic_predictability.py` use canonical runtime/offline imports
- formal `experiments/offline/` now only contains entrypoints that import canonical `rs.runtime.offline` helpers

### Formal scheduling

Canonical scheduling implementation lives in:

- `src/rs/scheduling/contracts.py`
- `src/rs/scheduling/matching.py`
- `src/rs/scheduling/validation.py`
- `src/rs/scheduling/phase_local/*`
- `src/rs/scheduling/reference/*`
- `src/rs/scheduling/multiphase/*`
- `src/rs/scheduling/registry.py`
- `src/rs/scheduling/capabilities.py`
- `src/rs/scheduling/diagnostics.py`

Status:

- policy library used by the online runtime is canonical under `src/rs/scheduling/phase_local` plus the root registry/capabilities surface
- pure wire/agreement logic moved out of scheduling into `runtime/online/megatron_ep/control/agreement_wire.py`
- formal `scheduling/baselines/birkhoff.py` and `scheduling/reference/{oracle_guided,exact_small_instance}.py` now fail closed with explicit `unsupported` metadata instead of returning placeholder optimality claims

Detailed migration mapping lives under `docs/migration/`. The formal mainline
inventory intentionally records only the active canonical paths.
