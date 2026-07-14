Ownership map:

- M1 exclusive:
  - `runtime/online/megatron_ep/lifecycle.py`
  - `runtime/online/megatron_ep/host.py`
  - `runtime/online/megatron_ep/runtime.py`
  - `runtime/online/megatron_ep/public_types.py`
  - `runtime/online/megatron_ep/config.py`
  - `runtime/online/megatron_ep/state/**`
  - `runtime/online/megatron_ep/target_planning/**`
  - `runtime/online/megatron_ep/control/communication_lane.py`

- M2 exclusive:
  - `core/contracts/execution.py`
  - `runtime/online/megatron_ep/control/rank_map.py`
  - `runtime/online/megatron_ep/control/plan_publisher.py`
  - `runtime/online/megatron_ep/control/plan_validator.py`
  - `runtime/online/megatron_ep/control/execution_guard.py`
  - `runtime/online/megatron_ep/materialization/**`
  - `runtime/online/megatron_ep/execution/**`
  - `runtime/online/megatron_ep/async_release/**`
  - `runtime/online/megatron_ep/phase/**`
  - `runtime/online/megatron_ep/compiler_facade.py`

- M3 exclusive:
  - `core/contracts/checks.py`
  - `core/contracts/measurement.py`
  - `core/contracts/debug.py`
  - `core/contracts/artifact.py`
  - `core/contracts/result.py`
  - `core/contracts/trace.py`
  - `runtime/**/observation/**`
  - `runtime/**/guards/**`
  - evidence/reporting adapters

- Integration only:
  - lifecycle <-> M2 materializer/executor wiring
  - lifecycle <-> M3 sinks/probes/writers wiring
  - final host/runtime constructor injection
