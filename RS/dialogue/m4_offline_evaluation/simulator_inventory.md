**Formal simulator namespace**

- `rs.simulation`

**Current state**

- namespace created in Phase A
- fail-closed placeholder only
- no second formal runtime simulator was added

**Legacy simulation paths**

- `runtime/online/megatron_ep/async_release/simulator.py`: `SIMULATION_ONLY`
- `scheduling/multiphase/streaming_simulator.py`: `SIMULATION_ONLY`

**Reason**

Materialized-plan simulation and online functional parity require M2 execution contracts to be READY. Phase A keeps the formal namespace and blocks unresolved execution semantics instead of inventing a second simulation model.
