**Formal simulator namespace**

- `rs.simulation`

**Current state**

- namespace created in Phase A
- `WindowPlan` simulation now reuses the same formal task-set semantics as `OfflineEvaluator`
- `MaterializedPlan` simulation remains fail-closed until M2 execution contracts are authoritative
- no second formal runtime simulator was added

**Legacy simulation paths**

- `runtime/online/megatron_ep/async_release/simulator.py`: `SIMULATION_ONLY`
- `scheduling/multiphase/streaming_simulator.py`: `SIMULATION_ONLY`

**Reason**

Materialized-plan simulation and online functional parity require M2 execution contracts to be READY. Phase A now closes the logical `WindowPlan` side but still blocks unresolved execution/materialization semantics instead of inventing a second runtime model.
