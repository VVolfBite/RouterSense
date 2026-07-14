**Formal oracle surface**

- `rs.offline.oracle.cp_sat.solve_cp_sat`
- `rs.offline.oracle.cp_sat.OracleResult`

**Current runtime environment**

- OR-Tools is not installed in the execution environment
- formal CP-SAT wrapper therefore returns `UNSUPPORTED` fail-closed

**Legacy oracle sources**

- `rs.scheduling.reference.exact_small_instance`: `REFERENCE_ONLY`
- `rs.scheduling.reference.oracle_guided`: `REFERENCE_ONLY`
- `experiments/offline/run_prediction_oracle_baseline_closure.py`: `REFERENCE_ONLY`

**Phase B / later work**

- migrate experiment-local CP-SAT model into formal package when OR-Tools-backed validation is available
- add exact-local vs exact-joint certification and gap reporting on top of `EvaluationTaskSet`
