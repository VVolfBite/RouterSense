Observed cost/unit surfaces:

- planning estimator uses row-based cost
- offline evaluator now uses:
  - `matrix_unit = rows`
  - `bytes_per_row`
  - `bandwidth`
  - `launch_cost`
- simulation namespace is intentionally blocked until M2 materialization/execution contracts are authoritative

Phase A explicit choice:

- do not mix rows and bytes implicitly
- `EvaluationTask` stores both `row_count` and `byte_count`
- `EvaluationSpec` carries `bytes_per_row` and `bandwidth`
