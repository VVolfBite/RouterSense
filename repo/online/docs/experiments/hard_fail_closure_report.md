# Hard-Fail Closure Report

Scope of this milestone:

- add structured invariant modes and error contracts;
- hard-fail startup, config, artifact, state, compiler, transport, offline, and report-eligibility violations;
- make distributed failures rank-consistent;
- reject invalid offline / C2 / A2 results from entering formal reports;
- keep the current async runtime main path intact.

Observed validation results on the final no-GPU baseline:

- offline validation smoke: passed with `audit_invalid_count=0`;
- normal runtime-integrated low-memory Gloo gate: passed;
- fault-injection Gloo gate: 2 cases passed with rank-consistent exits;
- transport and unified-interface regression suites: passed;
- hard-fail unit tests for state / startup / report eligibility: passed.

Key runtime outcomes from the final Gloo gate:

- `actual_p0_total_rows > 0`;
- `actual_p0_matrix_unit = rows`;
- `p1_is_exact_transpose = true`;
- `canonical_task_count > 0`;
- `batch_isend_irecv_call_count > 0`;
- `phase_sync_fallback_count = 0`;
- `stored_p1_plan_digest == consumed_p1_plan_digest`.

Qualification outcome:

- invalid offline replay rows now fail the run instead of being marked completed;
- strict mode rejects canonical-task-missing compiler fallback to legacy bridge;
- unknown runtime state fields hard-fail outside diagnostic mode;
- formal report generation rejects dirty, fallback, timeout, invalid-audit, and legacy/shadow runs.
