# Invariant Error Codes

Current structured guard families:

- `RS-STARTUP-*`: startup and artifact identity failures.
- `RS-CONFIG-*`: canonical config, policy, predictor, and runtime-line validation failures.
- `RS-STATE-*`: typed runtime state field violations.
- `RS-PLANNING-*`: scheduling request, digest, and prepared-plan lifecycle failures.
- `RS-COMPILER-*`: canonical task, compiler input, and legacy bridge violations.
- `RS-AGREEMENT-*`: distributed digest / agreement mismatches.
- `RS-TRANSPORT-*`: preflight, enqueue, tensor-role, split, fallback, and timeout failures.
- `RS-LIFECYCLE-*`: phase ordering and cleanup failures.
- `RS-OFFLINE-*`: offline replay audit failures.
- `RS-ARTIFACT-*`: manifest / SHA / output identity failures.
- `RS-REPORT-*`: report eligibility failures.

Representative codes in this milestone:

- `RS-STARTUP-001`: official entrypoint requires schema version 1.
- `RS-STARTUP-002`: strict or runtime-safe execution on a dirty tree.
- `RS-STARTUP-003`: missing runtime commit SHA.
- `RS-CONFIG-001`: runtime line mismatches official entrypoint.
- `RS-CONFIG-002`: non-positive bucket size.
- `RS-CONFIG-003`: non power-of-two bucket size.
- `RS-CONFIG-004`: non-canonical policy name in official config.
- `RS-CONFIG-005`: predictor not in canonical taxonomy or not eligible for the entrypoint.
- `RS-CONFIG-006`: reference-only or non-deployable policy requested online.
- `RS-STATE-001/002/003`: unknown runtime state field read / write / pop.
- `RS-COMPILER-MISSING-TASKS`: strict compile attempted with remote rows but no canonical tasks.
- `RS-OFFLINE-001`: offline replay produced invalid audit rows.
