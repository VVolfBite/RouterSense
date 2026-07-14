M4 no longer relies on the old single-pass flow traversal.

Closed in this branch:
- evaluator now validates each wave against a pre-wave snapshot and only commits after the whole wave passes;
- same-wave predecessor/dependent pairs fail closed;
- compute delay and release time feed ready-time and makespan;
- task-set and truth digests are recomputed and forged digests are rejected;
- builder rejects world-size and identity mismatches before returning a request;
- oracle entrypoint shares the same cost model surface and provides a tiny exact fallback for small instances.

Still blocked:
- rollout and paired aggregation need full revalidation after the evaluator semantics change;
- simulator is still partial and cannot be claimed READY;
- offline/online parity depends on latest M2 materialization and integration wiring.
