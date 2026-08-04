M4 no longer relies on the old single-pass flow traversal.

Closed in this branch:
- evaluator now validates each wave against a pre-wave snapshot and only commits after the whole wave passes;
- same-wave predecessor/dependent pairs fail closed;
- compute delay and release time feed ready-time and makespan;
- task-set and truth digests are recomputed and forged digests are rejected;
- builder rejects world-size and identity mismatches before returning a request;
- oracle entrypoint shares the same cost model surface and the tiny exact fallback no longer claims certified optimum;
- replay now fails closed instead of falling back to legacy/planner-reported makespan;
- planning-visible replay metadata no longer exposes execution truth digest;
- rollout history is isolated by group and prior sequence only;
- paired aggregation filters invalid/ineligible records and separates predictor comparisons;
- M123 integration is now merged into this branch and Phase B parity is wired through the real M2/M123 contracts;
- offline builder vs online-sync request parity now passes on the 4-rank replay fixture;
- offline planner vs online-sync planner `WindowPlan` parity now passes on the same fixture;
- direct offline materialization vs `RuntimeExecutionPipeline.prepare()` materialization digests now match per rank;
- offline expected completed task IDs now match Gloo functional completed task IDs on a 4-rank distributed gate.

Still blocked:
- simulator is still partial and cannot be claimed READY;
- the legacy offline replay smoke entrypoint still fails through the experiment-config runner path, which remains outside this branch's formal M4 core parity surface.
