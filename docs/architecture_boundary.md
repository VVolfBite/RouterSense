# Architecture boundary

This repository is the only deployable RouterSense source of truth. Historical
POC, recovery, Round 1/2 handoff and retired B/U/Tier1 material are not present
in the deployment mainline.

Rules:

- `src/rs` must never import `experiments` or deployment scripts;
- online runtime must never import offline reference/oracle execution as a
  fallback;
- related-work style baselines remain under `src/rs/reference/baselines` and do
  not enter the online planner registry;
- deployment scripts may configure and invoke the runtime but may not rewrite
  source code on remote nodes.
