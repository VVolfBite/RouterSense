Starting SHA: `6e555201efa900e4db0217ef6e989846e0e6d61f`

Current SHA: `c9b8a2600498c917ccdc5aa494991bcd76add066`

CPU/Gloo closure completed on the current codebase:

- selected-layer resolution is explicit and strict
- official child config writes requested/resolved layer selectors and IDs
- runtime selected-layer counters are recorded
- online prediction generation is gated by policy capability instead of async-mode alone
- unsupported online scheduler modes raise `UnsupportedSchedulerMode`
- `bucketed_fifo` resolves to the canonical FIFO phase policy
- `rs.scheduling` no longer imports `rs.runtime`
- targeted regression suite passed (`176 passed`)
- low-memory runtime-integrated Gloo gate passed

The remaining blocker is semantic, not environmental:

- the runtime still only executes current-window joint planning for `P0(L)` and `P1(L)`
- there is no end-to-end `TargetLayerPreparedJointPlan`
- there is no real `H1/H2` two-horizon target-plan generation and storage chain
- there is no live provisional/default-continue path that later upgrades into a target-plan lineage
- there is no release-batch frontier or late suffix splice in the real async executor

Because of that, a 4GPU bring-up on this commit would still validate only current-window joint planning, not the claim-critical target-layer preplanning path.

Status: `PRE_GPU_SEMANTIC_BLOCKER`
