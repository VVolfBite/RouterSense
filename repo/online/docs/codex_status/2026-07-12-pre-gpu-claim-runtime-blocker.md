Starting SHA: `d76a5221ade3c714bda89739c4c196bd15178f3a`

Current SHA: `ef042342900dcefe8e3a5f3dc13baacd14893730`

CPU/Gloo closure completed on the current codebase:

- selected-layer resolution is explicit and strict
- official child config writes requested/resolved layer selectors and IDs
- runtime selected-layer counters are recorded
- runtime prediction generation is gated by formal policy capability instead of async-mode family checks
- target-plan formal contracts now exist:
  `CurrentWindowJointPlan`, `PreparedPriorityHint`, `TargetLayerPreparedJointPlan`, `ProvisionalExecutionPlan`
- standalone target-plan modules now exist:
  two-horizon predictor, target-plan store, target-plan reconcile, release-frontier state model
- new target-plan contract suite passed (`12 passed`)
- low-memory runtime-integrated Gloo gate still passed on the modified codebase

The remaining blocker is now a single formal runtime mechanism:

- the new target-plan / provisional / late-suffix / release-frontier chain is not yet integrated into the formal Megatron lifecycle and async transport hot path

More specifically, the codebase now has standalone CPU-contract implementations for:

- H1/H2 prediction
- target-layer prepared-plan storage
- exact/repairable/reject reconciliation
- provisional lineage objects
- release-batch frontier objects
- late suffix replacement over pending/planned tasks

But the formal runtime still executes current-window planning only:

- `before_token_dispatch()` still creates and consumes `PreparedWindowPlan`
- the real transport adapter / async executor still do not consume `TargetLayerPreparedJointPlan`
- the real async executor still submits the whole phase rather than a mutable release-batch frontier

Because of that, a 4GPU bring-up on the final runtime would still validate only current-window joint planning, not the claim-critical target-layer preplanning path.

Status: `PRE_GPU_SEMANTIC_BLOCKER`
