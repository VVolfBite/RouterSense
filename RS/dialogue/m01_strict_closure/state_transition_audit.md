State transition audit summary:
- TargetPlanStore formal path now enforces LOGICAL_READY -> CLAIMED -> BOUND -> EXECUTING -> COMPLETED.
- Direct LOGICAL_READY -> EXECUTING and LOGICAL_READY -> COMPLETED are rejected.
- CLAIMED -> COMPLETED is rejected.
- Non-current publication tokens are rejected before store publication.
- Lifecycle still retains compatibility methods (claim_for_reconciliation / consume_once) for legacy paths outside the formal M1 gate.
