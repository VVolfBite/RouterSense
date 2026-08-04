Integration risks identified before implementation:

1. `public_types.py` already mixes concrete runtime handle behavior with future interface contracts.
2. `TargetPlanStore` still exposes compatibility methods (`claim_for_reconciliation`, `consume_once`) that can bypass the final formal state machine boundary.
3. `lifecycle.py` still contains late-suffix compatibility paths and direct Gloo agreement helpers that will compete with the future `ControlCommunicationLane`.
4. `async_release/*`, `control/plan_agreement.py`, `control/agreement_wire.py`, and `target_planning` agreement helpers overlap in ownership and will conflict with a single M2 publisher unless explicitly isolated or deleted.
5. Current observation and reporting code still writes through runtime state and file helpers; M3 integration must remain passive or it will reintroduce state-owner ambiguity.
