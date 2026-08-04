Strict closure checkpoint.

What was completed in this pass:
- M0 formal contract closure remains green on CPU tests.
- Planning formal API no longer exports legacy builders; legacy runtime bridge is private.
- Planning topology and cost model now enforce fixed 1-send / 1-receive port contract.
- Training sample, history predictor alpha, and linear predictor ridge validation were added.
- Lifecycle no longer performs main-thread predictor execution in the dispatch recording path.
- Lifecycle now uses TargetLayerPlannerService.submit() rather than enqueue().
- TargetLayerPlannerService now uses keyed coalescing semantics with per-task version/session tokens.
- Ready publication is rejected when token/session/version is stale.
- Worker-produced H1/H2 predictions are written back into runtime state during publication pump.
- TargetPlanStore now enforces the main LOGICAL_READY -> CLAIMED -> BOUND -> EXECUTING -> COMPLETED path.
- RuntimeHandle close/detach now execute all callbacks before surfacing aggregate errors.
- Duplicate formal attach and legacy observer conflicts are rejected.
- Dispatch / combine / forward failure events were added to the formal wrapper path.

What is still blocking M1 READY:
- No dedicated ControlCommunicationLane with deterministic poll(sequence) protocol yet.
- No required 4-rank formal Gloo proof for delayed-rank / failed-rank / cancelled-generation publication ordering.
- Formal no-late-suffix gate has not been proven by the required multi-rank evidence.

Test checkpoint:
- compileall: passed
- targeted M0/M1 CPU suite: 97 passed
- offline smoke: experiments/run_offline_replay.py minimal_offline.yaml returned exit code 0
