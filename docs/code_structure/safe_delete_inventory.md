# Formal Delete Inventory

This document records the post-convergence state. It is not a compatibility promise.

## Removed from active `src/rs`

- historical B/U/Tier1 strategy families;
- recovered lookahead strategy executors;
- paired-U adapters and weight-tuning helpers;
- executable aliases for retired strategy IDs;
- the unused `scheduling/multiphase/global_ready_set.py` facade, which referenced a removed implementation;
- deprecated planner/predictor registry shims superseded by `planning.asset_registry` and `prediction.asset_registry`.

Retired IDs fail closed in the formal registry. Old experiment artifacts remain readable as data, but their strategy IDs are not executable.

## Retained formal runtime assets

- orthogonal P01/P012/P0123 planners;
- Current/Future timing wrappers;
- Local/Joint scopes;
- Event/Global engines;
- GMWD, RSBC and RSCF cores;
- FIFO, Greedy and Birkhoff deployable controls;
- exact Local/Joint references for bounded offline instances.

## Retained paper/reference baselines

FAST-inspired ordering, Aurora fixed-placement ordering, iSLIP, power-of-two choices and reverse-order controls live under `rs/reference/baselines/`. They are offline/reference-only and cannot enter the online deployment registry.

## Deferred boundaries

`runtime/online/megatron_ep/host.py` remains an intentional bootstrap boundary because its distributed process-group and monkeypatch semantics are externally exercised. It should only be split after a dedicated host-interface migration, not as a cosmetic file-size change.
