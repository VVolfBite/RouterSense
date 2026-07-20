# Runtime contracts

## Online Megatron EP runtime

The formal package is `rs.runtime.online.megatron_ep`.

- `host.py`: process groups, Megatron dispatcher discovery, hook install/restore.
- `lifecycle.py` and `lifecycle_parts/`: P0/P1 observation, current/future
  planning, prediction publication, bounded target reconciliation, evidence.
- `control/`: root-authoritative agreement and tensor-only control traffic.
- `phase/`: canonical flow/layout contracts.
- `execution/`: compiler facade, preflight, async P2P/sync execution and audit.
- `target_planning/`: immutable Future-P012 planning service and truth binding.

Frozen safety properties:

1. executable P0/P1 coverage is derived from actual traffic, never predicted P2;
2. canonical tasks must have no missing/extra task IDs before transport starts;
3. send/receive row, byte, dtype and shape contracts agree on all ranks;
4. a failure before transport may fail closed; a failure after P2P begins cannot
   switch to a different transport path;
5. prediction repair emits each actual edge once and keeps newly inserted P1
   blocked until its P0 dependency completes;
6. Current, Safe Local, and Future planners consume the same measured link-cost
   profile.

## Offline runtime

`rs.runtime.offline` owns trace loading, traffic reconstruction, prediction
artifacts, logical execution, oracle references and reproducible reports. It
must not be used as an online fallback.

## Scheduling boundary

`rs.scheduling` owns logical flows, matching, release-aware planning and plan
objects. It must not import torch, Megatron, NCCL, deployment scripts or
experiment entrypoints.
