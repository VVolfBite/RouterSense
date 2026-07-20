# RouterSense Pre-Evaluation Mainline

This document freezes the formal codebase shape before policy evaluation.

## Formal Mainline

- `src/rs/core/`: shared contracts, hashing, artifact helpers, validation.
- `src/rs/scheduling/`: policy ABI and reusable scheduling logic.
- `src/rs/runtime/offline/`: offline trace, traffic, prediction, replay.
- `src/rs/runtime/online/megatron_ep/`: formal Megatron EP online runtime.
- `experiments/offline/`: offline experiment entrypoints only.
- `experiments/online/`: online runner entrypoints only.
- `legacy/`: historical code retained for reference, never a formal runtime import target.

## Dependency Direction

```text
core
  ↑
scheduling
  ↑
runtime/offline        runtime/online
  ↑                        ↑
experiments
  ↑
scripts / deploy
```

Forbidden reverse dependencies:

- `scheduling` must not import torch, Megatron, NCCL, experiment runners, or artifact paths.
- `runtime` must not import `experiments`.
- `experiments` must not contain reusable runtime logic.
- `legacy` must not be imported by formal runtime code.

## Round-1 Scope

Round 1 is structural only:

- create the formal package paths;
- expose existing validated implementations through compatibility wrappers;
- keep old code reachable while marking it legacy;
- avoid algorithm, executor, or hook behavior changes.

## Frozen Runtime Boundary

The following semantics are considered frozen before evaluation:

- P0/P1 pre-transport hook timing
- dispatcher probe contract
- dual-endpoint transfer layout
- root-authoritative sync-before-phase agreement
- Megatron phase transport adapter
- phase sync wave executor
- P0 hidden-state plus routing-probability atomic bundle
- P1 bundle contract

New policies must plug into the formal scheduling ABI instead of editing these layers.

## Immediate Follow-up Rounds

Round 2:

- split mixed-responsibility files;
- normalize naming for P0/P1/P2, flow, offsets, rank identity, and plan objects;
- finish moving agreement and layout helpers out of FIFO-named modules.

Round 3:

- run full CPU tests;
- run archive self-check;
- add offline golden regression fixtures;
- run minimum online GPU correctness checks;
- tag a pre-evaluation snapshot.
