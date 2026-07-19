## Pre-Evaluation Handoff

This repository is now at a usable pre-evaluation mainline state.
Both online runtime and formal offline entrypoints are cut over to the canonical `src/rs` mainline.

### What is already true

- `src/rs/runtime/online/megatron_ep` is the canonical home for the current Megatron online runtime implementation
- `src/rs/runtime/offline` and `src/rs/scheduling` are importable formal packages
- `experiments/offline/*` now only contains canonical entrypoints that import `rs.runtime.offline` helpers
- formal placeholder baseline/reference APIs have been cut over to fail-closed `unsupported` contracts instead of reporting fake oracle/optimality metadata
- default `PYTHONPATH=src pytest -q` now runs only formal non-legacy CPU coverage
- source archive packaging self-check passes for the mainline scope
- online experiment wrappers under `experiments/online` are callable with `PYTHONPATH=src:.`
- historical duplicate trees have been removed from the formal tree and parked outside the default validation path
- historical offline experiments now live under `legacy/historical_poc/experiments_offline`
- historical `src/rs/{evaluation,scheduler,trace,legacy}` trees have been moved under `legacy/historical_poc/src_rs_legacy`
- tracked local/current deploy inventories have been removed from Git and replaced by example files plus ignore rules

### What is not yet fully closed

- some historical design docs under `docs/` still reference legacy paths and should be archived or rewritten before paper-facing documentation is finalized
- online prepared-window support is now present for:
  - calibrated P2 hint propagation
  - shadow window state / release event observation
  - prepared phase-plan shadow compilation
  - artifact-side shadow-vs-actual alignment analysis
- online multiphase joint execution is still not implemented:
  - current executor remains phase-local
  - prepared-window plans are used for hinting and observability, not for a shared cross-phase pending queue

### Immediate next cleanup targets

1. convert remaining historical docs to archive-oriented references
2. keep archive provenance and source-only packaging aligned with the post-cutover tree
3. validate prepared-window hint consumption and shadow-plan alignment on single-node multi-GPU runs
4. begin formal evaluation work only inside `src/rs/scheduling`, `experiments`, `configs`, and metrics/plot scripts

### Current safe editing zones

- `src/rs/scheduling/*`
- `src/rs/runtime/offline/*`
- `src/rs/runtime/online/megatron_ep/*`
- `experiments/offline/*`
- `experiments/online/*`
- `experiments/online/support/*`
- `scripts/diagnostics/*`
- `configs/*`
- `docs/architecture/*`

### Frozen zones unless correctness bug is proven

- P0/P1 hook timing
- TransferLayout semantics
- P0 hidden/probs atomicity
- P1 bundle contract
- transport adapter semantics
- sync wave executor semantics
