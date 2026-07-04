# RouteSense Report

## Current State

The mainline is now explicitly split into three semantic lanes:

- `offline`
- `online`
- `legacy`

This split is about truth conditions, not just folder names.

## What Each Lane Means

### `offline`

Allowed:

- oracle future trace
- full-sequence single-GPU router observation
- calibrated counterfactual analysis
- scheduler research

Not allowed:

- production EP throughput claims
- online runtime claims
- claiming deployable prediction when using oracle future trace

### `online`

Target meaning:

- real per-rank local input ownership
- real expert residency
- real EP dispatch/compute/combine execution
- no future ground truth in the hot path

Current status:

- package skeleton and metadata contracts exist
- Phase 1 does not yet implement a working online EP runtime

### `legacy`

Meaning:

- deprecated compatibility path for the old distributed trace replay harness

Current old distributed path is now classified as:

- `pipeline=legacy`
- `execution_mode=legacy_trace_replay`
- `trace_origin=legacy_trace_replay`

It is not the formal online runtime.

## Code Facts The Mainline Now Explicitly Admits

The old distributed replay path still has these semantics:

- each rank loads a full model
- each rank traces the same prompt
- future router truth is available through full trace collection
- source ownership is synthetic
- expert residency comes from full-model extraction, not physical sharding
- transport is custom replay, not real host EP runtime dispatch/combine

That is why the path is now marked `legacy_trace_replay` everywhere relevant.

## Result Semantics

Formal result metadata now includes:

- `pipeline`
- `claim_scope`
- `trace_origin`
- `future_information_mode`
- `is_real_ep_runtime`
- `source_ownership_mode`
- `expert_residency_mode`
- `transport_backend`
- `correctness_status`
- `performance_claim_eligible`

Important implications:

- offline router trace outputs are explicitly non-performance-claimable
- offline calibrated analysis must reject non-online-native observations
- legacy replay outputs cannot masquerade as online EP results

## What Can Be Claimed Now

Allowed:

- the repository now has an auditable offline/online/legacy boundary
- legacy replay outputs are explicitly labeled as replay-only
- offline calibrated analysis now has an input provenance gate
- online scheduler bridge API now explicitly rejects `oracle_full_trace`

Not allowed:

- real online EP performance
- native EP baseline performance
- matching-realized scheduled transport performance
- deployable prediction benefit
- offline oracle makespan interpreted as measured NCCL speedup

## What Is Implemented In Phase 1

Implemented:

- `src/rs/contracts/`
- `src/rs/offline/`
- `src/rs/online/`
- `src/rs/legacy/`
- `experiments/offline/`
- `experiments/online/`
- `experiments/legacy/`
- legacy replay result relabeling
- boundary tests for provenance and future-information rules

Not implemented yet:

- real online native A2A EP
- online observer
- calibrated offline simulator
- scheduled P2P online backend

## Current Recommended Usage

Runnable now:

- offline router prediction collection
- legacy trace replay compatibility harness

Present but intentionally failing fast:

- online native EP benchmark
- online scheduled EP benchmark
- calibrated offline schedule simulator

That failure behavior is intentional. Phase 1 favors semantic correctness over
premature benchmarkability.
