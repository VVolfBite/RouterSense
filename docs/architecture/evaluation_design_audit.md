# Evaluation Design Audit

This document records the current RouterSense evaluation design as implemented in `RS/` and the remaining boundaries before multi-GPU experiments.

## Core Claim Structure

RouterSense has two evidence tracks plus one prediction sub-track.

1. Theory/modeling: the communication scheduler is a release-constrained bipartite matching / job-shop style problem. This is primarily a paper argument, not a runtime artifact.
2. Offline trace analysis: collect router traces, rebuild traffic matrices and FlowWindow inputs, replay multiple schedulers under a common logical cost model, and estimate the opportunity for multiphase scheduling and lookahead.
3. Online execution: run real Megatron EP, observe P0/P1 facts, inject phase-local executable plans, audit actual NCCL bucket/wave execution, and compare strategy metrics.

## Corrections To Keep Explicit

- `birkhoff_phase_local` is not the formal oracle. It is an online-capable phase-local decomposition/adaptation. The formal fluid reference is `birkhoff_von_neumann_fluid`, and the small discrete certified reference is `exact_small_instance_reference`.
- `routersense_p0p1p2_hint` is currently an online phase-local policy that consumes PreparedWindowPlan-derived P2 hints. It is not the same as online multiphase joint execution.
- `routersense_multiphase_lookahead:*` and Tier 1 `U_*` policies remain offline logical schedulers. They must not be presented as executed by the current online `phase_sync_wave` runtime.
- `calibrated_artifact` now has two meanings that must not be mixed: offline calibrated predictor artifacts are still fail-closed; online `calibrated_artifact` P2 hints are derived from a prior layer `PreparedWindowPlan` and are valid for phase-local online evaluation.
- Runtime comparison metrics are measurements, not proof of performance claims until the multi-GPU experiment protocol is run and archived.

## Offline Trace Line

Implemented support:

- `experiments/offline/collect_router_trace.py` collects single-GPU router traces.
- `experiments/offline/run_flow_schedule_study.py` builds FlowWindow-style inputs and replay/audit outputs.
- `experiments/offline/run_tier1_cpu_validation.py` runs recovered Tier 1 candidates without CUDA or NCCL.
- `docs/migration/tier1_recovery_provenance.json` records historical source and semantic witness provenance.

Required checks before paper tables:

- Trace schema and matrix reconstruction must be audited on at least one small hand-checkable fixture and real model trace.
- Comparisons must remain grouped by service model: atomic, fluid, Lagrangian/other.
- Oracle or perfect-trace modes must stay `evaluation_eligible=false`.
- `runtime_lookahead` must never schedule real P2 transfer.

## Online Runtime Line

Implemented support:

- `RouterSenseObserver` records native dispatcher facts.
- `RouterSenseInjectionRuntime` builds PhaseReadyContext, agrees phase plans, activates transport, records timeline, and now carries PreparedWindowPlan-derived P2 hints across layers.
- `ExecutionAudit` verifies planned vs actual task execution.
- `experiments/online/run_strategy_comparison.py` orchestrates repeated strategy runs and produces `comparison_report.json` / `.md`.

Required checks before multi-GPU interpretation:

- `rank*_plan_arrival_records.jsonl` must show whether P2 hints arrived before commit, in flight, or not at all.
- `rank*_scheduled_phase_plans.jsonl` and `rank*_transport_execution.jsonl` must support execution-audit status `passed` for every claimed injected policy.
- `transport_mutation=true` is required for phase-local policy execution claims.
- `default_continue` results must distinguish shadow plan arrival from actual transport mutation.

## Metrics Coverage

The current comparison framework computes:

- communication makespan from `before_wave` / `after_wave` timeline events;
- P0/P1 transport summaries from transport execution rows;
- wave counts from scheduled plans;
- plan agreement timing from `PhaseExecutionPlan.metrics`;
- plan arrival counts and average age from `plan_arrival_records`;
- net benefit and benefit ratio relative to a configured baseline.

Remaining useful extensions:

- rank-wise tail metrics and per-layer p95/p99 summaries;
- richer plots for long-tail layers;
- explicit native-vs-injected logits comparison aggregation in strategy reports;
- cross-run manifest pinning for model hash, commit, CUDA/NCCL versions, and topology.
