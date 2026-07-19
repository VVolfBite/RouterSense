# RouterSense Offline P01/P012 Decomposition and RSCF Study

Date: 2026-07-17  
Trace commit: `5647b219a3ff8c9092e056d8e4f42ed8dcddce43`  
Trace artifact SHA-256: `afe11714efb54eeccb2ea57692fc15ff056b997646f12f573028f07894838f79`

## Status

This is a development-stage offline study on real OLMoE routing traces with
virtual EP reconstruction. It is suitable for mechanism selection and source
validation, but it is not the frozen paper test set. The current corpus contains
encoding damage in Chinese prompts, repeated prompt templates, and mechanical
length padding. RSCF v4 was selected using this development data chain, so its
numbers must be revalidated on the replacement trace before paper claims are
frozen.

## Fixed execution model

The study assumes fixed logical token ownership, fixed-top-k token-choice
routing, dropless execution, expert outputs returning to the source rank, P1
volume equal to the transpose of P0, and a fixed payload size per assignment
within each phase. P1 volume is therefore known after P0 routing, while P1 ready
time remains a runtime event. P2 is the next-layer dispatch and is not exactly
known until the next router executes.

## Controlled comparison axes

The experiments keep three axes separate:

- Coupling scope: `Local(f)` or `Joint(f)`.
- Information scope: P01, predicted P012, or perfect P012.
- Planning timing: upfront or reactive row reveal.

The primary same-core comparison gives Local and Joint exactly the same perfect
P012 matrices upfront. Local invokes the shared family kernel independently for
P0, P1, and P2; Joint invokes it over one release-aware global ready set.

The P2 information simulator uses the following semantics:

- Perfect: true P2 is known upfront and participates in the release-aware
  execution window.
- Reactive: P0/P1 are known upfront; each true P2 source row is revealed only
  after that source rank completes its P1 inbound barrier.
- Predicted: a forecast may affect scores, but executable P2 bytes are created
  only when the corresponding true row is revealed.

Predicted bytes never become execution truth.

## Trace scope

- 48 prompts: 32 development and 16 validation.
- 16 MoE layers, 64 experts, fixed top-8 routing.
- 572,928 token-expert trace records.
- 3,072 TrafficInstances over vEP 2, 4, 8, and 16.
- Validation split: 1,024 instances; 960 have a real next-layer P2.
- Routing was captured on one RTX 4080. EP traffic is reconstructed with
  stable token ownership and contiguous balanced placement; it is not measured
  multi-GPU transport timing.

## Baseline family recheck

The table uses all 1,024 validation TrafficInstances. Improvement is
`(Local - Joint) / Local`. Planner time is Python CPU kernel time and must not be
subtracted directly from abstract traffic makespan.

| Family | P01 mean / median | P01 regression | P012 mean / median | P012 regression | Local / Joint median kernel ms |
|---|---:|---:|---:|---:|---:|
| Greedy Control | -20.55% / -22.59% | 82.13% | -20.75% / -21.53% | 85.45% | 4.52 / 5.70 |
| GMWD-style | -20.72% / -21.71% | 82.71% | -21.94% / -22.17% | 88.96% | 5.17 / 5.92 |
| RSBC | +2.80% / 0.00% | 7.52% | +7.57% / +6.08% | 3.42% | 4.44 / 4.61 |
| FAST-Stage | +1.23% / 0.00% | 30.76% | +4.46% / +3.30% | 13.09% | 4.14 / 5.87 |

The result is not that GMWD or Greedy are bad single-phase algorithms. It shows
that directly extending a single-phase residual-volume objective to a global
multiphase ready set is not sufficient. A useful Joint family needs an explicit
release-aware cross-phase objective.

## P2 contribution

For structural decomposition, define:

- `Local(P012)`: independent P0, P1, and P2 plans.
- `P01-only total`: `Joint(P01) + Local(P2)`.
- `Joint(P012-perfect)`: one perfect-information release-aware window.

Then:

- P01 coupling gain = `Local(P012) - P01-only total`.
- P2 and cross-phase gain = `P01-only total - Joint(P012-perfect)`.
- Total Joint gain is their sum.

The P2 term includes both advance P2 information and the ability to interleave
released P2 traffic with unfinished earlier phases. It is intentionally not
labelled as prediction accuracy gain.

### RSBC and RSCF on P2-available validation windows

| Family | P01 coupling mean / median | P2 + cross-phase mean / median | Total mean / median | Total regression |
|---|---:|---:|---:|---:|
| RSBC | +1.90% / 0.00% | +6.05% / +4.00% | +7.91% / +7.02% | 3.13% |
| RSCF v4 | +2.24% / 0.00% | +7.32% / +5.12% | +9.50% / +9.03% | 2.19% |

On positive-gain RSCF windows, the median ratio attributed to the P2/cross-phase
term is 1.0 and the mean is about 0.80. The ratio is unstable when total gain is
small, so the percentage-point decomposition above is the primary result.

### RSCF by virtual EP size

| vEP | P01 median | Full P012 median | P2 + cross-phase median | Perfect upfront vs reactive median |
|---:|---:|---:|---:|---:|
| 2 | 0.00% | 0.00% | 0.00% | 0.00% |
| 4 | 0.00% | 5.06% | 3.43% | 3.62% |
| 8 | 0.00% | 14.37% | 10.81% | 9.05% |
| 16 | 0.00% | 18.58% | 16.75% | 12.37% |

This trace therefore does not support treating P2 as merely a way to hide
planning overhead. At vEP8 and vEP16, P2 information and cross-phase execution
are the dominant source of scheduling gain.

## RSCF heuristic

RouterSense Critical Frontier (RSCF) is a model-agnostic extension of RSBC. It
uses only residual traffic, source/destination endpoint loads, and the generic
P0-to-P1-to-P2 release DAG. It does not consume model name, expert identity,
expert count, layer ID, top-k value, or OLMoE-specific features.

For each ready flow, RSCF combines:

1. the shared RSBC residual, barrier, age, and immediate release components;
2. a smooth dual price over barrier-to-tail critical paths;
3. source and destination endpoint bottleneck dual prices;
4. the fraction of a barrier unlocked by serving the flow, multiplied by the
   transitive downstream tail it exposes;
5. a small destination-hotspot exposure term for future P2 traffic.

The resulting scores are passed to the same maximum-weight bipartite matching
core used by Local and Joint. Local and Joint share every RSCF parameter and
differ only in ready-set visibility and phase coupling.

Registered provisional v4 parameters:

```text
residual_weight             = 0.15
barrier_weight              = 0.50
age_weight                  = 0.30
release_gain_weight         = 2.50
critical_path_weight        = 0.25
transitive_unlock_weight    = 2.50
endpoint_dual_weight        = 1.00
dual_temperature            = 0.20
transitive_tail_weight      = 0.25
destination_hotspot_weight  = 0.10
duplex_pair_weight          = 0.00
size_bias_power             = 0.00
```

The disabled pairability and size-bias terms remain only as ablation hooks and
are not part of the active v4 mechanism claim.

## RSCF effect and overhead

Across all 1,024 validation instances:

- P01 Local-to-Joint: mean +3.29%, median 0.00%, regression 4.88%.
- P012 Local-to-Joint: mean +9.11%, median +8.39%, p90 +21.13%, regression 2.25%.
- Perfect P2 upfront vs reactive row reveal: mean +6.74%, median +5.34%,
  regression 4.27%.

Compared with RSBC on P2-available windows, RSCF raises median total gain from
7.02% to 9.03% and lowers regression from 3.13% to 2.19%.

A stable vEP4 timing subset used 16 instances, two warmups, and five measured
runs per scope:

| Family | Joint effect mean / median | Local kernel p50 | Joint kernel p50 | Joint p95 | Joint/local p50 ratio |
|---|---:|---:|---:|---:|---:|
| RSBC | 6.09% / 4.46% | 1.10 ms | 1.22 ms | 1.57 ms | 1.11x |
| RSCF | 7.42% / 5.92% | 2.65 ms | 3.51 ms | 3.94 ms | 1.33x |

RSCF is currently a Python CPU reference. Its larger-vEP cost grows sharply
because critical prices are rebuilt each wave. Before online deployment it
needs cached/incremental endpoint statistics, vectorized sparse state updates,
and likely a compiled implementation. Runtime net benefit must be evaluated
with measured GPU communication and compute overlap; abstract traffic units and
planner milliseconds are not directly commensurate.

## Prediction sensitivity POC

A development POC perturbed true P2 while preserving each source-row total. It
is a rank-destination corruption model, not a FATE accuracy claim. Sixty-four
stratified instances and two replicates were used.

| Corruption | Mean remote L1 | Predicted gain vs reactive mean / median | Median capture ratio | Regression vs reactive |
|---:|---:|---:|---:|---:|
| 10% | 0.089 | 7.77% / 6.21% | 1.00 | 3.13% |
| 25% | 0.178 | 7.84% / 6.40% | 1.00 | 3.13% |
| 40% | 0.247 | 7.55% / 4.33% | 1.00 | 4.69% |

The current signal is that RSCF is less brittle than hard edge ordering under
moderate row-total-preserving errors. At higher error, median captured gain
falls. Real FATE logits, candidate scores, and calibrated uncertainty are still
required before choosing a production prediction representation.

## Engineering guard

`rscf_runtime_safe` / `Safe(rscf)` is included only as a final offline guard. It
runs `Joint(rscf)` and `Local(rscf)` with the same information and retains the
lower audited makespan. It is not deployable and is not presented as the main
algorithmic improvement.

## Correctness fixes

The formal runtime adapter previously forwarded generic non-null
`PlanningWeights` defaults into every scoped family, silently replacing each
family's registered kernel parameters. The adapter now preserves the immutable
family specification for scoped Local/Joint policies.

The scheduler continues to honor the caller-provided `max_waves` exactly and
fails closed when residual traffic remains. Dense offline experiments request a
larger explicit cap through their own entrypoint; runtime safety limits are not
silently expanded.

## Validation

- Python compileall: pass.
- Focused scheduling/P2 suite: 37 passed, 2 skipped.
- Additional closure suite: 15 passed, 1 skipped.
- A broader suite reached 43 passed and 3 skipped before one unrelated
  repository-metadata failure: the unpacked source has no `.git`, so canonical
  result-bundle tests reject `commit_sha=unknown`.
- The full repository run was not claimed as passing; it timed out after the
  same class of repository-metadata failures.

## Required follow-up on the replacement trace

1. Freeze a clean natural-language development/validation/test split.
2. Rerun the exact P01/P012 decomposition without retuning on validation/test.
3. Export real FATE scores or full expert logits, not only selected top-k.
4. Measure RSCF planning time and net latency on the real multi-GPU runtime.
5. Optimize incremental critical-price maintenance before enabling RSCF online.
6. Keep the multi-candidate safe wrapper as a separate engineering ablation.
