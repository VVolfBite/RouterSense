# Controlled Local/Joint Scheduling Families

RouterSense now separates an algorithm's **wave-selection kernel** from its
**information scope**.

```text
Local(family)  = run the same kernel independently on P0, P1, and executable P2
Joint(family)  = run the same kernel once on the global release-aware ready set
```

The local and joint forms share matching method, weights, tie-breaking,
service model, bucket/task contracts, cost model, and solver budget.  The only
intended treatment variable is the visible ready set and cross-phase release
coupling.

## Strict same-core families

| Family | Local ID | Joint ID | Purpose |
|---|---|---|---|
| Gated Greedy | `gated_greedy_local` | `gated_greedy_joint` | Low-cost control for joint-ready-set value |
| Gated MaxWeight | `gated_maxweight_local` | `gated_maxweight_joint` | Historical Tier-1 maximum-weight candidate |
| Barrier Criticality | `barrier_criticality_core_independent` | `barrier_criticality_joint` | Main RouterSense family |
| Birkhoff-Ranked | `birkhoff_ranked_local` | `birkhoff_ranked_joint` | BvN round ordering used as the same priority kernel in both scopes |
| Adaptive Price | `adaptive_price_local` | `adaptive_price_joint` | Dual/price-style candidate with identical price updates |

Expression aliases are accepted:

```text
Local(gated_maxweight)
Joint(gated_maxweight)
Local(barrier_criticality)
Joint(barrier_criticality)
Local(birkhoff)
Joint(birkhoff)
```

Historical `B_*` and `U_*` names for greedy, MaxWeight, barrier criticality,
and adaptive price resolve through this same layer.  They are compatibility
names rather than independent implementations.

## Legacy candidates not treated as strict pairs

`B_lagrangian_phase_local` versus `U_lagrangian` and `B_birkhoff` versus
`U_ibbr` remain available for exploratory analysis.  They are not admitted to
the strict family claim because the two sides do not share one solver core:
`U_ibbr` adds iterative repair, while the recovered Lagrangian implementations
use different update structures.

## Measurements

`experiments.paper.family_evaluation.evaluate_family_pairs` records, for each
instance and family:

- local and joint replay-valid makespan;
- joint-minus-local makespan and improvement percentage;
- win/tie/loss outcome;
- median, p95, and maximum planning wall time;
- joint-minus-local planning overhead and runtime ratio;
- deterministic semantic plan digests;
- common-core contract equality;
- wave count and served volume by phase.

A CPU fixture runner is available:

```bash
python experiments/offline/run_family_pair_validation.py \
  --fixture tests/fixtures/tier1/unlock_hotspot_4rank.json \
  --families gated_greedy,gated_maxweight,barrier_criticality,birkhoff_ranked,adaptive_price \
  --warmups 5 \
  --repeats 30 \
  --output outputs/family_pilot.json
```

The runner validates experiment infrastructure only.  A positive result on a
small witness is not evidence that joint scheduling universally wins; formal
paper claims require paired statistics across frozen real-trace instances.
