# Literature-Grounded Local/Joint Scheduling Families

RouterSense separates a scheduling kernel from the information scope exposed to
that kernel:

```text
Local(f) = run kernel f independently on P0, P1, and executable P2
Joint(f) = run the same kernel f on the global release-aware P0/P1/P2 ready set
```

The treatment variable is the visibility/release scope.  Matching method,
weights, service model, task/bucket/cost contracts, tie-breaking, and solver
budget are identical within each pair.

## Primary paper families

| Paper label | Local ID | Joint ID | Literature status |
|---|---|---|---|
| Greedy Control | `greedy_control_local` | `greedy_control_joint` | Generic control |
| GMWD-style | `gmwd_local` | `gmwd_joint` | Greedy Max-Weight Decomposition core |
| RouterSense Barrier Criticality (RSBC) | `rsbc_local` | `rsbc_joint` | RouterSense original |
| FAST-Stage | `fast_stage_local` | `fast_stage_joint` | FAST-inspired single-tier stage core |

Expression aliases are accepted:

```text
Local(greedy_control) / Joint(greedy_control)
Local(gmwd)           / Joint(gmwd)
Local(rsbc)           / Joint(rsbc)
Local(fast_stage)     / Joint(fast_stage)
```

Historical names such as `B_gated_maxweight_matching`,
`U_gated_maxweight_matching`, `B_barrier_criticality_core_independent`, and
`U_barrier_criticality_global_matching` resolve through this same scope layer.

## Naming boundaries

### GMWD-style

The implementation operates directly on residual MoE demand, selects a
maximum-weight bipartite matching, and subtracts the minimum selected residual
quantum.  This matches the decomposition core in Amponsah and Addanki (2026),
while RouterSense extends it to a release-aware multiphase window.  It is called
`GMWD-style`, not a full system reproduction, because the paper's photonic
reconfiguration and compute cost model are outside the current runtime model.

### FAST-Stage

FAST combines intra-server rebalancing with balanced one-to-one scale-out
transfers over a two-tier fabric.  RouterSense currently implements only the
one-to-one stage-ordering core and BvN-derived stage priority.  Therefore the
honest label is `FAST-Stage` or `FAST-inspired`; the code and paper must not call
it a complete FAST reproduction until server/NIC hierarchy and intra-server
rebalancing are implemented.

### RSBC

RSBC is the RouterSense-original family.  It combines residual demand,
barrier criticality, age, and a normalized release-gain term.  The release-gain
calculation is part of the shared kernel.  Local does not receive downstream
phase matrices, while Joint can exploit those matrices through the same score
function.

## Experimental strict families

`aurora_order_local/joint` implements only pressure-aware transmission ordering
under fixed placement and is marked Aurora-inspired.  It is not in the primary
paper set because Aurora also optimizes expert/model placement and heterogeneous
cluster cases.

`adaptive_price_local/joint` remains a strict shared-core experimental family,
but is not assigned a named MoE-system label.

## Legacy non-strict candidates

`B_lagrangian_phase_local` versus `U_lagrangian` and `B_birkhoff` versus
`U_ibbr` remain exploratory.  They are excluded from the information-scope
claim because their two sides do not share one solver/update core.

## Measurements

`experiments.paper.family_evaluation.evaluate_family_pairs` records:

- replay-valid Local/Joint makespan;
- win/tie/loss and relative improvement;
- median, p95, and maximum planning time;
- Joint-minus-Local planning overhead and ratio;
- deterministic semantic plan digests;
- common-core contract equality;
- literature label and mapping level;
- wave count and served volume by phase.

The primary evaluation must report raw Joint and any guarded/safe deployment
variant separately.  A safe fallback must not hide raw Joint regressions.
