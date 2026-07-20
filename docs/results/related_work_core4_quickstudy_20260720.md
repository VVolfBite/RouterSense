# Related-work style core quick study — 2026-07-20

## Scope

This checkpoint evaluates four related-work style scheduling cores under the
same RouterSense fixed-endpoint, phase-serial logical execution contract:

- `gmwd_style_reference`
- `fast_stage_reference`
- `islip_reference`
- `aurora_order_reference`

The run covers 792 traffic instances from Qwen1.5-MoE, OLMoE, and
DeepSeek-V2-Lite trace packages. P0, P1, and the actual P2 matrix are executed.
Expert-compute delay is zero and time is logical traffic service time, not
measured CUDA/NCCL wall time. All current trace instances use virtual EP=4. The
FAST-style core therefore uses its deterministic inferred 2-server × 2-GPU
logical topology; this is a scheduling-core study, not a claim about the
physical topology of the trace capture machine.

## Implemented mechanisms

| Policy | Implemented style core | Explicitly not reproduced |
|---|---|---|
| GMWD-style | residual matrix, maximum-weight matching, common service-quantum subtraction | photonic reconfiguration, profiled expert compute, communication/compute overlap objective |
| FAST-style | server-level traffic collapse, one-to-one server stages, GPU-edge realization, intra-server idle-port fill | endpoint-mutating rebalance, redistribution cost, scale-up/scale-out pipeline timing |
| iSLIP-style | request/grant/accept, persistent round-robin pointers, first-iteration pointer updates | switch-cell timing and hardware queueing model |
| Aurora-style | bottleneck-sender seed, descending sender-load order, receiver-conflict-avoiding slot placement | expert colocation, heterogeneous GPU assignment, deployment optimization |

## Aggregate results

All policies produced valid plans for all 792 instances.

| Policy | Mean makespan | Median makespan | Median waves | Median planning | Mean vs Birkhoff | W/T/L vs Birkhoff | Mean vs Greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 105.874 | 97.0 | 9 | 0.439 ms | -9.13% | 37/88/667 | -0.09% |
| Greedy | 107.931 | 98.0 | 10 | 0.459 ms | -9.44% | 46/59/687 | 0.00% |
| Birkhoff phase-local | 98.657 | 91.0 | 31 | 6.064 ms | 0.00% | 0/792/0 | +8.16% |
| GMWD-style | **97.501** | **90.5** | 30 | 1.930 ms | **+1.34%** | **255/521/16** | **+9.47%** |
| FAST-style | 97.843 | 90.5 | 30 | 1.496 ms | +0.89% | 227/465/100 | +9.07% |
| iSLIP-style | 106.514 | 98.0 | 9 | 0.639 ms | -10.31% | 37/86/669 | -1.18% |
| Aurora-style | 106.597 | 98.0 | 10 | 0.451 ms | -8.57% | 46/73/673 | +0.54% |

Planning time is Python CPU wall time and must not be subtracted from logical
makespan without a calibrated transport-time model.

## Interpretation

1. GMWD-style and FAST-style are currently the meaningful strong external
   scheduling baselines. They preserve the advantage of residual/fluid service
   decomposition and slightly improve over the existing phase-local Birkhoff
   implementation on this trace set.
2. Their median advantage over Birkhoff is zero because most instances tie;
   the positive mean comes from a subset of skewed windows. These baselines are
   credible but not artificially weak.
3. iSLIP-style and Aurora-style are useful behavioral controls rather than
   performance leaders under the current objective. iSLIP targets iterative
   arbitration/fairness, while the fixed-placement Aurora adaptation lacks the
   placement component that gives the complete system more freedom.
4. The formal RouterSense RSCF Local-vs-Joint regression remains PASS after
   these changes. Across EP 4/8/12/16, the best Event/Global communication gain
   remains 9.40%, 13.72%, 12.93%, and 11.61%, respectively, with zero selected
   window regressions.

## Validation completed for this checkpoint

- 792/792 valid plans for each of seven policies.
- Four related-work cores validated on phase-barrier, receiver-incast,
  full-duplex-pair, and skewed-8-rank fixtures.
- Targeted core/strategy/config/deployment tests passed.
- Static project-import scan checked 1,111 internal imports with zero unresolved
  references.
- The full 160-file release gate was intentionally not rerun for this agile
  checkpoint.
