# RSCF Joint versus FAST-style across EP sizes

This is the preserved offline comparison record produced before the PPIO
hardware deployment. It is a logical-time experiment, not a CUDA/NCCL result.

## Scope

- 252 non-terminal validation windows;
- DeepSeek-V2-Lite, OLMoE and Qwen1.5-MoE traces;
- EP sizes 4, 8, 12 and 16;
- perfect P2;
- zero expert-compute delay;
- one common logical execution/audit model;
- FAST-style phase-local core versus RSCF Local/Event and RSCF Joint/Event.

## Result

| EP | FAST-style mean makespan | RSCF Local mean | Local vs FAST | RSCF Joint mean | Joint vs FAST | Joint win/tie/loss |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 106.67 | 105.38 | 1.11% | 96.44 | **9.79%** | 252 / 0 / 0 |
| 8 | 69.28 | 69.15 | 0.25% | 59.90 | **13.45%** | 249 / 2 / 1 |
| 12 | 63.05 | 62.38 | 1.42% | 52.68 | **16.19%** | 248 / 3 / 1 |
| 16 | 51.97 | 49.49 | 6.16% | 42.18 | **19.35%** | 252 / 0 / 0 |

Positive percentages mean lower communication makespan than FAST-style.

## Interpretation

At EP=4--12, RSCF Local and FAST-style are close. The larger separation appears
when the same traffic is planned jointly across P0/P1/P2. This supports the
claim that the main gain comes from multi-stage dependency-aware planning rather
than from selecting an artificially weak phase-local matcher.

The EP=16 Local gap should not be presented as a complete FAST-system result.
The implemented baseline is FAST-style: it contains the server-level one-to-one
stage core but not the complete endpoint-mutating rebalance and hardware
pipeline of the original system.

## Files

- `docs/results/data/fast_vs_rscf_ep_all252_summary.csv`
- `docs/results/data/fast_vs_rscf_ep_all252.json.gz`
