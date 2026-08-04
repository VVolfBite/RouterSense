# RouteSense PoC1 Offline Summary

## Core

- Pairwise ranking accuracy: 0.4633093525179856
- Pairwise comparisons: 3475

## Strategy Snapshot

- random: mean=0.000758, p95=0.044434, safe@0.01=0.8750
- raw_routing: mean=-0.001098, p95=0.019531, safe@0.01=0.9141
- oracle: mean=-0.066552, p95=-0.000005, safe@0.01=0.9922

## Conditional Findings

- Entropy split: high=0.488862837045721, low=0.43866591294516677
- Gap split: small=0.4672036823935558, large=0.459412780656304
- Delta magnitude split: large=0.43609022556390975, small=0.47327044025157233

## Oracle Hard Subset

- top32 pairwise=0.4311717861205916, oracle_mean_delta=-0.23781204223632812, random_mean_delta=-0.014803409576416016

## Interpretation

- Current signal is weak globally.
- Subgroup signal is stronger in high-entropy and small-gap cases.
- Oracle still shows substantial headroom relative to current observable routing features.
