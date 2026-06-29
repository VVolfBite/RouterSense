# RouteSense PoC1 Offline Summary

## Core

- Pairwise ranking accuracy: 0.4786421499292786
- Pairwise comparisons: 3535

## Strategy Snapshot

- random: mean=0.007788, p95=0.087891, safe@0.01=0.6250
- raw_routing: mean=0.011797, p95=0.085938, safe@0.01=0.7109
- oracle: mean=-0.078440, p95=-0.001129, safe@0.01=0.9922

## Conditional Findings

- Entropy split: high=0.511641113003975, low=0.4458850056369786
- Gap split: small=0.5022650056625142, large=0.4550593555681176
- Delta magnitude split: large=0.4132171387073348, small=0.5203892493049119

## Oracle Hard Subset

- top32 pairwise=0.41037204058624577, oracle_mean_delta=-0.23235702514648438, random_mean_delta=-0.010039806365966797

## Interpretation

- Current signal is weak globally.
- Subgroup signal is stronger in high-entropy and small-gap cases.
- Oracle still shows substantial headroom relative to current observable routing features.
