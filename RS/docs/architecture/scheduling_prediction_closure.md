# Scheduling and Prediction Closure

This document fixes the formal boundary between the scheduling core and the
next-layer traffic predictor.

## Scheduling family contract

Each scoped family has one immutable kernel specification. `Local(f)` applies
that kernel independently to P0, P1, and P2. `Joint(f)` applies the same kernel
to one release-aware global ready set. The primary families retain their native
base objective and add the same RouterSense critical-frontier potential:

- Greedy keeps greedy residual ordering.
- GMWD keeps residual maximum-weight decomposition.
- RSBC keeps barrier and release-gain scoring.
- FAST-Stage keeps BvN-derived stage priority.

The lift uses only rank-to-rank residual traffic, endpoint load, and the
P0-to-P1-to-P2 release DAG. It must not inspect model identity, layer identity,
expert identity, expert count, or top-k. Literature-derived families must be
reported with the `-CF` / `+ RouterSense lift` qualifier when this lift is
active.

## Runtime information levels

At the completion of router L, P0(L) is observed and P1(L) is exactly derived
in volume as the transpose of P0. P1 readiness remains event-driven. P2 is the
next MoE dispatch and is not true until router L+1 executes.

The exact information ladder is therefore:

1. `exact_joint_p01_reactive`: start with true P0/P1; reveal each true P2
   source row only after its P1 barrier; solve each currently available tiny
   problem exactly and commit one wave.
2. `exact_joint_p012_predicted`: use the same rolling exact procedure, but an
   advisory forecast of unrevealed P2 participates in planning. Forecast bytes
   are never executable.
3. `oracle_joint_p012_perfect`: expose true P0/P1/P2 upfront and solve the whole
   tiny release-aware window exactly. This is a performance upper bound, or
   equivalently a makespan lower bound, for the exact reference model.

The first two are rolling policies under different information. Only the third
is clairvoyant-global exact.

## Prediction contract

The canonical predictor output is `TrafficForecastEnvelope`, not a hard
execution matrix. It contains an expected matrix, calibrated integer bounds,
per-source remote pressure, and stable partial precedence. Exact
predict-then-optimize controls consume the expected matrix. Online heuristics
primarily consume rank pressure and critical-frontier values; stable order is a
low-weight tie-break.

Full routed-expert scores are mapped using a fixed-top-k capped-inclusion
projection. For each token, inclusion mass lies in [0, 1] and sums to top-k.
Largest-remainder rounding preserves the known assignment total of each source
rank. Shared/local experts are excluded from routed EP traffic before mapping.

## Required output metrics

Report traffic error and scheduling value separately:

- remote matrix relative L1;
- rank-pressure relative L1;
- exact perfect-P2 information value;
- exact predicted-P2 value and capture ratio;
- heuristic predicted-P2 value and exact-predicted gap;
- Local/Joint P01 and P012 gains;
- planner runtime separately from logical makespan.

A low matrix error is not sufficient evidence of scheduling value. Conversely,
a predictor can be useful even when edge-level error is nontrivial if it
preserves the critical frontier.
