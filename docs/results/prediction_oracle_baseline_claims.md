# Prediction / Oracle / Baseline Claims

## Can Enter Paper

- The unified canonical bucket-wave exact oracle reached certified `OPTIMAL` for both scopes on 32/32 sampled tiny instances.
- `O_joint <= O_local` held in all 32 cases with zero dominance violations. Mean improvement was 6.49%; 9 strict-win instances had 23.07% mean improvement.
- Formal `O_local` and `O_joint` share tasks, bucketization, wave cost, release semantics and replay; only scope changes. Historical atomic CP-SAT and BvN fluid results are sensitivity references only.
- On replay fixtures, the strongest joint heuristic `U_gated_maxweight_matching` achieved median gain 1.24% vs FIFO and 1.31% vs Birkhoff.

## Partially Supported

- copy-current recovered 121.84% of perfect-trace-hint potential gain on average.
- perfect-trace-hint beat zero-hint on 85.94% of paired comparisons; this is not stable enough to claim universal scheduler consumption.

## Not Yet Safe to Claim

- Matrix prediction accuracy alone does not prove schedule improvement; the closure only reports empirical correlation.
- safe-U CPU planning overhead is measured offline, not end-to-end GPU net benefit.