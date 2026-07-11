# Prediction / Oracle / Baseline Claims

## Can Enter Paper

- CT exact oracle reached `OPTIMAL` on 32 sampled small instances.
- On sampled exact instances, `O_joint <= O_local` held for all OPTIMAL cases and mean improvement was 6.49%.
- On replay fixtures, the strongest joint heuristic `U_gated_maxweight_matching` achieved median gain 1.24% vs FIFO and 1.31% vs Birkhoff.

## Partially Supported

- copy-current recovered 121.84% of perfect-trace-hint potential gain on average.
- perfect-trace-hint beat zero-hint on 85.94% of paired comparisons; this is not stable enough to claim universal scheduler consumption.

## Not Yet Safe to Claim

- Matrix prediction accuracy alone does not prove schedule improvement; the closure only reports empirical correlation.
- safe-U CPU planning overhead is measured offline, not end-to-end GPU net benefit.