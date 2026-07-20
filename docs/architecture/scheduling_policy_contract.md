# Formal scheduling policy contract

The installable RouterSense mainline exposes one orthogonal planning model:

`Planner(timing, horizon, scope, engine, core)`

- timing: `current` or `future`
- horizon: `p01`, `p012`, or `p0123`
- scope: `local` or `joint`
- engine: `event` or `global`
- core: `gmwd`, `rsbc`, or `rscf`

Canonical IDs use all five axes, for example:

`future:p012:joint:global:rscf`

The runtime never translates retired policy names. Unknown names fail closed.
The online execution wire accepts only deployable controls and the
`prepared_priority` materializer; offline reference algorithms are excluded.

P012 is the deployment default. P0123 remains a current-timing ablation until
its Future lifecycle is separately production-gated. RSCF is the primary core,
RSBC the stable alternative, and GMWD an algorithm ablation.
