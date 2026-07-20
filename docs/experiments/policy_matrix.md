# Formal policy matrix

Primary deployment candidates:

- `future:p012:joint:event:rscf`
- `future:p012:joint:global:rscf`

Strict controls:

- `current:p012:local:event:rscf`
- `current:p012:joint:event:rscf`
- `current:p012:local:global:rscf`
- `current:p012:joint:global:rscf`

The same matrix may be instantiated with RSBC or GMWD for core ablations.
Offline paper references are resolved through `rs.reference.baselines` and are
never sent over the online runtime policy wire.
