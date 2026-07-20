# Formal comparison configurations

This directory contains only current deployable RouterSense strategies and
strong phase-local controls. Retired pre-orthogonal configuration
files were moved to `archive/round1_removed_20260720/configs/`.

The active runtime comparison surface is:

- `disabled`
- `fifo_async_p2p`
- `greedy_async_p2p`
- `birkhoff_phase_local_async_p2p`
- `routersense_<current|future>_<p012|p0123>_<local|joint>_<event|global>_<gmwd|rsbc|rscf>_async`

FAST, Aurora and iSLIP remain offline paper references under
`rs.reference.baselines` and are not accepted by the online runtime wire.
