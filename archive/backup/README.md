# Backup Index

This directory is git-controlled and intentionally keeps only the current
high-value RouteSense backup snapshots.

## Shared Historical Docs

The root-level historical planning and task documents are also copied here:

- `docs/ELIMINATION_LOG.md`
- `docs/new-direction.md`
- `docs/poc-line-1.md`
- `docs/poc1-legacy-task.md`
- `docs/poc1-refined.md`
- `docs/rs-task.md`

These are archival copies only. The original files at repository root are left
in place for now.

## Current retained archives

1. `20260703_cross_layer_prediction_validity`
   - Why kept: preserves the evidence that cross-layer prediction is simple and
     effective.
   - Structure: `artifacts/`, `reports/`, `docs/`, `README.md`

2. `20260703_oracle_fast_gap_study`
   - Why kept: preserves the evidence that oracle / strong fast baselines still
     leave material optimization headroom.
   - Structure: `artifacts/`, `reports/`, `docs/`, `README.md`

3. `20260703_execution_window_multiscale`
   - Why kept: preserves the current best execution-window scheduler comparison
     used for deployment candidate selection.
   - Structure: `artifacts/`, `reports/`, `docs/`, `README.md`

No other active backup snapshots are retained here at this time.
