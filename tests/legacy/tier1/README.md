# Legacy Tier-1 Witnesses

These tests preserve the pre-P012 Tier-1 scheduler recovery effort. They are
not contracts for the current `gmwd` / `rsbc` / `rscf` P012, P0123, or
Future-P012 mainline.

The suite was already guarded by `ROUTERSENSE_ENABLE_LEGACY_TIER1=1` and was
excluded from normal pytest collection. During the 2026-07-19 all-ready audit,
forcing the suite against the current refactored scheduler produced 23 passes
and 20 failures. The failures include obsolete diagnostic keys, historical
plan-digest expectations, and missing helpers inside the old test file.

Do not repair these witnesses by changing current planner semantics. Revive
them only in a dedicated historical-compatibility branch with an explicit
schema adapter and regenerated provenance evidence.
