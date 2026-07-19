# Invariant Modes

RouterSense formal experiments now expose three runtime guard modes:

- `evaluation_strict`: for formal offline, C2, and A2. Any fallback, invalid audit, legacy compiler bridge, compiler shadow compare, dirty git tree, timeout, or missing required metric is a hard failure.
- `runtime_safe`: allows explicit safety fallback, but the run is marked `valid_for_performance_evaluation=false`.
- `diagnostic`: allows legacy bridge, shadow compare, dirty-tree execution, and explicit fault-injection workflows. Results are not eligible for formal performance reporting.

Official configs under `configs/official/` are pinned to `evaluation_strict` for online and GPU validation entrypoints. Validation smoke configs may use `diagnostic` when they intentionally exercise compatibility paths during local development.
