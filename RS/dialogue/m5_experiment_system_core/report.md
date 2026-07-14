M5 now has a single formal experiment-system surface under `rs.experiments`.

Closed in this branch:
- typed `ExperimentSpec`, `SuiteSpec`, `PlanningCase`, and `RunPlan`;
- `ExperimentConfigLoader` with schema v2 validation and explicit rejection of lossy schema v1 configs;
- canonical `python -m rs.experiments.cli` subcommands for inspect/plan/run/validate/list;
- `RunnerRegistry` with formal offline, Gloo, and single-GPU flow entrypoints;
- formal v2 official configs that preserve model/topology/workload/runtime/evaluation semantics;
- suite-level offline aggregation that emits typed paired aggregates;
- Gloo runner subprocess isolation with quiet JSON summaries;
- committed-tree SHA fallback via handoff manifest when `.git` is absent;
- default `pytest --collect-only` succeeds on this branch.

Deliberately deferred:
- GPU single-card flow requires Megatron/model prerequisites on the host and currently reports `environment_not_run` when they are absent;
- retirement of legacy `rs.core.experiment_config` after downstream migration.
