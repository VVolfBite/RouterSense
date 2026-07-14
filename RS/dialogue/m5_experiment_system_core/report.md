M5 Phase A now has a single formal experiment-system surface under `rs.experiments`.

Closed in this branch:
- typed `ExperimentSpec`, `SuiteSpec`, `PlanningCase`, and `RunPlan`;
- `ExperimentConfigLoader` with schema v2 validation and v1 migration;
- canonical `python -m rs.experiments.cli` subcommands for inspect/plan/run/validate/list;
- `RunnerRegistry` for Phase A run kinds.

Deliberately deferred:
- service-backed execution inside the non-diagnostic runners;
- reporting/plotting/reproduce flows;
- retirement of legacy `rs.core.experiment_config` after downstream migration.
