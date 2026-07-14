M5 Phase A now has a single formal experiment-system surface under `rs.experiments`.

Closed in this branch:
- typed `ExperimentSpec`, `SuiteSpec`, `PlanningCase`, and `RunPlan`;
- `ExperimentConfigLoader` with schema v2 validation and explicit rejection of lossy schema v1 configs;
- canonical `python -m rs.experiments.cli` subcommands for inspect/plan/run/validate/list;
- `RunnerRegistry` for Phase A run kinds;
- formal v2 official configs that preserve model/topology/workload/runtime/evaluation semantics;
- default `pytest --collect-only` succeeds on this branch.

Deliberately deferred:
- service-backed execution inside the non-diagnostic runners;
- reporting/plotting/reproduce flows;
- retirement of legacy `rs.core.experiment_config` after downstream migration.
