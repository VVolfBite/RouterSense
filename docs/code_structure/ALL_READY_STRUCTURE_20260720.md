# RouterSense All-Ready Source Structure

## Formal boundaries

- `rs.core.contracts`: configuration, planning, execution, evidence, and performance contracts.
- `rs.planning`: canonical planner registry and Current/Future wrappers.
- `rs.scheduling.p012_future._kernel`: GMWD, RSBC, and RSCF kernels, with Event planning, binding, and shared math split by responsibility.
- `rs.reference.baselines`: offline-only FAST, Aurora, iSLIP, and diagnostic references.
- `rs.runtime.online.megatron_ep`: Megatron EP observation, prepared planning, materialization, execution, and evidence collection.
- `rs.evaluation`: typed offline metric derivation and comparison helpers.

## Runtime lifecycle split

`lifecycle.py` is a public facade. Implementation is divided under
`lifecycle_parts/` into configuration, state, prediction, planning,
asynchronous/joint planning, dispatch hooks, combine hooks, evidence, and
exports. The facade preserves the external import surface without retaining a
monolithic implementation.

## Planner cold start

`target_planning/warmup.py` owns process-level production-kernel preparation.
`TargetLayerPlannerService.start()` completes this synchronous warmup before it
starts the worker thread or accepts tasks. It covers:

- Future-P012 Joint Event RSCF planning;
- Future-P012 Joint Global RSCF planning;
- ordinary P2 truth binding;
- Future prepared-order binding.

Failure is sticky and fails runtime attach closed. Warmup status, duration, and
planner IDs are recorded on the planner-service timeline.

## Formal metric contract

Offline replay and online evidence use `routersense.performance_metrics.v1`,
including communication makespan, P95/P99/max P1 completion, first returned P1
token, planning time, truth-binding time, target-layer entry overhead, wave
count, baseline identity, planner axes, prediction fidelity, and trace/sample
digests. Logical offline time and online wall-clock time remain explicitly
separated by metric domain and unit.

## Deleted material

Historical B/U/Tier1 algorithms, retired executable aliases, old lookahead and
paired-U paths, weight-tuning helpers, unused global-ready-set wrappers, and
deprecated planner/predictor registry shims are outside the active source tree.
FAST, Aurora, and iSLIP remain reference-only and cannot enter the online
planner registry.

## Deliberately retained boundary

`runtime/online/megatron_ep/host.py` remains the bootstrap boundary for process
groups and Megatron monkeypatch installation. It is not split cosmetically;
changes to that ordering require physical distributed retesting.
