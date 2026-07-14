Integration now contains the reviewed M1 lifecycle closure plus the latest M2 and M3 branch implementations.

Verified on this merged baseline:
- ancestry includes reviewed M1 plus latest M2 and M3 branch heads;
- `rs.runtime.online.megatron_ep.host` imports cleanly after restoring merged execution exports;
- integration compileall passes;
- focused M2/M3 unit tests pass;
- formal M1 attach/control-group gate still passes after merge;
- formal M1 lifecycle publication gate still passes after merge;
- formal M2 4-rank Gloo execution gate still passes after merge, including sparse-wave collective rounds.

Runtime wiring now present in code:
- host/bootstrap instantiates `CanonicalPlanPublisher`, `RuntimeExecutionPipeline`, and passive `RuntimeInstrumentation`;
- lifecycle caches canonical `PublishedPlan` authority from the M1 publication path;
- target-phase preparation calls `RuntimeExecutionPipeline.prepare(...)`;
- transport activation carries `PreparedExecution`;
- payload execution can call `RuntimeExecutionPipeline.execute(...)` and emit typed measurement events.

Still blocked:
- cross-module cancellation/failure/passive-evidence tests are still missing on this merged baseline;
- M3 runtime side-effect assertions have not yet been rerun through the merged host/runtime path;
- store completion/failure propagation still needs explicit integration assertions.
