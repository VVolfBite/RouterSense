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

Merged-baseline closure now additionally proves:
- prepared-target cancellation/cleared-store short-circuit does not enter `RuntimeExecutionPipeline.prepare(...)`;
- prepared-target `materialization_invalid -> FAILED` is asserted directly through `_try_prepared_target_plan_for_p0(...)`;
- `after_token_combine(...)` completes prepared-target execution through `TargetPlanStore.complete(...)` on the merged runtime path.

Current integration verdict:
- `M123_PARALLEL_INTEGRATION_READY`

Deferred, but no longer hard blockers for M123:
- richer passive evidence/result-bundle wiring beyond typed measurement events;
- broader merged-runtime scenario coverage once M4 parity consumes this baseline.
