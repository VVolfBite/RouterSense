Integration now contains the reviewed M1 lifecycle closure plus the latest M2 and M3 branch implementations.

Verified on this merged baseline:
- ancestry includes `44a4376` (M1), `a35d7c5` (M2), and `aa5a43e` (M3);
- integration compileall passes;
- focused M2/M3 unit tests pass;
- formal M2 4-rank Gloo execution gate still passes after merge.

Still blocked:
- the formal runtime path is not yet calling `RuntimeExecutionPipeline` from lifecycle/host;
- passive M3 sinks are not yet wired into runtime/execution events on this merged baseline;
- cross-module cancellation/failure/passive-evidence tests still need to be added and run.
