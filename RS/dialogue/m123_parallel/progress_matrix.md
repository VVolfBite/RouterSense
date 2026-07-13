| 模块 | 当前状态 | 已完成且有证据 | 已实现未验证 | 当前阻塞 | 跨模块依赖 | 下一入口 |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | READY | formal prediction/planning contracts and CPU regression on `452f9c6` | none | none | M1/M2/M3 must preserve M0 regressions | carry forward regression suite |
| M1 | BLOCKED | lifecycle no main-thread prediction; submit status handling; keyed queue; hard store transitions; attach/close invariants | dedicated deterministic publication lane absent | no ControlCommunicationLane poll protocol; no 4-rank delayed/failed/cancelled proof; no formal no-late-suffix dynamic proof | M2 published/materialized interfaces; M3 sinks must remain passive | freeze publication slot and lane contracts |
| M2 | PLANNED | interface ownership and audit scope identified | none | no unified PublishedPlan/MaterializedPlan/Executor chain yet | consumes M1 publication contracts; exposes execution contracts to M1 integration | freeze execution contracts and path map |
| M3 | PLANNED | interface ownership and audit scope identified | none | no separated checks/measurement/debug/evidence path yet | attaches to M1/M2 only as sink/probe/writer | freeze checks/measurement/result/trace contracts |
| M4 | PLANNED | none | none | depends on stabilized M0-M3 surfaces | M0-M3 | after M123 integration |
| M5 | PLANNED | none | none | depends on M0-M4 evidence quality and experiment plumbing | M0-M4 | after M4 |
