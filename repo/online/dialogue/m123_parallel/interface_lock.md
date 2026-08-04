Interface lock scope:

- M1 contracts frozen in `rs.runtime.online.megatron_ep.public_types` and `rs.runtime.online.megatron_ep.control.communication_lane`.
- M2 contracts frozen in `rs.core.contracts.execution`.
- M3 contracts frozen in `rs.core.contracts.checks`, `measurement`, `debug`, `artifact`, plus `result` and `trace` extensions.

Frozen rules:

1. Publication ordering will be keyed by `PublicationSlot.semantic_digest()`, not local queue order.
2. Runtime publication collective ownership will belong only to `ControlCommunicationLane`.
3. Post-publication execution will pivot on `PublishedPlan -> MaterializedPlan -> ValidationResult -> ExecutionOutcome`.
4. Observability remains passive: checks, measurement, debug, trace, and artifact writers are protocols only and do not own runtime decisions.
5. Integration ownership remains:
   - M1 owns lifecycle/state/publication
   - M2 owns publication-to-execution
   - M3 owns sinks/evidence only
