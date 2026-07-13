Cross-module dependency summary:

- M1 -> M2:
  - `PublicationSlot`
  - `LocalPreparationToken`
  - `LocalPublicationCandidate`
  - `PublicationPollResult`
  - `PublishedPlan`

- M2 -> M1:
  - `TargetPlanState` terminal and claim/bind/executing semantics must remain single-owner in lifecycle/store.

- M1 -> M3:
  - `RuntimeEvent`
  - `RuntimeDecision`
  - `RuntimeIdentity`
  - publication and execution lifecycle checkpoints

- M2 -> M3:
  - `ValidationResult`
  - `ExecutionOutcome`
  - `MaterializedPlan` summaries

- M3 -> integration only:
  - `MeasurementSink`
  - `DebugProbe`
  - `TraceSink`
  - `ArtifactWriter`
  - `ResultBundle`
  - `ReferenceTraceBundle`
