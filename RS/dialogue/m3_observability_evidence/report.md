M3 branch now has real evidence/result contracts instead of protocol-only stubs.

Closed in this branch:
- `ResultBundle` fail-closed validation with typed deserialize.
- `ReferenceTraceBundle` typed roundtrip with validation.
- `FilesystemArtifactWriter` path confinement and atomic replace.
- `NullMeasurementSink` / `PerfLightMeasurementSink` / `BufferedDebugProbe` bounded behavior.
- `RuntimeInstrumentation` as the single integration adapter surface.

Still blocked:
- latest Runtime/M2 pipeline is not yet wired on this branch baseline, so formal passive-evidence integration tests still belong to `convergence/m123-integration`.
- old `dialogue/status.json` at repo root is historical noise and not authoritative for this branch.
