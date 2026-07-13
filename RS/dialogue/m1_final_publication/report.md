Current M1 publication track checkpoint:

- Introduced a real `GlooControlCommunicationLane` owner in `runtime/online/megatron_ep/control/communication_lane.py`.
- The lane now defines deterministic slot polling semantics around `PublicationSlot`, gathered status, terminal short-circuit (`FAILED`, `CANCELLED`, `EXPIRED`), and root-canonical result selection.

Still open before `M1_RUNTIME_LIFECYCLE_READY`:

1. lifecycle publication pump is not yet wired through the lane;
2. root canonical non-root decode is not yet connected to `TargetLayerPreparedJointPlan` publication;
3. 4-rank delayed / failed / cancelled Gloo gate is not yet implemented;
4. formal no-late-suffix dynamic proof is still absent.
