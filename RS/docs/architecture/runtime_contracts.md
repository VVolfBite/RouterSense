## Runtime Contracts

### Online Megatron EP runtime

Formal package:

- `rs.runtime.online.megatron_ep`

Canonical responsibilities:

- `host.py`
  - distributed init
  - model load
  - dispatcher discovery
  - hook install / restore
- `lifecycle.py`
  - P0 / P1 before/after hooks
  - transport activation / clear
  - scheduling-policy handoff
- `artifact_recorder.py`
  - record-only artifact output
- `phase/`
  - `PhaseReadyContext`
  - `OutgoingSegment`
  - `IncomingSlot`
  - `TransferLayout`
  - phase validation
- `control/`
  - root-authoritative agreement
  - mailbox / state machine / timeline
- `execution/`
  - bucketization
  - transport adapter
  - sync wave executor
  - layout validation
- `policy_adapter.py`
  - online runtime to scheduling boundary
- `p2_provider.py`
  - allowed P2 hint modes only

Frozen semantic boundaries:

- P0/P1 hook timing
- TransferLayout meaning
- P0 hidden + routing_probs atomic bundle
- P1 bundle contract
- MegatronPhaseTransportAdapter semantics
- sync_wave_executor collective semantics
- root-authoritative plan agreement semantics

### Offline runtime

Formal package:

- `rs.runtime.offline`

Canonical responsibilities:

- trace collection / loading
- traffic matrix construction
- predictor / calibration artifact handling
- offline runner orchestration

### Scheduling boundary

Formal package:

- `rs.scheduling`

Scheduling owns:

- logical flow model
- matching
- scoring
- baseline / reference policies
- logical plan objects

Scheduling must not own:

- torch tensors
- NCCL collectives
- Megatron dispatcher access
- artifact IO

### Remaining contract debt

- `rs.scheduling.policy.agreement` still contains distributed runtime control logic
- `experiments/offline` still uses historical POC entrypoints rather than direct formal runtime/scheduling composition
