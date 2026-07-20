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
- `observation/artifact_recorder.py`
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
- `pending_window/policy_adapter.py`
  - prepared logical plan to current phase-plan compiler
- `control/p2_provider.py`
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

- `experiments/offline` still exposes narrow study-specific entrypoints rather than a single config-driven runner
- several oversized online runtime modules still need responsibility-based splitting before long-term freeze
