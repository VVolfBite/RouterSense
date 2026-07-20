# Distributed Failure Protocol

Distributed invariant failures must be rank-consistent.

Current protocol:

1. Each rank evaluates local invariant state.
2. At safe synchronization points, ranks call `distributed_invariant_gate(...)`.
3. The gate exchanges a fixed-size integer tensor over Gloo / the active process group.
4. If any rank reports failure, all ranks raise `RouterSenseInvariantError`.
5. Every rank writes a local failure artifact before exiting.

This milestone adds two low-memory fault-injection cases:

- `rank1_preflight_failure`
- `rank0_planning_digest_failure`

Acceptance:

- all ranks exit without deadlock;
- all ranks record the same primary error code;
- no rank waits indefinitely for a peer that already failed.
