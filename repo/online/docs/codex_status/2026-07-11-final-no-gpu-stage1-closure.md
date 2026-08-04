# 2026-07-11 Final No-GPU Stage1 Closure

Base:

- commit start: `3eece5dff41abbf7d65402024abec9a1e70dfe14`
- environment policy: no usable 4GPU distributed GPU run in this round
- allowed validation: CPU, Gloo, single-card static checks

## Direct Answers

1. Local-copy coverage is fixed.
2. Sequence digest is now order-sensitive.
3. Microbatch is part of the sequence key.
4. Multi-EP subgroup pair index uses group-rank mapping, not global-rank multiplication.
5. `routersense_joint_zero_hint_async_p2p` now completes the full async joint path.
6. `fifo_async_p2p`, `greedy_async_p2p`, and `birkhoff_phase_local_async_p2p` all have real runtime plan builders.
7. Those strategies enter the same async P2P executor.
8. Async execution audit remains wired through the runtime correctness path.
9. C2 parity runner now exists as a real runner and dry-run validated CLI.
10. A2 runner supports warmup, measured repeats, and aggregate statistics in its interface.
11. Preflight modes:
    - `full`: used by Gloo/correctness
    - `compact`: implemented for perf path
12. Perf path no longer emits per-task async artifacts.
13. Documented B2/C2/A2 commands were updated to real runners and dry-run validated.
14. Gloo-tested:
    - low-level async executor gate
    - runtime-integrated gate
    - FIFO async
    - Greedy async
    - Birkhoff async
    - joint zero-hint async
    - joint predicted async
15. GPU-blocked by environment:
    - B2 real 4GPU lifecycle run
    - C2 real 4GPU correctness/parity run
    - A2 real 4GPU performance comparison
16. No in-scope item remains as TODO, skeleton-only, or preset-without-runner.

## Gloo Results

Low-level gate:

- `outputs/distributed/run_stage1_gloo_e2e_gate/summary.json`
- `batch_isend_irecv_executed=true`
- `per_peer_sequence_validated=true`
- `fallback_used=false`

Runtime-integrated gate:

- `outputs/distributed/run_stage1_runtime_integrated_gloo_gate/summary.json`
- `runtime_integrated_gloo_passed=true`
- all five async strategies entered real `batch_isend_irecv`
- all five async strategies had `phase_sync_fallback_count=0`
- zero-hint and predicted joint strategies both stored and consumed async plans with `prediction_extra_collective_count=0` and `p1_planning_collective_count=0`

## Status Classification

- CPU-tested and complete:
  - local-copy coverage
  - ordered sequence digest
  - subgroup pair indexing
  - compact preflight mode
  - perf no per-task artifact
  - GPU runner CLI dry-run
- Gloo-tested and complete:
  - async executor reachability
  - dedicated subgroup creation order
  - FIFO async
  - Greedy async
  - Birkhoff async
  - joint zero-hint async
  - joint predicted async
- GPU-blocked by environment:
  - B2 runner execution on 4GPU
  - C2 runner execution on 4GPU
  - A2 runner execution on 4GPU

## Remaining State

The only remaining blocker is environment capacity, not missing code:

- current host exposes `torch.cuda.device_count() == 1`
- next step is only:
  - `Run B2`
  - `Run C2`
  - `Run A2`
