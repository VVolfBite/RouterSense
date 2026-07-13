# Runtime Static Audit

## Confirmed
- Async release submits each release batch with `batch_isend_irecv`, then waits over the returned work handles before advancing to the next release batch.
- Current correctness mode uses `commit_batch(limit=1)`, so many small release batches can inflate submission and wait wall time even when active transport is only a few milliseconds.
- P0 matrix collection, plan agreement, and some preflight paths use distributed collectives and CPU materialization. These are control-plane costs, not active P2P transport.
- Existing `communication_makespan_us` spans first transport submission to last transport completion. For selected P0/P1 this may include inter-phase gaps and compute/control work; it must not be called network busy time.

## Requires GPU Timeline
- Whether the 80-90 ms selected window is dominated by pre-submit control, many small release submissions, post-submit wait, inter-phase gap, or post-transport processing.
- Whether compact preflight eliminates hot-path P1 collectives in the measured 4GPU timeline.
- Whether small task/wave granularity correlates with high submit span or wait span.
- Which rank is critical for hook wall time and request wait.

## Excluded This Round
- No change to work wait policy.
- No task merging or buffer reuse.
- No barrier removal.
- No P2P launch order change.
- No pack/unpack algorithm change.

## Next Dynamic Focus
Inspect `submit_queue_us`, `submit_span_us`, `request_wait_us`, `post_transport_us`, `inter_phase_gap_us`, task size distribution, and rank imbalance for B-core and U-zero on the same 8x16 selected-layer run.
