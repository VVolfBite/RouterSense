# RouteSense Report

## Current Conclusion

- The current distributed mainline in `RS/` supports `trace_replay`, not a
  real EP runtime.
- The current 2-rank environment is valid for wiring, correctness protocol
  bring-up, and collective calibration.
- The current 2-rank environment is not valid for production EP performance
  claims, joint-scheduling benefit claims, or online prediction claims.

## Scope Correction

The old wording around `native_baseline`, `wave_collective`, and
`scheduled_transport` overstated what the code was doing.

The current mainline now uses:

- `execution_mode=trace_replay|real_ep`
- `transport_execution_mode=unscheduled_collective_replay`
- `transport_execution_mode=wave_collective_replay`
- `transport_execution_mode=scheduled_collective_partition_replay`

And the current code explicitly rejects:

- `execution_mode=real_ep`

Current result JSON now carries:

- `execution_mode=trace_replay`
- `claim_scope=transport_replay_only`
- `is_real_ep_runtime=false`
- `uses_oracle_future_trace=true`
- `baseline_semantics=unscheduled_collective_replay|scheduled_collective_replay`
- `correctness_status=not_checked|passed|failed|unsupported`

## What The Current Mainline Actually Proves

It proves:

- the trace-derived dispatch plan can be materialized into distributed replay
  execution
- scheduled and unscheduled collective replay paths can both be invoked through
  the same bridge
- the benchmark/reporting path can now say when correctness was not checked,
  instead of fabricating a pass

It does not yet prove:

- real EP token ownership semantics
- real EP model sharding semantics
- online prediction benefit
- offline makespan gains turning into NCCL wall-clock gains
- fair production-native versus scheduled runtime performance

## Important Terminology Corrections

The repository now treats the following older terms as obsolete for the formal
mainline:

- `native_baseline`
  replaced by `unscheduled_collective_replay`
- `scheduled_transport`
  replaced by `scheduled_collective_partition_replay`
- `wave_collective`
  replaced by `wave_collective_replay`
- `physically_sharded_experts`
  replaced by `rank_local_expert_weight_cache_from_full_model`

The last rename matters because the current adapter still derives rank-local
expert weights from a full model load. That is not physical sharding.

## Benchmark Interpretation Boundary

Even with the naming cleanup, current trace replay still has hard limits:

- `future_trace` is oracle lookahead, not prediction
- `trace_replay` is not a real EP runtime
- a collective replay backend is not the same thing as realizing endpoint
  matching on the wire
- current 2-rank experiments do not have the topology needed to validate the
  core multi-matching argument from the offline PoC line

So any current benchmark should be described as:

- transport replay benchmark
- replay correctness protocol benchmark
- collective calibration benchmark

Not as:

- production EP benchmark
- scheduler speedup benchmark
- online serving benchmark

## Validation Semantics

Current output semantics are now stricter:

- when validation is disabled, the result reports
  `correctness_status=not_checked`
- the harness no longer reports a fake correctness pass when no validation was
  executed
- batch summaries now tolerate `not_checked` results instead of crashing during
  error aggregation

This is still not the final correctness design. The remaining work is to
replace weak replay checks with full route-identity, ownership, completeness,
and numerical validation.

## Current Allowed Claims

Until the remaining semantic issues are fixed, the mainline should limit itself
to the following claims:

- the distributed replay wiring is live
- current transport modes can be invoked and audited explicitly
- the benchmark/output schema now exposes replay-only scope instead of implying
  a real EP runtime

## Current Disallowed Claims

The mainline should not currently claim:

- real EP runtime support
- native EP baseline support
- online prediction benefit
- production NCCL speedup from current offline schedule results
- fair end-to-end throughput superiority of scheduled transport on the current
  2-rank setup
