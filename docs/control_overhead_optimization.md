# Control Overhead Optimization

## Goal

The pre-GPU closure removes obvious hot-path control overhead without redesigning the whole runtime.

## Perf-Mode Rules

In `perf` profile, the runtime should not perform:

- hot-path `json.dumps`
- hot-path filesystem open/write/flush
- full plan `to_dict()`
- full-plan JSON hashing
- control timeline dict append
- per-task execution dict construction
- shadow-plan builds for diagnostics

Instead, `RuntimePerfCounters` aggregates fixed stage counters:

- count
- total time
- max time

## Key Runtime Changes

- `_timeline()` returns immediately in perf mode
- `_record_planning_timing()` updates counters instead of appending records in perf mode
- heartbeat file writes are suppressed in perf mode
- async executor perf path records only compact counts/timings
- runtime state keeps only bounded current/next-layer prediction/plan state

## What This Does Not Yet Prove

It proves the obvious Python/JSON/I/O hot-path leaks are removed from the main perf path.
It does not yet prove final GPU performance leadership; that requires 4GPU `Run A2`.
