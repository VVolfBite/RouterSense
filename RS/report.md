# RouteSense Report

## Current Conclusion

- The old N17 `2 nodes x 1 GPU` throughput table is **not** a valid end-to-end `native` vs `scheduled` performance comparison.
- It **does** prove one important thing:
  - RouteSense can inject scheduled transport into the real OLMoE EP path and pass correctness checks.
- It does **not** prove:
  - `scheduled` communication is already faster than `native`
  - control-plane overhead is the dominant cause of the throughput gap
  - joint scheduling has been fairly tested at a scale where it should beat `native`

## Why The Old N17 Benchmark Was Invalid

The old harness mixed several different costs into the scheduled timed path:

1. trace collection
2. scheduled dispatch / expert / combine execution
3. a full native reference replay inside the same scheduled sample
4. correctness comparison

That means the old scheduled `batch_wall_ms` was not measuring a production scheduled path.

There was a second fairness issue:

- `native_baseline` still ran matrix construction, matrix aggregation, and scheduler preamble before entering the native transport branch.

So the old comparison was polluted on both sides:

- scheduled was too expensive because it included validation replay
- native was too expensive because it still paid scheduler-side setup

## Code Fixes Applied In This Round

### 1. Native baseline now short-circuits scheduler preamble

File:

- [runner.py](D:/Project/Test/RouterSense/RS/src/rs/runtime/distributed_ep/adapter/runner.py:126)

What changed:

- `execution_mode=native_baseline` now returns before matrix build, all-reduce, planner solve, and wave conversion.
- Native control-plane fields are now zeroed except optional self-check timing.

Why:

- This makes the benchmarked native branch much closer to an actual one-shot transport baseline.

### 2. Validation replay is now optional

Files:

- [runner.py](D:/Project/Test/RouterSense/RS/src/rs/runtime/distributed_ep/adapter/runner.py:143)
- [exp_wave_execution.py](D:/Project/Test/RouterSense/RS/experiments/distributed/exp_wave_execution.py:171)

What changed:

- Added `verify_correctness` plumbing in the runner.
- Added CLI flag:
  - `--validation off|sampled|always`
- Added:
  - `--validation-every N`

Default:

- `--validation off`

Why:

- Scheduled transport no longer pays for a full native replay inside the timed benchmark path unless explicitly requested.

### 3. Trace time is now outside benchmark throughput time

File:

- [exp_wave_execution.py](D:/Project/Test/RouterSense/RS/experiments/distributed/exp_wave_execution.py:224)

What changed:

- `trace_ms` is still recorded per sample.
- `batch_wall_ms` now accumulates only the cross-rank critical-path sample execution time used for benchmark throughput.
- Validation time is subtracted from `sample_wall_ms`.

Why:

- The benchmark now measures control-plane + transport + local expert execution, not trace extraction.

### 4. Batch timing now uses cross-rank critical path

File:

- [exp_wave_execution.py](D:/Project/Test/RouterSense/RS/experiments/distributed/exp_wave_execution.py:312)

What changed:

- Per-sample batch accounting now uses the max `sample_wall_ms` across gathered rank payloads.

Why:

- Collective runtime is determined by the slowest participating rank, not by rank 0 local wall time.

## Current Theoretical Position

For the current `2 x 1 GPU` setup, the user-supplied reasoning is correct in substance:

- the cross-rank nontrivial traffic is effectively a `2 x 2` exchange
- native already executes one `all_to_all_single` per phase
- without real communication-compute overlap, wave splitting does not create new matching freedom
- extra waves mainly introduce launch, pack/unpack, allocation, and synchronization overhead

So the right expectation is:

- this environment is suitable for correctness and calibration
- it is not the right environment to expect joint scheduling to beat native on throughput

## What The Old N17 Result Should Now Be Called

The most accurate statement is:

- N17 verified real scheduled-transport injection and correctness under a real 2-node OLMoE execution chain.
- N17 did **not** produce a fair native-vs-scheduled throughput comparison.
- N17 did **not** test joint scheduling at a scale where the PoC mechanism should be expected to win.

## New Benchmark Controls

Current recommended benchmark modes:

1. performance mode
   - `--validation off`
2. periodic regression mode
   - `--validation sampled --validation-every 64`
3. correctness mode
   - `--validation always`

## Validation Status

Local regression run completed after this change:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest RS\tests\test_scheduled_execution_bridge.py -q`
- result: `8 passed`

Coverage added in this round:

- native baseline skips scheduler preamble
- scheduled execution skips native replay when validation is off

## What Still Needs To Be Done

These changes fix benchmark boundaries, but they do **not** yet solve the main runtime bottlenecks.

Still pending:

1. export per-wave diagnostics:
   - wave count
   - bytes
   - pack/unpack timings
   - rank-critical timing
2. remove avoidable executor materialization overhead:
   - `.clone()`
   - repeated `torch.cat`
   - Python row packing
3. rerun a clean calibration benchmark on `2 x 1 GPU`
4. move to at least `4 GPUs`, and preferably `8 GPUs`, for a meaningful joint-scheduling performance test

## Immediate Next Step

Do **not** use the old N17 table as evidence that scheduled communication is already faster than native.

The next valid step is:

1. rerun the benchmark with the new timing boundaries
2. persist per-wave execution details
3. treat `2 x 1 GPU` only as a calibration environment
4. promote the real performance experiment to `N >= 4`, ideally `N = 8`
