# AR Answer Request

## Context

- Project: RouterSense distributed MoE scheduling system.
- Core POC1 claim: joint scheduling (`U_*`, especially `U_gated_maxweight_matching_atomic`) beats independent scheduling (`B_*`, especially `B_birkhoff`) in offline synthetic/trace-driven evaluation.
- Current real-system problem: in real 2-node distributed inference, scheduled transport has not beaten `native_baseline` on throughput.
- Real deployment setting:
  - `2 nodes x 1 GPU`
  - model: `allenai/OLMoE-1B-7B-0924-Instruct`
  - `64 samples`
  - `world_size=2`
  - `distributed_control_plane=true`
- Current best real scheduled result from [report.md](/root/RouterSense/RS/report.md:1):
  - `birkhoff + wave`: `5.3877 samples/s`, `2.7490 ms` mean scheduled comm
- Current native baseline from [report.md](/root/RouterSense/RS/report.md:1):
  - `native_baseline`: `5.7405 samples/s`, `3.4101 ms` mean native comm
- Therefore, scheduled comm can be lower than native comm, but end-to-end throughput is still worse.
- Relevant code:
  - scheduler injection entry: [runner.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/adapter/runner.py:1)
  - wave execution path: [wave_executor.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_executor.py:1)
  - wave materialization path: [wave_planner.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_planner.py:1)
- Relevant POC1 offline summary:
  - [olmoe_n8_s64_summary.json](/root/RouterSense/archive/backup/20260703_multimodel_u_scheduler_snapshot/artifacts/olmoe_n8_s64_summary.json:1)
  - Key offline numbers from that file:
    - `B_birkhoff_improvement_pct.mean = 33.47%`
    - `B_birkhoff_wave_improvement_pct.mean = 51.96%`
    - `U_gated_maxweight_matching_atomic_improvement_pct.mean = 59.14%`
    - `U_gated_maxweight_matching_improvement_pct.mean = 63.81%`

## Q1: Overhead Breakdown

### Background

- From [report.md](/root/RouterSense/RS/report.md:1), `birkhoff + wave` has:
  - `scheduled_comm_ms.mean = 2.7490`
  - `native_comm_ms.mean = 2.0342`
  - `samples_per_second = 5.3877`
- `native_baseline` has:
  - `native_comm_ms.mean = 3.4101`
  - `samples_per_second = 5.7405`
- In [wave_executor.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_executor.py:1), each wave does `pack -> all_to_all_single -> unpack`, and unpack currently includes `.clone()`.

### Request

- Quantify, per sample and per rank, the time contribution of:
  - GPU pack/unpack
  - CPU pack/unpack bookkeeping
  - NCCL launch/startup overhead per `all_to_all_single`
  - actual bulk communication time
  - control-plane time before communication starts
  - planner solve time
- Use the existing timing fields in `WaveTimingRecord` and `control_plane_ms` where possible.
- State which component most likely explains why lower communication time does not translate into higher throughput.

## Q2: What Exactly Is `control_plane_ms`?

### Background

- In [runner.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/adapter/runner.py:1), `control_plane_ms` is reported as:
  - `matrix_build_ms`
  - `all_reduce_ms`
  - `planner_ms`
  - `wave_convert_ms`
  - `conservation_check_ms`
- But [report.md](/root/RouterSense/RS/report.md:1) shows even `native_baseline` has `control_plane_ms.mean = 6.3912`, despite not using scheduled transport.

### Request

- Enumerate the exact code path that contributes to `control_plane_ms` for:
  - `native_baseline`
  - `scheduled_transport`
- Explain why `native_baseline` still pays nontrivial control-plane cost.
- Estimate how much of `control_plane_ms` is measurement artifact versus real deployment overhead.

## Q3: Theoretical Ceiling in a 2x2 Traffic Matrix

### Background

- POC1 gains were measured on larger traffic structures such as the 8-GPU OLMoE summary in [olmoe_n8_s64_summary.json](/root/RouterSense/archive/backup/20260703_multimodel_u_scheduler_snapshot/artifacts/olmoe_n8_s64_summary.json:1).
- The current real deployment is only `2 GPUs`, so the traffic matrix is effectively `2x2`.

### Request

- For a `2x2` dispatch/combine traffic matrix, derive the theoretical maximum communication improvement over naive one-shot `all_to_all_single`.
- Determine whether `Birkhoff` is already optimal in `2x2`.
- Determine whether cross-phase joint scheduling can still produce additional gain in `2x2`, or whether the search space has essentially collapsed.
- Give either a short proof or a concrete counterexample.

## Q4: Wave Count and NCCL Call Count

### Background

- In [wave_executor.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_executor.py:1), `CollectiveWaveExecutor` issues one `dist.all_to_all_single(...)` per wave.
- In `ScheduledAllToAllTransport(split_into_micro_ops=True)`, each scheduled op can become its own micro-wave.

### Request

- For the current `64-sample / 2-GPU / OLMoE-1B` runs, compute:
  - mean `dispatch_wave_count`
  - mean `combine_wave_count`
  - mean total NCCL calls per sample
- Estimate the mean bytes transferred per wave.
- Assuming a fixed NCCL startup cost `alpha = 0.5 ms` or `1.0 ms`, estimate the startup-only overhead contributed by multiple waves.
- Compare this startup-only estimate against the observed gap between scheduled and native throughput.

## Q5: Code-Level Optimization Headroom

### Background

- Current execution path in [wave_executor.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_executor.py:1) includes:
  - `_pack_wave_tensor(...)`
  - per-wave `all_to_all_single`
  - unpack via `view(...).clone()`
  - concatenation of `received_parts`
- Current schedule materialization in [wave_planner.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_planner.py:1) builds explicit `WaveSpec` objects with per-pair route item lists.

### Request

- Evaluate whether the main bottleneck is:
  - extra NCCL launches
  - pack/unpack memory movement
  - object/materialization overhead
  - synchronization behavior
- Propose concrete optimizations that preserve semantics:
  - avoid `.clone()`
  - fuse pack buffers
  - preallocate receive buffers
  - reduce Python-side per-wave work
  - exploit CUDA graphs or stream overlap
- For each proposed optimization, estimate an approximate `ms` saving per sample.
- State whether these optimizations alone are likely enough to let scheduled transport beat native baseline at `2 nodes x 1 GPU`.

## Q6: Minimal Experiment Needed to Recover POC-Style Gains

### Background

- Real deployment currently uses `OLMoE-1B-7B-0924-Instruct`, `2 nodes`, and `64 samples`.
- POC1 offline gains in [olmoe_n8_s64_summary.json](/root/RouterSense/archive/backup/20260703_multimodel_u_scheduler_snapshot/artifacts/olmoe_n8_s64_summary.json:1) were much larger:
  - `U_gated_maxweight_matching_atomic_improvement_pct.mean = 59.14%`
  - `B_birkhoff_improvement_pct.mean = 33.47%`

### Request

- Recommend the smallest realistic experiment configuration that gives joint scheduling a fair chance to show measurable real-system benefit:
  - batch size
  - model scale
  - number of GPUs
  - number of routed experts or communication-heavy layers
- If `2 GPUs` is fundamentally too small, state the minimum plausible GPU count.
- Also answer whether communication surface can be amplified without changing the model family, for example by increasing:
  - `hidden_size`
  - routed token count
  - top-k experts
  - number of active MoE layers

## Q7: How Should the POC1 Makespan Model Be Corrected?

### Background

- The POC1 improvement metrics in [olmoe_n8_s64_summary.json](/root/RouterSense/archive/backup/20260703_multimodel_u_scheduler_snapshot/artifacts/olmoe_n8_s64_summary.json:1) are based on an idealized makespan-style communication model.
- The real runtime in [wave_executor.py](/root/RouterSense/RS/src/rs/runtime/distributed_ep/core/wave_executor.py:1) includes non-ideal terms:
  - fixed launch overhead per collective
  - pack/unpack work per wave
  - runtime object/materialization overhead

### Request

- Rewrite the abstract cost model so that each wave cost includes at least:
  - `alpha_ms`: fixed NCCL launch/startup overhead
  - `beta_ms`: pack/unpack or runtime overhead per wave
  - `gamma * bytes`: size-dependent communication term
- Show how this corrected model changes the ranking criterion between:
  - joint scheduling
  - phase-local Birkhoff
  - native one-shot all-to-all
- Using the current real numbers in [report.md](/root/RouterSense/RS/report.md:1), infer rough ranges for `alpha_ms` and `beta_ms`.
- State whether the corrected model still supports the original thesis that joint scheduling is better, and under what scale assumptions that thesis becomes observable again.

## Output Requirements

- Please answer each question separately.
- Please keep answers quantitative whenever possible.
- Please cite which source file each number or assumption came from.
- Please distinguish clearly between:
  - facts already measured in this repo
  - theoretical inference
  - speculation about unmeasured bottlenecks
