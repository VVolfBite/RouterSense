# RouteSense Report

## Current Status

- Real 2-node distributed OLMoE execution is working on the current PPIO setup.
- Scheduler injection into the real cross-node EP transport is working.
- `transport_granularity` is now a real runtime axis:
  - `wave`: one `all_to_all_single` per wave
  - `atomic`: one `all_to_all_single` per transfer op
- `distributed_control_plane=true` is working after the matrix aggregation fix.
- `greedy` now also works in real distributed runs even when the strategy returns only a scalar makespan and no explicit schedule.

## Critical Fixes In This Round

### 1. Distributed control-plane aggregation fix

Previous bug:

- each rank built a full global matrix from the same full trace
- `all_reduce` then summed already-global matrices again
- real failure:
  - `wave schedule over-consumed pair (0, 1) ... requested 156, available 78`

Fix:

- non-distributed path still uses the full global matrix
- distributed path now builds only the local rank contribution matrix before `all_reduce`

Relevant commits:

- `b8dba1c` `Fix distributed control-plane matrix aggregation`
- `d46669d` `Document distributed control-plane fix`

### 2. Fallback schedule bridge fix

Previous bug:

- `execute_scheduled_inference()` computed a fallback schedule for strategies like `greedy`
- but wave conversion still consumed the original empty `result.schedule`

Fix:

- wave conversion now receives the resolved effective schedule

Related commit:

- `a469e20` `Fix fallback schedule wave conversion`

### 3. Wave-planner empty-phase fallback

Previous bug:

- some scalar-only strategies could still reach wave conversion with empty per-phase entries
- real failure:
  - `RuntimeError: incomplete wave allocation for phase 0: {(0, 1): 78, (1, 0): 81}`

Fix:

- if a phase has no schedule entries, `wave_planner` now synthesizes a one-wave default schedule directly from `DispatchPlan`

Status:

- targeted local regression checks passed
- real `greedy + wave` run now passes on 2 nodes

## N17 Real 2-Node 64-Sample Matrix

### Control Variables

- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Prompt file: `artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- Sample limit: `64`
- Layer index: `0`
- World size: `2`
- `distributed_control_plane = true`

### Result Table

| Strategy | Mode | Granularity | samples/s | mean scheduled comm (ms) | mean native comm (ms) | planner (ms) | control plane (ms) | correctness |
|---|---|---|---:|---:|---:|---:|---:|---|
| `U_gated_maxweight_matching_atomic` | `scheduled_transport` | `wave` | `5.3618` | `3.3961` | `2.0510` | `0.2686` | `8.2602` | `pass` |
| `U_gated_maxweight_matching_atomic` | `scheduled_transport` | `atomic` | `5.2359` | `4.0536` | `1.9581` | `0.2716` | `9.8080` | `pass` |
| `birkhoff` | `scheduled_transport` | `wave` | `5.3877` | `2.7490` | `2.0342` | `0.2456` | `7.9196` | `pass` |
| `birkhoff` | `scheduled_transport` | `atomic` | `5.3432` | `3.8574` | `1.9559` | `0.2512` | `8.4458` | `pass` |
| `greedy` | `scheduled_transport` | `wave` | `5.0007` | `3.2339` | `2.1309` | `0.0771` | `13.5766` | `pass` |
| `greedy` | `scheduled_transport` | `atomic` | `5.3247` | `3.3229` | `2.0681` | `0.0802` | `7.2513` | `pass` |
| `greedy` | `native_baseline` | `wave` | `5.7405` | `0.0000` | `3.4101` | `0.0770` | `6.3912` | `pass` |

## Interpretation

### Main conclusion

- Real distributed deployment is now working end-to-end.
- Scheduler injection is correct.
- On this current setup, scheduled transport does **not** beat native baseline on throughput.

### What changed after the control-plane fix

Before the fix, scheduled comm looked like roughly `8-11 ms`.

After the fix, scheduled comm dropped to roughly:

- `2.75-3.40 ms` for `wave`
- `3.32-4.05 ms` for `atomic`

This means the earlier matrix was not trustworthy for runtime performance comparison.

### What the current data says

- Best scheduled throughput in this matrix:
  - `birkhoff + wave` at `5.3877 samples/s`
- Best overall throughput:
  - `native_baseline` at `5.7405 samples/s`
- Best scheduled mean communication:
  - `birkhoff + wave` at `2.7490 ms`
- `greedy` is now fully operational, but it is not outperforming the stronger schedulers.

### Practical interpretation

- We have succeeded at the first real objective:
  - verify that this environment can actually do distributed EP inference with custom scheduling logic
- We have **not** yet shown a performance win over native transport at 2 nodes / OLMoE-1B / 64 samples.
- Current evidence still supports the earlier hypothesis:
  - this scale is probably too small for POC1-style schedule quality gains to dominate NCCL launch cost, packing overhead, and runtime noise

## Result Files

- `/tmp/rs_fair_benchmark_cp/U_gated_maxweight_matching_atomic_wave/result.json`
- `/tmp/rs_fair_benchmark_cp/U_gated_maxweight_matching_atomic_atomic/result.json`
- `/tmp/rs_fair_benchmark_cp/birkhoff_wave/result.json`
- `/tmp/rs_fair_benchmark_cp/birkhoff_atomic/result.json`
- `/tmp/rs_fair_benchmark_cp/greedy_wave/result.json`
- `/tmp/rs_fair_benchmark_cp/greedy_atomic/result.json`
- `/tmp/rs_fair_benchmark_cp/native_baseline/result.json`

## Qwen Download Status

- Target model: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/root/model-cache/Qwen1.5-MoE-A2.7B`
- Remote path: `/vllm-workspace/models/Qwen1.5-MoE-A2.7B`
- Local model is complete.
- Remote model is still incomplete.
- Missing remote shards:
  - `model-00007-of-00008.safetensors`
  - `model-00008-of-00008.safetensors`

## Recommended Next Steps

- Run the same matrix at larger scale:
  - `sample_limit = 128`
  - `sample_limit = 256`
- Keep `native_baseline` in every batch-size matrix.
- After that, decide whether to:
  - stay on OLMoE for scaling experiments
  - or switch to a larger MoE model where the communication surface is bigger.
