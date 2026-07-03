# RouteSense Report

## Current Status

- Real 2-node distributed execution is working on the current PPIO setup.
- Scheduler strategy and transport execution granularity are now decoupled.
- `transport_granularity` is now an explicit runtime / experiment dimension:
  - `wave`: one `all_to_all_single` per wave
  - `atomic`: one `all_to_all_single` per transfer op
- This is no longer hardcoded by strategy name.

## N15 Delivered

Implemented the `reply.md` request:

- `execute_scheduled_inference(..., transport_granularity="wave")`
- `exp_wave_execution.py` now exposes:
  - `--transport-granularity wave|atomic`
- result JSON `run` block now records:
  - `"transport_granularity": "..."`

This makes comparisons fair:

- `birkhoff + wave`
- `birkhoff + atomic`
- `U_gated_maxweight_matching_atomic + wave`
- `U_gated_maxweight_matching_atomic + atomic`

are all valid and no longer depend on hidden strategy-specific transport behavior.

## Latest Validation Runs

### 1. Birkhoff + Wave

- Result file: `/tmp/rs_wave_runs/20260703T193719Z-birkhoff-scheduled_transport/result.json`
- `transport_granularity = "wave"`
- Sample count: `2`
- Throughput: `1.3454 samples/s`
- Token throughput: `30.2722 tokens/s`
- Mean scheduled comm: `181.4011 ms`
- Mean native comm: `0.9943 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

### 2. Birkhoff + Atomic

- Result file: `/tmp/rs_wave_runs/20260703T193757Z-birkhoff-scheduled_transport/result.json`
- `transport_granularity = "atomic"`
- Sample count: `2`
- Throughput: `1.5966 samples/s`
- Token throughput: `35.9236 tokens/s`
- Mean scheduled comm: `135.4313 ms`
- Mean native comm: `1.1327 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

## Prior 64-Sample Real Results

### U_gated_maxweight_matching_atomic + Wave

- Result file: `/tmp/rs_wave_runs/20260703T191314Z-U_gated_maxweight_matching_atomic-scheduled_transport/result.json`
- Throughput: `5.2200 samples/s`
- Token throughput: `167.1227 tokens/s`
- Mean planner cost: `0.2726 ms`
- Mean control-plane cost: `0.3907 ms`
- Mean scheduled comm: `10.4739 ms`
- P50 scheduled comm: `2.2999 ms`
- P95 scheduled comm: `26.1253 ms`
- Mean native comm: `2.1243 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

### Birkhoff + Wave

- Result file: `/tmp/rs_wave_runs/20260703T190241Z-birkhoff-scheduled_transport/result.json`
- Throughput: `5.2212 samples/s`
- Token throughput: `167.1591 tokens/s`
- Mean planner cost: `0.2429 ms`
- Mean control-plane cost: `0.3542 ms`
- Mean scheduled comm: `11.6446 ms`
- P50 scheduled comm: `2.5364 ms`
- P95 scheduled comm: `50.6964 ms`
- Mean native comm: `1.9114 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

## Interpretation

- The previous hardcoded strategy-name switch was a real fairness bug in evaluation.
- That bug is now fixed.
- Any future comparison between scheduling algorithms must explicitly state transport granularity.
- The 64-sample result that previously made atomic look much worse was largely confounded by runtime fragmentation.
- After the earlier wave-level runtime change, `U_gated_maxweight_matching_atomic + wave` became competitive with `birkhoff + wave`.
- Now that granularity is explicit, the next step is to benchmark the full matrix on the same workload instead of attributing runtime-path differences to algorithm quality.

## Qwen Download Status

- Target model: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/root/model-cache/Qwen1.5-MoE-A2.7B`
- Remote path: `/vllm-workspace/models/Qwen1.5-MoE-A2.7B`
- Local model is now complete.
- Remote model currently has metadata plus shards `00001`-`00006`.
- Remote still needs:
  - `model-00007-of-00008.safetensors`
  - `model-00008-of-00008.safetensors`

## Next Immediate Actions

- Finish remote backfill for Qwen shards `00007` and `00008`.
- Run the full fair benchmark matrix:
  - `U_gated_maxweight_matching_atomic × {wave, atomic}`
  - `birkhoff × {wave, atomic}`
  - optionally `greedy × {wave, atomic}`
- Only then compare algorithm quality under matched execution granularity.
