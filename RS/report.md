# RouteSense Report

## Current Status

- Real 2-node distributed execution is working on the current PPIO setup.
- The runtime path now supports scheduler-injected transport over real NCCL `all_to_all_single`.
- `U_gated_maxweight_matching_atomic` has been switched to wave-level execution in runtime, instead of re-splitting each scheduled wave into per-chunk micro-steps.

## Latest Real 64-Sample Results

- Prompt set: `artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- Sample count: `64`
- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Execution mode: `scheduled_transport`
- Environment: `2 nodes x 1 GPU`

### Birkhoff

- Result file: `/tmp/rs_wave_runs/20260703T190241Z-birkhoff-scheduled_transport/result.json`
- Throughput: `5.2212 samples/s`
- Token throughput: `167.1591 tokens/s`
- Mean planner cost: `0.2429 ms`
- Mean control-plane cost: `0.3542 ms`
- Mean scheduled comm: `11.6446 ms`
- P95 scheduled comm: `50.6964 ms`
- Mean native comm baseline: `1.9114 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

### U_gated_maxweight_matching_atomic

- Previous chunk-split result file: `/tmp/rs_wave_runs/20260703T190143Z-U_gated_maxweight_matching_atomic-scheduled_transport/result.json`
- Throughput: `3.7551 samples/s`
- Token throughput: `120.2214 tokens/s`
- Mean planner cost: `0.3280 ms`
- Mean control-plane cost: `0.4744 ms`
- Mean scheduled comm: `32.2009 ms`
- P95 scheduled comm: `120.6917 ms`
- Mean native comm baseline: `2.1877 ms`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

## Interpretation

- Real distributed scheduling injection is confirmed working.
- The main bottleneck is not solver time; it is transport execution overhead.
- Birkhoff currently outperforms the previous atomic runtime path by a large margin in real throughput and communication overhead.
- The atomic strategy was still being over-fragmented by runtime micro-step execution. That runtime behavior has now been changed to wave-level execution and needs to be re-benchmarked.

## Qwen Download Status

- Target model from prior reports: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/root/model-cache/Qwen1.5-MoE-A2.7B`
- Remote path: `/vllm-workspace/models/Qwen1.5-MoE-A2.7B`
- Local currently has `7/8` weight shards.
- Remote currently has partial shards and is being backfilled from local files.

## Next Immediate Actions

- Re-run `U_gated_maxweight_matching_atomic` after wave-level runtime change on the same 64-sample set.
- Compare it again against `birkhoff`.
- If communication is still worse than baseline, optimize wave coalescing / pack-unpack overhead next.
