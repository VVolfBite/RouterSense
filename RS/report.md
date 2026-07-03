# RouteSense Report

## Current Status

- Real 2-node distributed execution is working on the current PPIO setup.
- Scheduler-injected transport over real NCCL `all_to_all_single` is running correctly on real hardware.
- `U_gated_maxweight_matching_atomic` runtime execution has been changed from chunk-split micro-steps to wave-level execution.
- That runtime change materially improved real throughput and communication cost.

## Latest Real 64-Sample Results

- Prompt set: `artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- Sample count: `64`
- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Execution mode: `scheduled_transport`
- Environment: `2 nodes x 1 GPU`

### U_gated_maxweight_matching_atomic

- Result file: `/tmp/rs_wave_runs/20260703T190733Z-U_gated_maxweight_matching_atomic-scheduled_transport/result.json`
- Throughput: `5.2200 samples/s`
- Token throughput: `167.1227 tokens/s`
- Mean planner cost: `0.2726 ms`
- Mean control-plane cost: `0.3907 ms`
- Mean scheduled comm: `10.4739 ms`
- P50 scheduled comm: `2.2999 ms`
- P95 scheduled comm: `26.1253 ms`
- Mean native comm baseline: `2.1243 ms`
- Mean communication delta vs native: `-8.3496 ms`
- Mean communication ratio vs native: `5.9427x`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

### Birkhoff

- Result file: `/tmp/rs_wave_runs/20260703T190241Z-birkhoff-scheduled_transport/result.json`
- Throughput: `5.2212 samples/s`
- Token throughput: `167.1591 tokens/s`
- Mean planner cost: `0.2429 ms`
- Mean control-plane cost: `0.3542 ms`
- Mean scheduled comm: `11.6446 ms`
- P50 scheduled comm: `2.5364 ms`
- P95 scheduled comm: `50.6964 ms`
- Mean native comm baseline: `1.9114 ms`
- Mean communication delta vs native: `-9.7332 ms`
- Mean communication ratio vs native: `8.3019x`
- Correctness: `token/gate conservation pass`, `max_abs_error = 0.0`

## Delta From Previous Atomic Runtime

- Previous chunk-split atomic result file:
  `/tmp/rs_wave_runs/20260703T190143Z-U_gated_maxweight_matching_atomic-scheduled_transport/result.json`
- Previous throughput: `3.7551 samples/s`
- Previous mean scheduled comm: `32.2009 ms`
- Previous P95 scheduled comm: `120.6917 ms`

### Improvement After Wave-Level Runtime

- Throughput improvement: about `1.39x`
- Mean scheduled communication reduction: about `67.5%`
- P95 scheduled communication reduction: about `78.4%`
- Mean control-plane cost reduction: about `17.7%`

## Interpretation

- The major regression was not the atomic scheduler logic itself.
- The main problem was runtime over-fragmentation: the scheduler emitted waves, but runtime re-split those waves into chunk-like micro-steps.
- After forcing `U_gated_maxweight_matching_atomic` to execute at wave granularity, its real throughput became effectively tied with `birkhoff`.
- `atomic` now slightly loses on planner cost, but slightly wins on mean and tail communication in this 64-sample run.
- The runtime path is now much closer to the intended scheduler semantics.

## Qwen Download Status

- Target model from prior reports: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/root/model-cache/Qwen1.5-MoE-A2.7B`
- Remote path: `/vllm-workspace/models/Qwen1.5-MoE-A2.7B`
- Local currently has `7/8` weight shards.
- Remote has partial shards and is being backfilled from local files.
- The `huggingface_hub + hf-mirror` path was unstable for this model in the current environment, so direct file backfill is the safer route.

## Next Immediate Actions

- Complete the remote Qwen model backfill.
- Re-run the same real benchmark on additional scheduler variants now that atomic runtime semantics are fixed.
- If needed, further reduce runtime overhead by coalescing pack/unpack work across waves without reintroducing chunk-split execution.
