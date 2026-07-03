# RouteSense Report

## Current Status

- Real 2-node distributed execution is working on the current PPIO setup.
- Scheduler strategy and transport execution granularity are now decoupled.
- `transport_granularity` is now an explicit runtime dimension:
  - `wave`: one `all_to_all_single` per wave
  - `atomic`: one `all_to_all_single` per transfer op
- The 64-sample fair benchmark matrix for `U_gated_maxweight_matching_atomic` and `birkhoff` has been completed.

## N16 Fair Benchmark Matrix

### Control Variables

- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Prompt file: `artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- Sample limit: `64`
- Layer index: `0`
- Execution mode: `scheduled_transport`
- World size: `2`

### Important Note

- `--distributed-control-plane` was requested in `reply.md`, but in the current implementation it double-counts already-global trace matrices and breaks correctness.
- Real failure reproduced:
  - `wave schedule over-consumed pair (0, 1) ... requested 156, available 78`
- Therefore the fair benchmark matrix below was run with:
  - `distributed_control_plane = false`
- This keeps all four runs comparable and correct under the current code path.

## Results Table

| Strategy + Granularity | samples/s | tokens/s | mean scheduled comm (ms) | P50 (ms) | P95 (ms) | mean native (ms) | planner (ms) | correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `U_gated_maxweight_matching_atomic + wave` | `5.3156` | `170.1835` | `8.1347` | `1.9395` | `17.0423` | `2.0841` | `0.2711` | `pass` |
| `U_gated_maxweight_matching_atomic + atomic` | `5.1954` | `166.3346` | `11.2786` | `2.8101` | `18.7167` | `1.9624` | `0.2714` | `pass` |
| `birkhoff + wave` | `5.2884` | `169.3106` | `10.0073` | `2.5518` | `18.3399` | `2.0717` | `0.2442` | `pass` |
| `birkhoff + atomic` | `5.3298` | `170.6366` | `9.2986` | `2.7351` | `12.0308` | `1.9136` | `0.2475` | `pass` |

## Result Files

- `U_gated_maxweight_matching_atomic + wave`
  - `/tmp/rs_fair_benchmark/U_gated_maxweight_matching_atomic_wave/result.json`
- `U_gated_maxweight_matching_atomic + atomic`
  - `/tmp/rs_fair_benchmark/U_gated_maxweight_matching_atomic_atomic/result.json`
- `birkhoff + wave`
  - `/tmp/rs_fair_benchmark/birkhoff_wave/result.json`
- `birkhoff + atomic`
  - `/tmp/rs_fair_benchmark/birkhoff_atomic/result.json`

## Interpretation

### Algorithm Quality Comparison

Use matched granularity only.

- `wave vs wave`
  - `atomic` has lower mean scheduled comm than `birkhoff`
    - `8.13 ms` vs `10.01 ms`
  - `atomic` also has lower P50 and P95 comm
    - P50: `1.94 ms` vs `2.55 ms`
    - P95: `17.04 ms` vs `18.34 ms`
  - Throughput is slightly higher for `atomic`
    - `5.3156` vs `5.2884 samples/s`
- `atomic vs atomic`
  - `birkhoff` has lower mean scheduled comm than `atomic`
    - `9.30 ms` vs `11.28 ms`
  - `birkhoff` has much lower P95 comm
    - `12.03 ms` vs `18.72 ms`
  - Throughput is slightly higher for `birkhoff`
    - `5.3298` vs `5.1954 samples/s`

Conclusion:
- Under `wave` execution, `U_gated_maxweight_matching_atomic` currently looks slightly better.
- Under `atomic` execution, `birkhoff` currently looks slightly better.
- So the algorithm ranking is not invariant to transport granularity, which means runtime execution shape is still materially interacting with scheduler quality.

### Granularity Cost Comparison

Use matched strategy only.

- `U_gated_maxweight_matching_atomic`
  - `wave` beats `atomic` on scheduled comm
    - `8.13 ms` vs `11.28 ms`
  - `wave` also wins on throughput
    - `5.3156` vs `5.1954 samples/s`
- `birkhoff`
  - `atomic` beats `wave` on scheduled comm
    - `9.30 ms` vs `10.01 ms`
  - `atomic` also slightly wins on throughput
    - `5.3298` vs `5.2884 samples/s`

Conclusion:
- Granularity cost is strategy-dependent.
- There is no universal “wave is always better” or “atomic is always better” conclusion from this matrix.

## Control-Variable Check

- Native comm means across all four runs:
  - `2.0841`, `1.9624`, `2.0717`, `1.9136`
- Range relative to minimum:
  - about `8.91%`

Conclusion:
- Native comm variation is below the `20%` acceptance threshold.
- Hardware/network control is stable enough for this benchmark set.

## Qwen Download Status

- Target model: `Qwen/Qwen1.5-MoE-A2.7B`
- Local path: `/root/model-cache/Qwen1.5-MoE-A2.7B`
- Remote path: `/vllm-workspace/models/Qwen1.5-MoE-A2.7B`
- Local model is complete.
- Remote model currently has metadata plus shards `00001`-`00006`.
- Remote still needs:
  - `model-00007-of-00008.safetensors`
  - `model-00008-of-00008.safetensors`

## Next Immediate Actions

- Fix `distributed_control_plane` semantics so it can be safely enabled for truly distributed matrix construction.
- Extend the same fair matrix to `greedy`.
- Finish remote Qwen shard backfill.
