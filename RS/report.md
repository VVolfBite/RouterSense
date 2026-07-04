# RouteSense Report

## Current Conclusion

The mainline now has an auditable three-lane split:

- `offline`
- `online`
- `legacy`

The only verified online-adjacent execution capability is now:

- `world_size=1`
- `OLMoE`
- single-layer local-MoE reconstruction parity
- `world_size=2`
- rank-local route construction
- local/remote partition
- distributed metadata/count agreement

This means the code can capture router logits and MoE layer inputs/outputs from
an already completed HuggingFace forward, rebuild the local expert
contributions with copied weights, and verify numerical agreement for the
tested prompt and layer.

It can also, on two real distributed ranks, build route records from the
current layer router output using `source_rank = dist.get_rank()`, partition
them into local and remote sends, compute a stable expert placement, and verify
send/recv count agreement plus manifest/placement/request-protocol consistency.

It does not yet constitute native EP runtime execution, native A2A transport,
distributed expert residency, hidden-state dispatch/combine, remote expert
compute, distributed numerical parity, or transport-calibrated online
observation.

## What Is True Right Now

### `legacy`

The old distributed replay harness is now explicitly:

- `pipeline=legacy`
- `execution_mode=legacy_trace_replay`
- `trace_origin=legacy_trace_replay`

It is not the formal online EP runtime.

### `offline`

The formal offline lane supports:

- router prediction collection
- oracle/full-trace analysis inputs
- provenance-gated calibrated-analysis inputs

It still does not implement the calibrated simulator itself.

### `online`

The formal online lane now supports a verified single-rank local-MoE
reconstruction harness:

- route identity built from request/microbatch/layer/local token/top-k slot
- explicit local route preservation
- copied local expert weights extracted from a real OLMoE checkpoint
- local expert reconstruction and top-k combine
- numerical parity against the captured HuggingFace OLMoE `mlp(...)` output

It does not yet support:

- `world_size > 1` native A2A dispatch/combine
- hidden-state transport
- remote expert compute
- inverse combine
- distributed numerical correctness
- scheduled P2P transport
- deployable online prediction
- transport-calibrated observation for offline fitting

It now also supports a narrow `world_size=2` metadata-only distributed stage:

- real rank-local prompts
- `source_rank = dist.get_rank()`
- explicit local/remote route partition
- stable expert placement hash
- rank manifest hash
- tensor-based send/recv count agreement
- truthful `TransportOperationRecord` with `phase=count_exchange`

That stage does not move hidden states and must not be presented as native A2A
execution.

## Single-Rank Verified Result

Environment used:

- machine: local workstation
- GPU: `NVIDIA GeForce RTX 4080`
- model path: `D:\Project\Test\OLMoE`
- runtime path: `experiments/online/bench_native_ep.py`

Command:

```bash
python experiments/online/bench_native_ep.py \
  --world-size 1 \
  --model-path D:\Project\Test\OLMoE \
  --prompt "Explain MoE routing briefly." \
  --layer-index 0 \
  --precision fp16 \
  --device-index 0 \
  --output-dir artifacts/online/bench_native_ep_smoke
```

Observed result from the current truthful envelope:

- `execution_mode = world_size_1_local_moe_reconstruction_parity`
- `trace_origin = observed_single_rank_local_moe`
- `is_real_ep_runtime = false`
- `claim_scope = correctness_and_calibration_only`
- `performance_claim_eligible = false`
- `correctness_status = passed`
- `numerical_correctness_pass = true`
- `max_abs_error = 0.0001220703125`
- `mean_abs_error = 2.5033950805664062e-06`
- `route_count = 56`
- `local_route_count = 56`
- `remote_route_count = 0`

Interpretation:

- the single-rank reconstruction path is numerically aligned with the real
  OLMoE MoE block for the tested layer and prompt
- this validates local route build, copied-weight local expert reconstruction,
  and top-k combine semantics for `world_size=1`
- it does not validate distributed transport, multi-rank ownership, or A2A

## Single-Rank Local-MoE Observation

Command:

```bash
python experiments/online/collect_native_ep_trace.py \
  --world-size 1 \
  --model-path D:\Project\Test\OLMoE \
  --prompt "Explain MoE routing briefly." \
  --layer-index 0 \
  --precision fp16 \
  --device-index 0 \
  --output-dir artifacts/online/native_ep_trace_smoke
```

Observed result from the current truthful envelope:

- `execution_mode = world_size_1_local_moe_reconstruction_observation`
- `trace_origin = observed_single_rank_local_moe`
- `is_real_ep_runtime = false`
- `is_real_ep_transport = false`
- `is_transport_calibration_trace = false`
- `correctness_status = passed`
- trace artifacts were written:
  - `artifacts/online/native_ep_trace_smoke/<run_id>.jsonl`
  - `artifacts/online/native_ep_trace_smoke/<run_id>_metadata.json`
  - `artifacts/online/native_ep_trace_smoke/<run_id>_summary.json`

This observation is intentionally not accepted by the calibrated offline gate:

```bash
python experiments/offline/fit_ep_cost_model.py \
  --trace-metadata artifacts/online/native_ep_trace_smoke/<run_id>_metadata.json
```

Result:

- rejected unless the artifact is a real multi-rank
  `trace_origin = observed_online_native_ep` trace with:
  - `world_size >= 2`
  - `is_real_ep_runtime = true`
  - `transport_backend = online_native_a2a_ep`
  - remote routes
  - stage timings
  - expert bucket records
  - schema version 2 trace events in the JSONL artifact

## WS=2 Route Partition And Count Agreement

New supported command shape:

```bash
torchrun --nproc_per_node=2 experiments/online/bench_native_ep.py \
  --world-size 2 \
  --model-path <MODEL_PATH> \
  --prompt-rank0 "<PROMPT_0>" \
  --prompt-rank1 "<PROMPT_1>" \
  --layer-index 0 \
  --precision fp16 \
  --route-partition-only \
  --validate-metadata \
  --output-dir artifacts/online/bench_native_ep_ws2
```

And trace export:

```bash
torchrun --nproc_per_node=2 experiments/online/collect_native_ep_trace.py \
  --world-size 2 \
  --model-path <MODEL_PATH> \
  --prompt-rank0 "<PROMPT_0>" \
  --prompt-rank1 "<PROMPT_1>" \
  --layer-index 0 \
  --precision fp16 \
  --route-partition-only \
  --validate-metadata \
  --output-dir artifacts/online/native_ep_trace_ws2
```

This stage verifies:

- each rank owns a different local prompt
- `source_rank` comes from `dist.get_rank()`
- the current layer router builds all top-k routes for local hidden states
- routes split into `local_routes` and `remote_send_routes`
- local routes are preserved
- remote destination ranks match the fixed placement
- distributed send/recv row counts agree
- run id, layer id, request protocol, placement hash, and manifest hashes are
  checked via tensor collectives
- a truthful metadata-only transport record is emitted

This stage is exported as:

- `execution_mode = online_ws2_route_partition_only`
- `trace_origin = observed_online_ws2_route_partition`
- `claim_scope = distributed_route_partition_and_count_agreement_only`
- `is_real_ep_runtime = false`
- `is_real_ep_transport = false`
- `is_transport_calibration_trace = false`
- `correctness_status = metadata_passed|metadata_failed|not_checked`

This stage still does not provide:

- hidden-state A2A dispatch
- remote expert execution
- inverse combine
- distributed MoE numerical parity
- offline calibrated scheduling input

## Compatibility Fixes Applied

One real implementation issue surfaced during local validation:

- the local OLMoE checkpoint stores experts as `ModuleList`
- the earlier code assumed packed `gate_up_proj/down_proj`

This is now fixed:

- `extract_local_expert_weights()` accepts both packed and `ModuleList` expert
  layouts
- the online feature probe now truthfully reports which layout is present

## What Can Be Claimed

Allowed now:

- the repo has a real offline/online/legacy boundary
- legacy replay cannot masquerade as online EP
- offline calibrated analysis no longer accepts single-rank local-MoE
  reconstruction artifacts as native EP observation
- offline calibrated analysis also rejects the ws2 partition-only trace
- online scheduler hint mode rejects `oracle_full_trace`
- single-rank OLMoE local-MoE reconstruction parity is verified
- single-rank local-MoE observation export is verified
- `world_size=2` rank-local route construction, local/remote partition, and
  distributed count agreement are verified

Not allowed now:

- multi-rank online native EP performance
- actual A2A dispatch
- distributed online numerical correctness
- scheduled transport speedup
- matching-realized runtime benefit
- calibrated offline milliseconds as measured deployment time
- deployable prediction benefit
- treating `world_size=1` local-MoE observation as real DEP data
- treating `observed_online_ws2_route_partition` as transport calibration input

## Current Runnable Commands

Runnable and meaningful:

- `python experiments/offline/exp_router_prediction.py ...`
- `python experiments/online/bench_native_ep.py --world-size 1 ...`
- `python experiments/online/collect_native_ep_trace.py --world-size 1 ...`
- `torchrun --nproc_per_node=2 experiments/online/bench_native_ep.py --world-size 2 --route-partition-only ...`
- `torchrun --nproc_per_node=2 experiments/online/collect_native_ep_trace.py --world-size 2 --route-partition-only ...`
- `python experiments/legacy/exp_trace_replay.py ...`

Present, but either gated or still not implemented:

- `python experiments/offline/fit_ep_cost_model.py --trace-metadata ...`
  - only for future real multi-rank native EP traces; current single-rank local
    observation and ws2 partition-only observation are both rejected
- `python experiments/offline/exp_calibrated_schedule.py ...`
- `python experiments/online/bench_native_ep.py --world-size > 1 ...`
  - except the metadata-only ws2 `--route-partition-only` path
- `python experiments/online/bench_scheduled_ep.py ...`

## Tests Run

Targeted regressions:

- `python -m pytest RS/tests/test_online_native_runtime.py RS/tests/test_pipeline_boundaries.py RS/tests/test_scheduled_execution_bridge.py RS/tests/test_wave_execution_planner.py RS/tests/test_distributed_ep_scaffold.py -q`
  - `35 passed`
- `python -m pytest RS/tests/test_structure_refactor.py RS/tests/test_online_expert_store.py RS/tests/test_online_native_runtime.py -q`
  - `7 passed`
- `python -m pytest RS/tests/test_online_expert_store.py RS/tests/test_online_native_runtime.py RS/tests/test_pipeline_boundaries.py -q`
  - `13 passed`

Focused package/boundary suite:

- `python -m pytest RS/tests/test_pipeline_boundaries.py RS/tests/test_structure_refactor.py RS/tests/test_scheduled_execution_bridge.py RS/tests/test_wave_execution_planner.py RS/tests/test_distributed_ep_scaffold.py RS/tests/test_package_source_only.py -q`
  - `33 passed, 4 skipped`

The `4 skipped` are Windows-side archive-shell prerequisites, not semantic
test failures.

## Immediate Next Step

The next real milestone is:

- keep the current ws2 route-partition/count-agreement stage truthful
- add hidden tensor dispatch after count agreement
- then add owner-rank expert compute
- then add inverse combine
- then add distributed numerical correctness

That is the sequence required to cross from auditable ws2 metadata agreement
into a real multi-rank EP runtime.
