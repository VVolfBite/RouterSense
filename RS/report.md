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

This means the code can capture router logits and MoE layer inputs/outputs from
an already completed HuggingFace forward, rebuild the local expert
contributions with copied weights, and verify numerical agreement for the
tested prompt and layer.

It does not yet constitute native EP runtime execution, native A2A transport,
distributed expert residency, or transport-calibrated online observation.

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
- real remote-route transport
- online distributed correctness
- scheduled P2P transport
- deployable online prediction
- transport-calibrated observation for offline fitting

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
- online scheduler hint mode rejects `oracle_full_trace`
- single-rank OLMoE local-MoE reconstruction parity is verified
- single-rank local-MoE observation export is verified

Not allowed now:

- multi-rank online native EP performance
- distributed online correctness
- scheduled transport speedup
- matching-realized runtime benefit
- calibrated offline milliseconds as measured deployment time
- deployable prediction benefit
- treating `world_size=1` local-MoE observation as real DEP data

## Current Runnable Commands

Runnable and meaningful:

- `python experiments/offline/exp_router_prediction.py ...`
- `python experiments/online/bench_native_ep.py --world-size 1 ...`
- `python experiments/online/collect_native_ep_trace.py --world-size 1 ...`
- `python experiments/legacy/exp_trace_replay.py ...`

Present, but either gated or still not implemented:

- `python experiments/offline/fit_ep_cost_model.py --trace-metadata ...`
  - only for future real multi-rank native EP traces; current single-rank local
    observation is rejected
- `python experiments/offline/exp_calibrated_schedule.py ...`
- `python experiments/online/bench_native_ep.py --world-size > 1 ...`
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

The next real milestone is no longer another rename or metadata pass.

It is:

- `world_size=2` native online A2A metadata/count agreement
- then hidden tensor dispatch/combine
- then distributed correctness

That is the point where the project genuinely crosses from verified single-GPU
local-MoE reconstruction into real multi-rank EP runtime work.
