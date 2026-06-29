# POC1 Archive: local prompts on OLMoE-0924

## Scope

This archive captures the first real single-GPU POC1 ablation runs after switching from HF dataset loading to local prompt files.

Included runs:
- `smoke_inspect_0924`: real model MoE layer inspection on `allenai/OLMoE-1B-7B-0924`
- `manual_smoke_0924`: one manually verified real route ablation
- `local_prompt_trace_0924`: routing-context trace collection on local prompts
- `local_prompt_ablate_0924`: smoke-scale ablation run
- `local_prompt_poc_0924`: larger 32-window ablation run

## Data

Prompt sources:
- `theory_prompts_50.jsonl`: shorter prompt set used for early local smoke
- `theory_prompts_50_long.jsonl`: longer prompt set used for stable trace/ablation windows

## Model

- Model id: `allenai/OLMoE-1B-7B-0924`
- Local path: `/root/autodl-tmp/models/OLMoE-1B-7B-0924`
- Precision: fp16
- Hardware used for these runs: single 4090D

## Initial observations

Smoke run (`8 windows x 3 layers x top-k=8`):
- Pairwise ranking accuracy: `0.4939`
- Random mean delta_nll: `0.00881`
- Raw routing mean delta_nll: `-0.00893`

POC run (`32 windows x 4 layers x top-k=8`):
- Pairwise ranking accuracy: `0.4784`
- Random mean delta_nll: `0.00779`
- Raw routing mean delta_nll: `0.01180`
- Oracle mean delta_nll: `-0.07844`

Current interpretation:
- Raw routing weight does not yet behave like a strong realized-criticality ranking signal.
- It may still provide weak value as a safe-deferral heuristic in some groups, but that claim remains preliminary.
- Calibration has not yet been rerun on this local-prompt path.

## Notes

- This archive is intentionally committed under `archive/` for analysis convenience.
- The original raw runtime outputs remain under `legacy/poc1/outputs/` on the server but are not the preferred handoff location.


## Calibration update

A lightweight learned combination was added after the first archive snapshot.

Current result on `local_prompt_poc_0924`:
- selected feature subset: `['topk_rank']`
- validation MAE: `0.0330`
- test MAE: `0.0672`
- test R^2: `-0.0383`

Strategy-level effect after enabling `calibrated`:
- calibrated mean delta_nll: `-0.00879`
- calibrated median delta_nll: `-0.00171`
- calibrated p95 delta_nll: `0.19141`

Interpretation:
- the current learned combination does beat the zero baseline in mean terms,
  but it is not yet clearly strong or stable;
- the learned search did not discover a richer routing-context combination here,
  and instead fell back to `topk_rank` as the best small feature subset under document-held-out validation.


## Family-wise feature search update

Feature search is now split into three families:
- `routing_only`
- `state_like`
- `combined`

Current best subsets on `local_prompt_poc_0924`:
- routing_only: `['topk_rank']`
- state_like: `['topk_rank']`
- combined: `['effective_gate_weight', 'routing_entropy', 'topk_rank', 'layer_id']`

Validation comparison:
- routing_only validation MAE: `0.0330`
- state_like validation MAE: `0.0330`
- combined validation MAE: `0.0404`

Current interpretation:
- the present data does not show a clear advantage from adding more routing-context factors;
- both routing-only and state-like search collapse to the same strongest simple observable, `topk_rank`;
- combined features currently overfit relative to the simpler baseline.
