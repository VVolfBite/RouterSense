# RouteSense Legacy PoC1

This legacy PoC1 tree is reserved for real single-route ablation experiments on OLMoE.

## Purpose

PoC1 tests whether routing context can predict realized criticality measured by true counterfactual expert ablation:

1. baseline forward
2. zero one selected expert route weight for one token
3. rerun forward without rerouting or renormalization
4. measure `delta_nll`

## Current Status

- CPU-side scaffolding is in place.
- Real GPU validation is still required for the intervention path.
- The next hardware-backed gate is a single-GPU smoke on `allenai/OLMoE-1B-7B-0924`.
- The smoke config now supports a local prompt file so the main path does not depend on HuggingFace dataset availability.

## Recommended GPU Smoke Order

```bash
cd /root/autodl-tmp/RouterSense
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli doctor --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_doctor
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli inspect-model --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_inspect
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli collect-trace --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_trace
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli ablate --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_ablate
```

Only after the smoke ablation succeeds should the `poc.yaml` configuration be attempted.


## Next Recommended Configs

Long-context and multi-domain follow-up configs are now prepared:

```bash
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli run-all --config legacy/poc1/configs/code_long.yaml --output-dir legacy/poc1/outputs/code_long_0924
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli run-all --config legacy/poc1/configs/math_long.yaml --output-dir legacy/poc1/outputs/math_long_0924
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli run-all --config legacy/poc1/configs/dialog_long.yaml --output-dir legacy/poc1/outputs/dialog_long_0924
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli run-all --config legacy/poc1/configs/all_layers_code.yaml --output-dir legacy/poc1/outputs/all_layers_code_0924
```

These configs are intended for the next GPU-backed refinement round:
- long context (`context_length=256`)
- multi-domain comparison (`code`, `math`, `dialog`)
- all-layer sampling (`num_moe_layers=16`)


## Latest Result: `code_long_0924`

The first long-context code-domain run completed on real GPU with:
- `32` windows
- `4` sampled MoE layers
- `top_k=8`
- model `allenai/OLMoE-1B-7B-0924`

Primary artifact locations:
- live output: `legacy/poc1/outputs/code_long_0924/`
- committed archive snapshot: `archive/poc1-code-long-0924-20260629T181333Z/`

Current factual findings:
- pairwise ranking accuracy = `0.4633`
- random mean `delta_nll` = `0.000758`
- raw routing mean `delta_nll` = `-0.001098`
- oracle mean `delta_nll` = `-0.066552`

Interpretation:
- raw routing is still only marginally better than random on the weak objective of selecting one relatively safe route to cancel
- routing context still fails to recover realized criticality ranking reliably
- oracle remains much better than observable routing features, so the safe-deferral search space exists even though the current signal does not recover it

State-like vs routing-like result in this run:
- best `routing_only` feature family = `topk_rank`
- best `state_like` feature family = `topk_rank`
- best `combined` family = `effective_gate_weight + topk_rank + layer_id`, but it is worse than `topk_rank` alone
- calibration still does not produce positive test `R^2`

This means the current experiment has not separated a strong state signal from a strong routing signal. Both collapse to the same weak ordering effect.

Useful subgroup diagnostics from this run:
- `layer 0` pairwise accuracy = `0.5024`
- `layer 15` pairwise accuracy = `0.4173`
- high-entropy groups pairwise accuracy = `0.4889`
- low-entropy groups pairwise accuracy = `0.4387`
- oracle top-32 hard subset pairwise accuracy = `0.4312`
- oracle top-32 mean `delta_nll` = `-0.2378`
- random top-32 mean `delta_nll` = `-0.0148`

The last comparison is important: hard subsets still contain large recoverable value, but the current routing context factors do not recover that value.


## Batch Runner

For the next GPU-backed refinement round, a simple batch runner is available:

```bash
bash legacy/poc1/scripts/run_refine_batch.sh
```

It runs:
- code_long
- math_long
- dialog_long
- all_layers_code
