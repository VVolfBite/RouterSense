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
