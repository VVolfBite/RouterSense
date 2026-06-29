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

## Recommended GPU Smoke Order

```bash
cd /root/autodl-tmp/RouterSense
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli doctor --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_doctor
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli inspect-model --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_inspect
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli collect-trace --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_trace
PYTHONPATH=legacy/poc1/src python -m routesense_poc1.cli ablate --config legacy/poc1/configs/smoke.yaml --output-dir legacy/poc1/outputs/smoke_ablate
```

Only after the smoke ablation succeeds should the `poc.yaml` configuration be attempted.
