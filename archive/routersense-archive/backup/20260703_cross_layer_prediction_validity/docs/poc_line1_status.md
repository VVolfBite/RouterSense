# POC-line1 Status

## Scope

POC-line1 validates the **traffic-prediction** line after the project explicitly abandoned the earlier assumption that routing weight implies realized criticality.

This line does **not** use the old POC1 importance claim. It asks two different questions:

1. Are adjacent MoE layers correlated enough that future dispatch demand is partially predictable?
2. If we can see several layers of traffic ahead, is there a non-trivial scheduling gap between per-layer greedy and multi-layer oracle planning?

## Implemented

The current RS mainline now includes:

- full-sequence router trace collection for all discovered OLMoE MoE layers
- cross-layer overlap analysis
- batch-level rank-correlation analysis
- traffic-matrix construction for same-prompt batches
- round-robin and skewed expert placement builders
- offline oracle-vs-greedy scheduling simulation
- CPU regression tests for the full offline chain

Main modules:

- `RS/src/routesense/trace/olmoe_router_trace.py`
- `RS/src/routesense/evaluation/poc_line1.py`
- `RS/experiments/poc_line1/full_sequence_trace.py`
- `RS/experiments/poc_line1/cross_layer_analysis.py`
- `RS/experiments/poc_line1/oracle_vs_greedy.py`

## Current Boundary

This is **not yet** the final hardware-backed result.

What is ready now:

- code path
- artifact layout
- offline metrics
- CPU tests

What still needs single-GPU validation:

- real `full_sequence_trace` collection on OLMoE-1B-7B-0924-Instruct
- real prompt bundle trace export
- real Gate 1 analysis on collected trace
- real Gate 2 analysis from real trace-derived traffic matrices

## Recommended Next Commands

Single-GPU trace collection:

```bash
cd /root/autodl-tmp/RouterSense/RS
python experiments/poc_line1/full_sequence_trace.py \
  --model-id allenai/OLMoE-1B-7B-0924-Instruct \
  --model-path /root/autodl-tmp/models/OLMoE-1B-7B-0924 \
  --text "The history of science is a story of observation, argument, and revision." \
  --sample-id smoke-0 \
  --request-id smoke-0 \
  --output-dir artifacts/poc_line1/full_sequence_trace_smoke
```

Cross-layer analysis:

```bash
python experiments/poc_line1/cross_layer_analysis.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke/trace.jsonl \
  --output-dir artifacts/poc_line1/cross_layer_report_smoke
```

Offline oracle-vs-greedy:

```bash
python experiments/poc_line1/oracle_vs_greedy.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke/trace.jsonl \
  --placement round_robin \
  --output-dir artifacts/poc_line1/oracle_report_smoke
```

## Testing

Validated with:

```bash
cd /root/autodl-tmp/RouterSense/RS
pytest -q
```
