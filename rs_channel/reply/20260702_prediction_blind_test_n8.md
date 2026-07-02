# N=8 Prediction-Blind Test

Date: 2026-07-02

Artifacts:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8/summary.json`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8/table.md`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8/table_predicted.md`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8/table_zeros.md`

Validation:
- `pytest -q` → `43 passed`

Run:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -u experiments/poc_line1/exp_pairwise_candidate_compare.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_batch500_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_batch500_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_batch500_v2/gate_weights.pt \
  --placement round_robin \
  --num-gpus 8 \
  --topk 8 \
  --sample-limit 32 \
  --next-mode both \
  --skip-oracle \
  --output-dir artifacts/poc_line1/prediction_blind_test_n8
```

Blind-test table:

| algorithm | declared_aware | predicted_imp% | zeros_imp% | delta% | actual_aware | match |
|---|:---:|---:|---:|---:|:---:|:---:|
| birkhoff | no | 20.0539 | 46.1389 | 26.0850 | yes | no |
| ibbr | no | 22.1657 | 48.4479 | 26.2822 | yes | no |
| ejection_chain_tabu | no | 24.9718 | 49.3464 | 24.3745 | yes | no |
| lns_cp_repair | no | 24.9718 | 49.4568 | 24.4849 | yes | no |
| quantized_decomposed | no | -52.4058 | -3.1042 | 49.3015 | yes | no |
| cp_lpt | yes | 0.2431 | 30.0379 | 29.7949 | yes | yes |
| lagrangian | yes | 22.2583 | 48.2538 | 25.9955 | yes | yes |

Key conclusion:
- This experiment does **not** isolate “does the algorithm use phase-2 information to guide phase-0/1 ordering”.
- `next_mode=zeros` currently removes phase-2 traffic from the objective itself, so total makespan becomes much easier for every strategy.
- Because of that, most nominally prediction-blind algorithms also show large deltas and are flagged as `actual_aware`.

Interpretation:
- `cp_lpt` and `lagrangian` remain correctly declared as prediction-aware.
- The mismatches on `birkhoff`, `ibbr`, `ejection_chain_tabu`, `lns_cp_repair`, and `quantized_decomposed` are caused by the test definition, not by hidden phase-2 guidance inside those heuristics.
- In other words, the current blind test is measuring:
  - “Does phase-2 exist in the scheduling objective?”
  rather than:
  - “Does the algorithm exploit phase-2 values to improve phase-0/1 decisions?”

Code changes:
- `src/routesense/evaluation/analysis.py`
  - added `next_mode`
  - records `next_mode` in summary
  - supports zero-masked phase-2 input
- `experiments/poc_line1/exp_pairwise_candidate_compare.py`
  - added `--next-mode predicted|zeros|both`
  - writes predicted/zeros summaries plus blind-test comparison
- `src/routesense/scheduler/strategy.py`
  - added `prediction_aware` / `description` metadata interface
  - added `get_strategy_metadata()`
- `src/routesense/scheduler/strategies.py`
  - registered per-strategy metadata
- `src/routesense/scheduler/fast.py`
  - annotated key scheduler docstrings with prediction-awareness notes

Recommended next step:
- Add a stricter blind test that keeps phase-2 traffic in the final objective but hides it from phase-0/1 priority construction only. That would separate “objective coupling” from “true predictive guidance”.
