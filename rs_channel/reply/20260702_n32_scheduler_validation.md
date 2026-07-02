# N=32 Scheduler Validation

Date: 2026-07-02

Artifacts:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_compare_n32/table.md`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_compare_n32/summary.json`

Code changes:
- Pruned the active fast candidate pool in `src/routesense/evaluation/analysis.py` and `src/routesense/scheduler/fast.py`.
- Fixed `decomposed` merge to preserve sender-group locality instead of round-robin interleave.
- Added unified iterative early-stop controls in `src/routesense/scheduler/fast.py`:
  - patience = 5
  - significant relative improvement threshold = 20%
- Added prediction-aware/effective metrics in `src/routesense/evaluation/analysis.py`.
- Fixed token normalization so `num_gpus > token_count` no longer collapses traffic matrices to all-zero.

Validation:
- `pytest -q` → `43 passed`

N=32 run:
```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -u experiments/poc_line1/exp_pairwise_candidate_compare.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_batch500_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_batch500_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_batch500_v2/gate_weights.pt \
  --placement round_robin \
  --num-gpus 32 \
  --sample-limit 8 \
  --skip-oracle \
  --algorithms "birkhoff,ibbr,ejection_chain_tabu,lns_cp_repair,quantized_decomposed,cp_lpt,lagrangian" \
  --output-dir artifacts/poc_line1/candidate_compare_n32
```

Mean results:

| algorithm | sched_improve% | effective_improve% | prediction_aware | latency_ms |
|---|---:|---:|:---:|---:|
| birkhoff | 22.9234 | 22.9234 | no | 2.7363 |
| ibbr | 22.9234 | 22.9234 | no | 5.0310 |
| ejection_chain_tabu | 16.3735 | 16.3735 | no | 120.6275 |
| lns_cp_repair | 16.3735 | 16.3735 | no | 123.2095 |
| quantized_decomposed | -73.3796 | -73.3796 | no | 214.6740 |
| cp_lpt | 0.3187 | 0.3187 | yes | 1.5044 |
| lagrangian | 24.6830 | 24.6830 | yes | 26.0121 |

Observations:
- `lagrangian` is the strongest N=32 candidate by quality, but misses the practical latency target badly.
- `birkhoff` remains the best latency/quality tradeoff at this scale.
- `ibbr` no longer improves over `birkhoff` at N=32.
- `ejection_chain_tabu` and `lns_cp_repair` degrade sharply in runtime and also lose quality versus their N=8 behavior.
- `quantized_decomposed` is still broken as a competitive strategy; quality is catastrophically negative.
- `cp_lpt` remains prediction-aware in mechanism, but its gain is near zero at N=32.

Interpretation:
- Optimization space does not disappear at N=32. `lagrangian` reaching 24.68% while `birkhoff` is 22.92% shows there is still cross-phase value.
- The main issue is scalability of the stronger search/repair heuristics, not the absence of scheduling opportunity.
- For deployment-oriented candidate selection today, `birkhoff` is still the only clean sub-5ms option in this tested set.
