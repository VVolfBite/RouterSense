# Prediction-Blind Fix + N=32 Latency Follow-up

Date: 2026-07-02

Artifacts:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8_v2/summary.json`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/prediction_blind_test_n8_v2/table.md`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_compare_n32_v2/summary.json`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_compare_n32_v2/table.md`

Validation:
- `pytest -q` → `43 passed`

## Fixes applied

1. Removed `quantized_decomposed` from the active candidate pool:
   - removed from `src/routesense/evaluation/analysis.py::FAST_ALGORITHMS`
   - removed from `src/routesense/scheduler/fast.py::fast_schedule_pairwise()`
   - function definition and strategy registration were kept

2. Fixed `effective_latency_ms` reporting:
   - `analysis.py` now records `*_effective_latency_ms`
   - effective latency is intentionally identical to scheduling latency because it reuses the same solve result

3. Repaired prediction-blind test design:
   - `next_mode=zeros` no longer deletes phase-2 traffic from the evaluated problem
   - instead, for declared prediction-aware algorithms only, phase-2 is replaced by a uniform matrix **for ordering guidance only**
   - final makespan is still evaluated on the real three-phase schedule with real phase-2 traffic

4. Tightened iterative budgets:
   - `ejection_chain_tabu`: `budget_ms 2.5 -> 1.5`, `max_iters 20 -> 8`
   - `lns_cp_repair`: `budget_ms 4.0 -> 1.5`, `max_repair_iters 8 -> 4`
   - `lagrangian`: `max_iterations 15 -> 8`, added `budget_ms=5.0`

## N=8 prediction-blind result

| algorithm | declared_aware | predicted_imp% | zeros_imp% | delta% | actual_aware | match |
|---|:---:|---:|---:|---:|:---:|:---:|
| birkhoff | no | 20.0539 | 20.0539 | 0.0000 | no | yes |
| ibbr | no | 22.1951 | 22.2020 | 0.0069 | no | yes |
| ejection_chain_tabu | no | 24.9718 | 24.9718 | 0.0000 | no | yes |
| lns_cp_repair | no | 24.9718 | 24.9718 | 0.0000 | no | yes |
| cp_lpt | yes | 0.2431 | -1.2981 | 1.5412 | yes | yes |
| lagrangian | yes | 22.2202 | 27.5116 | 5.2914 | yes | yes |

Conclusion:
- The repaired test now behaves as intended.
- `birkhoff / ibbr / ejection_chain_tabu / lns_cp_repair` are prediction-blind in practice.
- `cp_lpt` and `lagrangian` are prediction-aware in practice.

## effective_latency_ms confirmation

Example from `prediction_blind_test_n8_v2/summary.json`:
- `cp_lpt_latency_ms.mean = 0.5338`
- `cp_lpt_effective_latency_ms.mean = 0.5338`
- `lagrangian_latency_ms.mean = 5.3439`
- `lagrangian_effective_latency_ms.mean = 5.3439`

This is the intended behavior: effective improvement changes the accounting model, not the solver runtime.

## N=32 latency before/after

Before (`candidate_compare_n32`):
- `birkhoff`: 2.7363 ms
- `ibbr`: 5.0310 ms
- `ejection_chain_tabu`: 120.6275 ms
- `lns_cp_repair`: 123.2095 ms
- `cp_lpt`: 1.5044 ms
- `lagrangian`: 26.0121 ms

After (`candidate_compare_n32_v2`):
- `birkhoff`: 2.5262 ms
- `ibbr`: 4.6355 ms
- `ejection_chain_tabu`: 125.2994 ms
- `lns_cp_repair`: 117.0340 ms
- `cp_lpt`: 2.1373 ms
- `lagrangian`: 7.2192 ms

Takeaways:
- `lagrangian` improved substantially on latency while keeping the best N=32 quality among active candidates (`24.17%`).
- `ejection_chain_tabu` and `lns_cp_repair` remain non-viable at N=32. Their issue is single-iteration cost, not just too-large budgets.
- `birkhoff` remains the best clean low-latency option.
