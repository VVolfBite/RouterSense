# 2026-07-11 B2 GPU Result

- Commit: `9796d72a47f47c559daf795d349e1bfa3d728592`
- Host: `autodl-container-43df4dbbb8-b31320c7`
- Git status at report generation: `M experiments/distributed/run_gpu_b2_lifecycle.py;  M src/rs/runtime/online/megatron_ep/lifecycle.py`
- Result: `B2_PASSED`

## B2-A Zero-Hint

- Status: `passed`
- planning_traffic_source: `pre_transport_phase_ready_context`
- actual_p0_total_rows: `1048576`
- p1_is_exact_transpose: `True`
- stored/consumed digest match: `True`
- async batch_isend_irecv calls: `12`
- real send ops: `36`
- real recv ops: `36`
- fallback count: `0`

## B2-B Nonzero Predictor

- Status: `passed`
- planning_traffic_source: `pre_transport_phase_ready_context`
- predictor: `copy_current_dispatch`
- prediction_confidence: `1.0`
- prediction source->target: `0 -> 1`
- prediction relative_l1_error: `0.1805419921875`
- prediction cosine_similarity: `0.974753507728759`
- consumed during P0 joint planning: `True`
- stored/consumed digest match: `True`
- async batch_isend_irecv calls: `12`
- real send ops: `36`
- real recv ops: `36`
- fallback count: `0`
