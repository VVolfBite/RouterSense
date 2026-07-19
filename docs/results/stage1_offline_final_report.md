# Stage1 Offline Final Report

Source of truth:

- `outputs/offline/stage1_paper_closure_final/final_offline_summary.json`
- `outputs/offline/stage1_paper_closure_final/paper_ready_tables.md`
- `outputs/offline/stage1_paper_closure_final/fixture_manifest.json`

## Main results

- Strongest phase-sync-compatible baseline: `phase_barrier_fifo`
- Shared-core proxy joint-only gain:
  - `gated_greedy`: `14.91%` makespan reduction for raw U vs paired B
  - `barrier_criticality_matching`: `11.15%` makespan reduction for raw U vs paired B
- Main safe-U family result retained from replay suite:
  - `RS_safe_gated_greedy` relative improvement: `11.52%`
- Oracle gap:
  - `O_joint` vs `O_local`: `13.33%` makespan reduction on the small exact fixture
  - `B_gap_to_O_local`: `26.67%`
  - `raw_U_gap_to_O_joint`: `46.15%`
  - `safe_U_gap_to_O_joint`: `46.15%`

## Predictor result

- Final selected online-eligible predictor: `zero_hint`
- Selection rule: minimum validation schedule regret, then overhead, then matrix error
- Held-out test metrics for the selected predictor:
  - mean relative L1: `0.75`
  - schedule regret: `0.0`
- Held-out scheduling result:
  - selected predictor gain vs zero: `0.0`
  - oracle predictor gain vs zero: `-5.28%`
- Prediction failure taxonomy:
  - `PREDICTION_NEUTRAL`: `48`

## Interpretation

- The offline scheduling opportunity remains real; the shared-core proxy results still show positive joint gains.
- The current online-eligible predictor selection collapses to `zero_hint`, not because the predictor stack is missing, but because the held-out scheduling objective does not reward the non-oracle candidates.
- Even oracle traffic does not improve held-out scheduling under the current scheduler core summary; this points to scheduler P2 consumption weakness rather than “just make the predictor more accurate”.

## Reproducibility

- Replay config: `configs/offline/stage1_paper_closure_final.yaml`
- Replay command:

```bash
PYTHONPATH=src:. python experiments/offline/run_stage1_paper_closure.py \
  --config configs/offline/stage1_paper_closure_final.yaml
```

- Fixture manifest: `outputs/offline/stage1_paper_closure_final/fixture_manifest.json`
- Included formal fixture set: `outputs/offline/replay_fixture_selected_256x128_birkhoffctx/fixtures`
