# Runtime Timeline GPU Report

commit: `29f88e7d8018102cb73a7cc84078a07e1b9757e5`
status: `executed`

## Headline
The selected-layer hook scope remains correct. The 88-91 ms selected window is not explained by P2P active transfer; measured per-phase submit/wait critical paths are sub-millisecond, while P0-to-P1 gaps are ~15-17 ms per selected layer/rank and aggregate window spans across selected phases/layers.

## Strategy Summary

### routersense_b_core_independent_async
- full forward: 244968.2 us
- all-rank selected window: 88784.0 us
- transport sum: 7813.1 us
- phase active critical path median/max: 932.3/1055.3 us
- phase submit span median/max: 926.7/1049.7 us
- phase request wait median/max: 5.455/6.217 us
- P0->P1 gap median/max: 17041.7/18033.6 us
- control/raw build: 9098.1/8960.1 us
- batch_isend_irecv calls: 72
- critical rank: 3

### routersense_u_core_zero_raw_async
- full forward: 234289.5 us
- all-rank selected window: 91478.7 us
- transport sum: 5617.5 us
- phase active critical path median/max: 744.5/1158.3 us
- phase submit span median/max: 739.8/1150.8 us
- phase request wait median/max: 4.713/7.458 us
- P0->P1 gap median/max: 15257.2/18758.2 us
- control/raw build: 5945.8/5839.6 us
- batch_isend_irecv calls: 72
- critical rank: 3

## Interpretation
- The heavy component is the selected communication window span, not the actual P2P primitive active time.
- Per-phase submit and wait are small; the window is dominated by gaps between selected dispatch/return phases and rank/layer scheduling spread.
- `preflight_collective_count` is still 18 per measured rank summary, so compact mode did not remove all collective preflight work from the selected path.
- The runner marks strategies ineligible only because C2 qualification was not part of this diagnostic run; fallback and timeout are zero.

## Files
- `dialogue/evidence/timeline_summary.json`
- `dialogue/evidence/timeline_per_rank.csv`
- `dialogue/evidence/rank_imbalance_summary.csv`
- `dialogue/evidence/phase_gap_summary.csv`
- `dialogue/evidence/task_granularity_summary.json`
