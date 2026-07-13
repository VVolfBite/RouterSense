# M-RUNTIME-ATTRIBUTION Report

## Scope
Ran the fixed attribution-light 4GPU matrix on full model forward: native, B-core async, and U-zero raw async. No scheduling, task merge, P2P ordering, expert compute, or model-forward semantics were changed.

## Results
- Native median full forward: 140834.6 us.
- B-core median full forward: 207986.7 us, delta vs native 67152.1 us.
- U-zero median full forward: 213655.9 us, delta vs native 72821.2 us.
- B-core selected window span: 85744.8 us. Active transport sum: 7967.4 us.
- U-zero selected window span: 90717.1 us. Active transport sum: 7830.1 us.
- B-core raw/core build median: 9233.1 us. U-zero raw-U build median: 8176.9 us.
- B-core preflight median: 103.1 us. U-zero preflight median: 104.9 us.
- B-core p2p enqueue/wait median: 2020.1 / 239.8 us.
- U-zero p2p enqueue/wait median: 2002.5 / 231.6 us.

## Count And Guard Contracts
- B-core selected P0/P1 all-rank counts: 24 / 24.
- U-zero selected P0/P1 all-rank counts: 24 / 24.
- none-heavy count remained 0.
- U-zero raw-U build count all-rank: 24.
- Preflight requested/effective/executor mode: compact; collective-count exact: True.
- fallback=0, timeout=0, all_work_completed=true for B-core and U-zero.

## Attribution Validity
The GPU run did not produce selected-layer module enter/exit, expert module enter/exit, or per-phase non-overlapping cost-tree rows. Therefore:
- phase_tree_valid=false
- selected_layer_tree_valid=false
- forward_tree_valid=false

This run is valid as a cost structure smoke, but not a closed non-overlapping attribution tree. The remaining unknown is still inside selected layer/hook/framework intervals, not active P2P transport or compact preflight.

## Current Hotspots From Available Fields
1. selected_window_span remains ~85.7 ms for B-core and ~90.7 ms for U-zero, while active transport is only ~8.0 ms and ~7.8 ms.
2. raw/core build is ~9.2 ms for B-core and ~8.2 ms for U-zero.
3. p2p enqueue is ~2.0 ms and wait is ~0.24 ms; these are not the 80 ms gap.

## Required Next Fix
Instrument selected MoE module enter/exit and expert module enter/exit, and export `selected_layer_cost_tree.csv` and `phase_cost_tree.csv` from actual runtime summaries. Without these boundaries, RouterSense cannot honestly state where the ~70 ms extra over native falls in the requested cost tree.
