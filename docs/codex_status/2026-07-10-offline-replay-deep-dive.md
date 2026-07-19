# 2026-07-10 Offline Replay Deep Dive

This note consolidates the current CPU/offline replay evidence after the real 4GPU collection run. It is intentionally separated by semantic mode:

1. `runtime_lookahead` / phase-sync-compatible
2. `execution_window` / oracle-information joint replay
3. safe-U fallback behavior
4. P2 prediction sensitivity
5. async_release simulator interpretation

The source artifacts used here are:

- `outputs/offline/m6h_safe_u_closure/replay_suite_summary.json`
- `outputs/offline/m6h_safe_u_closure/async_release/async_release_sim_summary.json`
- `outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json`
- `outputs/offline/m6o_pre_gpu_closure/p2_sensitivity_summary.json`
- `outputs/offline/m6k_cpu_closure/oracle_gap_summary.json`
- `outputs/offline/m6k_cpu_closure/u_weight_tuning_summary.json`

## 1. Runtime-Lookahead: Current Online-Compatible Mainline

Baseline:

- `birkhoff_phase_local`: mean makespan `2056304640`

Direct phase-sync-compatible policies:

- `phase_barrier_fifo`: `2054541312` (`-0.09%` vs Birkhoff)
- `fast_bvn_single_tier`: `2060951552` (`+0.23%`)
- `routersense_multiphase_lookahead:p0_p1_p2`: `2320945152` (`+12.87%`)
- `routersense_joint_priority_phase_sync`: `2577932288` (`+25.37%`)

Interpretation:

- The current online-compatible RouterSense adapters are not competitive with `birkhoff_phase_local`.
- `routersense_multiphase_lookahead:p0_p1_p2` is materially worse than Birkhoff in offline replay.
- `routersense_joint_priority_phase_sync` is worse still.
- This matches the real 4GPU online result direction: the current bridge/policy path does not yet convert joint-scheduling information into a good phase-sync ordering.

## 2. Paired B-vs-U: Raw U vs Safe U

Paired summary from `m6h_safe_u_closure/replay_suite_summary.json`:

| Family | B mean | raw U mean | safe U mean | raw U vs B | safe U vs B | safe selected U | safe fallback to B |
|---|---:|---:|---:|---:|---:|---:|---:|
| `birkhoff_bvn` | 2054541312 | 2054541312 | 2054541312 | `0.00%` | `0.00%` | `1.00` | `0.00` |
| `gated_greedy` | 2577932288 | 2307639296 | 2280992768 | `+10.48%` | `+11.52%` | `0.75` | `0.25` |
| `gated_maxweight_matching` | 2054541312 | 2081001472 | 2054541312 | `-1.29%` | `0.00%` | `0.50` | `0.50` |
| `barrier_criticality_matching` | 2316236800 | 2060402688 | 2058067968 | `+11.05%` | `+11.15%` | `0.875` | `0.125` |
| `barrier_price_adaptive_matching` | 2054541312 | 2062739456 | `2054541312` | `-0.40%` | `0.00%` | `0.75` | `0.25` |
| `lagrangian_cross_phase` | 2054541312 | 2054541312 | 2054541312 | `0.00%` | `0.00%` | `1.00` | `0.00` |

Interpretation:

- The only clearly useful ready families are:
  - `RS_safe_barrier_criticality`
  - `RS_safe_gated_greedy`
- `RS_safe_barrier_criticality` is the strongest current safe-U mainline:
  - `+11.15%` vs paired B
  - raw and safe are very close
  - fallback ratio is only `12.5%`
- `RS_safe_gated_greedy` is the second useful family:
  - `+11.52%` vs paired B
  - fallback ratio is higher at `25%`
- `RS_safe_gated_maxweight` and `RS_safe_barrier_price` are currently diagnostic guards, not mainline winners:
  - raw U can lose B
  - safe U just collapses to parity through fallback

## 3. Execution-Window: Joint Scheduling Space Exists

This is not online-compatible evidence. It is the offline oracle-information study.

From `table_b`:

- `B_birkhoff_wave`: `3227402240`
- `U_gated_maxweight_matching`: `2961891328` (`-8.23%`)
- `U_barrier_criticality_global_matching`: `2958116864` (`-8.34%`)

Interpretation:

- The execution-window joint space is real and stable.
- `U_barrier_criticality_global_matching` is the best current raw joint policy in this table.
- The gap between current online-compatible adapters and execution-window U is still large.
- This is the main evidence for contribution 1, but it is not yet evidence of current online runtime success.

## 4. Small Oracle Gap Result

From `outputs/offline/m6k_cpu_closure/oracle_gap_summary.json`:

- `O_local_phase_oracle = birkhoff_von_neumann_fluid`
- `O_joint_definition = exact_small_instance_reference_small_fixture`
- small-fixture makespans:
  - `O_local`: `15`
  - `O_joint exact`: `13`
  - `B_birkhoff`: `19`
  - `B_barrier_criticality_matching`: `19`
  - `U_barrier_criticality_global_matching`: `19`
  - `RS_safe_barrier_criticality`: `19`

Small-fixture gap summary:

- `O_joint_vs_O_local_gap = -13.33%`
- `B_gap_to_O_local = +26.67%`
- `raw_U_gap_to_O_joint = +46.15%`
- `safe_U_gap_to_O_joint = +46.15%`

Interpretation:

- The oracle story is coherent: joint is better than local on the exact small case.
- Current RouterSense U family is still far from the small exact joint reference.
- This reinforces that the open problem is not “whether joint space exists”, but “how to convert it into an online-safe ordering that actually wins”.

## 5. Prediction Replay: Current P2 Signal Is Weak

These numbers come from `outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json`.

### RS_safe_barrier_criticality

| P2 source | mean makespan | vs zero | gap to perfect | fallback | selected U | pred L1 | cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero_hint` | 2058067968 | `0.00%` | `-0.39%` | `0.00` | `1.00` | `0.9375` | `0.0000` |
| `copy_current_dispatch` | 2058067968 | `0.00%` | `-0.39%` | `0.125` | `0.875` | `0.2432` | `0.8888` |
| `fate_style_history` | 2067800064 | `+0.52%` | `+0.11%` | `0.125` | `0.875` | `0.2236` | `0.8974` |
| `fate_style_linear` | 2062594048 | `+0.25%` | `-0.15%` | `0.25` | `0.75` | `0.3030` | `0.8675` |
| `perfect_trace_oracle` | 2065696768 | `+0.40%` | `0.00%` | `0.1875` | `0.8125` | `0.0000` | `0.9375` |

Interpretation:

- Barrier-criticality is almost insensitive to P2.
- Better traffic prediction quality does not translate into replay gain here.
- Even oracle P2 is slightly worse than zero-hint.
- This implies the current scoring path is either not consuming P2 effectively, or the safe guard neutralizes the P2 branch before it helps.

### RS_safe_gated_greedy

| P2 source | mean makespan | vs zero | gap to perfect | fallback | selected U | pred L1 | cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero_hint` | 2191441920 | `0.00%` | `-1.39%` | `0.125` | `0.875` | `0.9375` | `0.0000` |
| `copy_current_dispatch` | 2280992768 | `+4.00%` | `+2.52%` | `0.25` | `0.75` | `0.2432` | `0.8888` |
| `fate_style_history` | 2279733248 | `+3.94%` | `+2.47%` | `0.3125` | `0.6875` | `0.2236` | `0.8974` |
| `fate_style_linear` | 2244550656 | `+2.35%` | `+0.83%` | `0.25` | `0.75` | `0.3030` | `0.8675` |
| `perfect_trace_oracle` | 2224132096 | `+1.49%` | `0.00%` | `0.3125` | `0.6875` | `0.0000` | `0.9375` |

Interpretation:

- Gated-greedy reacts to P2, but in the wrong direction for current non-oracle predictors.
- `copy_current_dispatch`, `history`, and `linear` all make replay worse than `zero_hint`.
- Even oracle P2 is only mildly different, and still worse than `zero_hint` in this table.
- This suggests the current P2 path is not simply “insufficiently accurate”; it may be structurally mis-weighted or attached to the wrong decision point.

## 6. P2 Sensitivity: What The Current Policies Actually Use

From `outputs/offline/m6o_pre_gpu_closure/p2_sensitivity_summary.json`:

### RS_safe_barrier_criticality

- `zero_hint`: baseline
- `copy_current_dispatch`: no change
- `actual_trace`: `+0.40%` worse
- `perfect_trace`: same as actual trace
- `amplified_actual_2x`: no change
- `amplified_actual_4x`: no change
- `shuffled_actual`: no change

Interpretation:

- The current barrier-criticality safe policy is effectively not using the P2 signal in a way that changes replay outcome.
- The absence of reaction even to amplified/shuffled variants strongly suggests the P2 branch is either masked, clipped, or dominated by other terms.

### RS_safe_gated_greedy

- `zero_hint`: baseline
- `copy_current_dispatch`: no change in sensitivity summary, but worse in prediction replay
- `actual_trace`: `-1.98%` better than zero
- `perfect_trace`: same as actual trace
- `amplified_actual_2x`: no change
- `amplified_actual_4x`: no change
- `shuffled_actual`: no change

Interpretation:

- Gated-greedy has a real but narrow P2 response window.
- It improves only when fed the actual oracle P2, and even then the gain is small.
- The lack of response to amplified/shuffled variants suggests the current safe-U selection path is saturating quickly or falling back before the P2 signal can change ordering.

## 7. Async Release Simulator

From `outputs/offline/m6h_safe_u_closure/async_release/async_release_sim_summary.json`:

- mean completion time: `2507796480`
- mean hidden planning fraction: `1.0`
- dependency violations: `0`
- fallback replans: `0`
- mean early release tasks: `12`
- mean blocked tasks: `12`
- source safe policy: `RS_safe_barrier_criticality`
- selected policy distribution:
  - `U_barrier_criticality_global_matching`: `12`
  - `B_barrier_criticality_matching`: `4`

Interpretation:

- The simulator is internally coherent:
  - no dependency violations
  - planning can be fully hidden in the simulator model
- But it is not yet a winning path:
  - async-release sim mean makespan is still `+21.96%` vs `birkhoff_phase_local`
- This means async-release skeleton is usable as a semantics sandbox, not yet as an evidence line for performance wins.

## 8. Weight Tuning Status

From `outputs/offline/m6k_cpu_closure/u_weight_tuning_summary.json`:

- `U_barrier_criticality_global_matching`
  - best train params: `{residual=0.75, barrier=1.75, age=0.15, prediction=0.35}`
  - train mean: `1940893696`
  - eval mean: `2179911680`
  - safe eval mean: `2176843776`
  - fallback ratio: `0.125`
  - `overfit_warning=true`
- `U_gated_greedy_maximal`
  - best train params: `{residual=0.85, barrier=1.5, age=0.1, prediction=0.2}`
  - train mean: `2071240704`
  - eval mean: `2328764416`
  - safe eval mean: `2327416832`
  - fallback ratio: `0.125`
  - `overfit_warning=true`

Interpretation:

- There is no stable recommendation to change default weights yet.
- The mainline should remain:
  - `RS_safe_barrier_criticality`
  - `RS_safe_gated_greedy`
- Weight tuning is diagnostic only for now.

## 9. Current Bottom Line

The offline replay evidence currently supports the following:

1. Joint scheduling space exists.
   - Strongly supported by execution-window replay and the small exact oracle.

2. The current online-compatible adapters do not realize that space.
   - `routersense_multiphase_lookahead:p0_p1_p2` loses to `birkhoff_phase_local`.
   - `routersense_joint_priority_phase_sync` is much worse still.

3. The best current safe-U families are:
   - `RS_safe_barrier_criticality`
   - `RS_safe_gated_greedy`

4. The present P2/prediction path is not yet the bottleneck to fix first.
   - Traffic prediction quality improved from `zero_hint` to `copy/history`.
   - But scheduling outcome barely improved, and often worsened.
   - That means the current policy consumption of P2 is the more immediate issue.

5. Async-release is still a simulator semantics line, not a performance line.

## 10. Immediate Next Offline Focus

Before adding new predictor complexity, the next useful offline tasks are:

1. Inspect how `RS_safe_barrier_criticality` consumes P2 and why sensitivity is effectively zero.
2. Inspect why `RS_safe_gated_greedy` only benefits from oracle P2 and not from the current non-oracle predictors.
3. Compare safe-U ordering decisions layer-by-layer against Birkhoff on the same fixture to find where the current joint bridge starts hurting.
4. Keep `expert-first` prediction work separate until expert-to-traffic semantics are fully locked to real 4GPU traces.
