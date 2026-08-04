# Multimodel U-Scheduler Snapshot

## Scope

This report captures the current single-node execution-window results before
moving to real distributed deployment.

Retained runs:
- OLMoE tier2 quick comparison on the retained execution-window setup
- Qwen trace collection on the same 256-prompt OASST corpus
- Qwen tier2 candidate comparison
- Qwen small-sample fluid-reference run

## Key interpretation

The cross-layer prediction idea itself is not the heavy part of the pipeline.
What is expensive today is the offline analysis harness:
- decoding many layer-pair predictions;
- repeatedly materializing cross-layer traffic matrices;
- then evaluating many schedulers over those expanded pairs.

In deployment, prediction can be reduced to lightweight online statistics or a
single next-layer routing decode, instead of replaying the full offline study.

## OLMoE retained result

Source snapshot:
- `artifacts/olmoe_n8_s64_summary.json`

Important retained numbers:
- `B_birkhoff`: `33.47%`
- `B_birkhoff_wave`: `51.96%`
- `B_barrier_aware_birkhoff`: `35.55%`
- `B_barrier_aware_birkhoff_wave`: `51.84%`
- `U_lagrangian`: `46.01%`
- `U_ibbr`: `35.14%`
- `U_gated_maxweight_matching_atomic`: `59.14%`
- `U_gated_maxweight_matching`: `63.81%`
- `U_barrier_criticality_global_matching_atomic`: `59.14%`
- `U_barrier_criticality_global_matching`: `63.76%`

Interpretation:
- wave scheduling materially improves both B and U families;
- U-wave remains the strongest family;
- on OLMoE, the best U-wave methods sit very close to the strongest small-slice
  fluid-wave reference that was tested earlier in this branch.

## Qwen trace collection

Source snapshot:
- `artifacts/qwen_trace_oasst256_summary.json`

Collected trace facts:
- corpus: same retained 256 non-repeated OASST prompts used for OLMoE
- model: `Qwen1.5-MoE-A2.7B`
- MoE layers: `24`
- top-k: `4`
- record count: `794880`

Interpretation:
- the trace path works on 2 local GPUs via `device_map=auto`;
- Qwen uses a deeper MoE stack than the retained OLMoE setup, so offline
  pairwise analysis cost is naturally higher.

## Qwen tier2 result

Source snapshot:
- `artifacts/qwen_tier2_s64_summary.json`
- `artifacts/qwen_tier2_s64_table.md`
- `artifacts/qwen_tier2_s64_table_e2e.md`

Important retained numbers:
- `U_barrier_price_adaptive_matching`: `39.06%`, `10.41 ms`
- `U_gated_greedy_maximal`: `38.63%`, `7.98 ms`
- `U_barrier_price_adaptive_matching_atomic`: `22.89%`, `4.36 ms`
- `B_barrier_aware_birkhoff`: `21.15%`, `9.60 ms`
- `U_gated_greedy_maximal_atomic`: `20.39%`, `2.79 ms`
- `B_birkhoff_wave`: `19.87%`, `12.47 ms`
- `U_ibbr`: `17.53%`, `5.20 ms`
- `U_lagrangian`: `16.90%`, `7.19 ms`
- `B_birkhoff`: `14.88%`, `0.97 ms`
- `U_cp_lpt`: `-3.53%`

Interpretation:
- the best Qwen methods are again the new U-family schedulers;
- the ranking is consistent with the OLMoE conclusion;
- atomic variants remain clearly weaker than wave variants;
- `U_cp_lpt` remains a weak candidate.

## Qwen fluid-reference result

Source snapshot:
- `artifacts/qwen_oracle_fluid_s32_summary.json`

Important retained numbers:
- `U_barrier_price_adaptive_matching`: `38.93%`
- `U_gated_greedy_maximal`: `38.45%`
- `oracle_wave_fluid`: `37.23%`
- `oracle_wave_atomic`: `17.47%`

Interpretation:
- the current `oracle_wave_fluid` implementation is not a strict upper bound;
- it should be treated as a fluid-wave reference line, not as a formal oracle;
- the observed practical ceiling on this branch appears to be around
  `39%` to `40%` on the retained N=8 execution-window studies.

## Working conclusion before deployment

The retained single-node evidence supports the following:
1. cross-layer prediction remains viable and does not appear intrinsically
   expensive;
2. the best current practical candidates are U-family wave schedulers;
3. the observed headroom relative to strong B baselines is still material;
4. a realistic current target ceiling is about `39%` to `40%`, not far above
   the best current U-family results;
5. the next stage should focus on real distributed execution and deployment
   semantics rather than more offline candidate proliferation.
