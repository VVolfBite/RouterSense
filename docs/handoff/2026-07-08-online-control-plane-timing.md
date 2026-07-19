# 2026-07-08 Online Control-Plane Timing

## Scope

This note records the 4-GPU timing results after replacing Python object
collectives in `plan_agreement.py` with tensor-based wire encoding.

Run artifact root:

`artifacts/comparison/wire_tensor_core_4gpu_64_20260708/`

Baseline for comparison:

`artifacts/comparison/wire_slim_core_4gpu_64_fixaudit_20260708/`

Model / workload:

- OLMoE-1B-7B-0924
- single-node EP=4
- 64 prompts
- 1 repetition

## What Changed

`src/rs/runtime/online/megatron_ep/control/plan_agreement.py`

- `planning_summary` gather path:
  `all_gather_object` -> `all_gather` on flat `int64` tensor
- `abstract plan` broadcast path:
  `broadcast_object_list` -> `broadcast` on flat `int64` tensor
- removed second verify collective:
  local materialized `plan_hash` is compared directly against root-broadcast
  `plan_hash`

No runtime hook, executor, `TransferLayout`, or NCCL payload semantics were
changed.

## New 4-GPU Results

### End-to-End

| strategy | total_forward_us | communication_makespan_us | communication_collective_active_us | scheduling_overhead_us |
| --- | ---: | ---: | ---: | ---: |
| disabled | 30,863,375 | 363,286 | 363,286 | 0 |
| routersense_p0p1_reservation | 31,103,957 | 340,708 | 247,628 | 319,371 |
| routersense_p0p1p2_hint | 33,280,028 | 403,209 | 271,202 | 567,845 |

### Agreement Hot Path

| strategy | avg_total_agreement_us | avg_all_gather_us | avg_broadcast_us | avg_hash_verify_us | avg_build_plan_us |
| --- | ---: | ---: | ---: | ---: | ---: |
| routersense_p0p1_reservation | 9,331.692 | 6,206.452 | 164.044 | 0.386 | 846.097 |
| routersense_p0p1p2_hint | 16,743.060 | 4,262.917 | 219.561 | 0.562 | 10,545.539 |

### Runtime Bookkeeping Around Planning

| strategy | avg_build_p2_hint_us | avg_store_prepared_plan_us | avg_record_window_state_us | avg_prepared_phase_plan_shadow_us |
| --- | ---: | ---: | ---: | ---: |
| routersense_p0p1_reservation | 32.009 | 3,583.129 | 2,833.902 | 4,643.781 |
| routersense_p0p1p2_hint | 2,300.906 | 5,150.485 | 4,043.258 | 6,920.784 |

## Improvement Versus Old Object Wire

### `routersense_p0p1_reservation`

- `avg_all_gather_time_us`: `57,389.045 -> 6,206.452` (`0.108x`)
- `avg_broadcast_time_us`: `7,854.822 -> 164.044` (`0.021x`)
- `avg_hash_verify_time_us`: `51,528.597 -> 0.386`
- `avg_total_agreement_time_us`: `118,457.166 -> 9,331.692` (`0.079x`)
- `scheduling_overhead_us`: `3,810,413 -> 319,371` (`0.084x`)

### `routersense_p0p1p2_hint`

- `avg_all_gather_time_us`: `50,998.052 -> 4,262.917` (`0.084x`)
- `avg_broadcast_time_us`: `7,750.431 -> 219.561` (`0.028x`)
- `avg_hash_verify_time_us`: `49,858.100 -> 0.562`
- `avg_total_agreement_time_us`: `116,707.030 -> 16,743.060` (`0.143x`)
- `scheduling_overhead_us`: `3,755,018 -> 567,845` (`0.151x`)

## Interpretation

The control-plane wire change worked.

The old overhead was dominated by Python object collective cost, not by payload
size:

- `planning_summary` payload was already about 202 bytes
- `abstract plan` payload was already about 944 bytes

After the tensor wire change, collective overhead dropped by roughly one order
of magnitude.

The next dominant bottlenecks are no longer gather/broadcast serialization.
They are now inside the online pending-window path:

- `routersense_p0p1p2_hint` `avg_build_plan_us` ~ 10.5 ms
- `avg_build_p2_hint_us` ~ 2.3 ms
- `avg_store_prepared_plan_us` ~ 5.15 ms
- `avg_prepared_phase_plan_shadow_us` ~ 6.92 ms

This means the next optimization target is the pending-window / prepared-plan
logic, not NCCL control-plane serialization.

## Event Counts

`rank0_control_timeline.jsonl` counts:

### `routersense_p0p1_reservation`

- `before_phase_plan`: 32
- `phase_execution_plan_agreed`: 32
- `before_wave`: 516
- `after_wave`: 516
- `before_payload_collective`: 516
- `after_payload_collective`: 516
- `planning_stage_timing`: 192
- `shadow_plan_arrival`: 32

### `routersense_p0p1p2_hint`

- `before_phase_plan`: 32
- `phase_execution_plan_agreed`: 32
- `before_wave`: 513
- `after_wave`: 513
- `before_payload_collective`: 513
- `after_payload_collective`: 513
- `planning_stage_timing`: 192
- `shadow_plan_arrival`: 32

The run completed end-to-end; this is not a partial or failed trace.

## Replay / Follow-Up

Replay-critical artifacts were preserved under the comparison run directory,
including:

- `rank*_phase_contexts.jsonl`
- `rank*_scheduled_phase_plans.jsonl`
- `rank*_transport_execution.jsonl`
- `rank*_control_timeline.jsonl`
- `rank*_planning_timing.jsonl`
- `rank*_plan_arrival_records.jsonl`
- `rank*_prepared_plan_bindings.jsonl`
- `rank*_prepared_phase_plan_shadow.jsonl`
- `rank*_window_state.jsonl`

Recommended next step:

1. replay `routersense_p0p1_reservation`
2. replay `routersense_p0p1p2_hint`
3. compare bucket order / wave order / prepared-plan alignment
4. optimize pending-window planner and prepared-plan bookkeeping

## Bottom Line

The tensor wire conversion removed the object-collective bottleneck.

The online path is no longer blocked by Python control-plane serialization.
The remaining work is to reduce pending-window planning/bookkeeping cost and to
improve `routersense_p0p1p2_hint` scheduling quality relative to
`routersense_p0p1_reservation`.
