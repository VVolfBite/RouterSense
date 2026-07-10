# GPU Run B2 / C2 / A2 Commands

These commands are prepared for the next 4GPU session. They were not executed in the current environment because only one CUDA device is visible.

## Preconditions

- `torch.cuda.device_count() >= 4`
- same code revision as the final pre-GPU closure commit
- no dirty runtime/config changes between `B2`, `C2`, and `A2`

## Run B2

Goal:

- validate real prediction lifecycle
- validate zero extra prediction collectives
- validate stored `P1` plan reuse
- validate async invocation and transport timing artifacts

Template:

```bash
torchrun --standalone --nproc_per_node=4 experiments/online/run_policy_correctness.py \
  --strategy routersense_joint_predicted_async_p2p \
  --profile execution \
  --output-dir outputs/distributed/run_b2_p2_lifecycle_$(date +%Y%m%d_%H%M%S)
```

Check:

- `prediction_extra_collective_count == 0`
- `p1_planning_collective_count == 0`
- `async_executor_invocation_count > 0`
- `batch_isend_irecv_call_count > 0`
- `safe_selected_policy` non-empty

## Run C2

Goal:

- real NCCL correctness for async P2P
- no fallback
- parity for `hidden_states`, `routing_probs`, and final combine

Template:

```bash
torchrun --standalone --nproc_per_node=4 experiments/online/run_policy_correctness.py \
  --strategy routersense_joint_predicted_async_p2p \
  --profile execution \
  --enable-real-collectives \
  --selected-layers 2 \
  --forward-epochs 2 \
  --output-dir outputs/distributed/run_c2_ar1_correctness_$(date +%Y%m%d_%H%M%S)
```

Require:

- `phase_sync_fallback_count == 0`
- `batch_isend_irecv_call_count > 0`
- parity checks pass
- no timeout

## Run A2

Goal:

- fair same-executor strategy comparison

Strategies:

- `native`
- `fifo_async_p2p`
- `greedy_async_p2p`
- `birkhoff_phase_local_sync`
- `birkhoff_phase_local_async_p2p`
- `routersense_joint_phase_sync`
- `routersense_joint_zero_hint_async_p2p`
- `routersense_joint_predicted_async_p2p`

Template:

```bash
for strategy in \
  native \
  fifo_async_p2p \
  greedy_async_p2p \
  birkhoff_phase_local_sync \
  birkhoff_phase_local_async_p2p \
  routersense_joint_phase_sync \
  routersense_joint_zero_hint_async_p2p \
  routersense_joint_predicted_async_p2p
do
  torchrun --standalone --nproc_per_node=4 experiments/online/run_policy_correctness.py \
    --strategy "$strategy" \
    --profile perf \
    --warmup-iters 3 \
    --measure-iters 7 \
    --output-dir "outputs/distributed/run_a2_${strategy}_$(date +%Y%m%d_%H%M%S)"
done
```

Interpretation:

- backend gain:
  - `birkhoff_phase_local_async_p2p` vs `birkhoff_phase_local_sync`
- joint gain:
  - `routersense_joint_zero_hint_async_p2p` vs `birkhoff_phase_local_async_p2p`
- prediction gain:
  - `routersense_joint_predicted_async_p2p` vs `routersense_joint_zero_hint_async_p2p`
- full-system gain:
  - `routersense_joint_predicted_async_p2p` vs `birkhoff_phase_local_sync`
