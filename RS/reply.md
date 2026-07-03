# N15：调度执行粒度解耦——wave / atomic 独立维度

## 背景

当前 `runner.py` 中 `ScheduledAllToAllTransport` 的 `split_into_micro_ops` 由策略名硬编码决定：

```python
# runner.py L274（当前）
split_into_micro_ops=(strategy_name != "U_gated_maxweight_matching_atomic")
```

这导致：
- `U_gated_maxweight_matching_atomic` 永远走 wave 粒度（每 wave 一次 `all_to_all_single`）
- Birkhoff / Greedy 等其他策略永远走 atomic 粒度（每 transfer op 一次 `all_to_all_single`）

**结论**：性能差异来自执行路径而非算法质量，对比不公平。

## 目标

将传输粒度（wave vs atomic）作为**独立于策略的正交维度**，任意策略 × 任意粒度均可组合运行。

## 改动清单

### 1. `RS/src/rs/runtime/distributed_ep/adapter/runner.py`

`execute_scheduled_inference` 函数签名新增参数：

```python
transport_granularity: str = "wave"
# "wave"  = 每 wave 一次 all_to_all_single（粗粒度）
# "atomic" = 每 transfer op 一次 all_to_all_single（细粒度）
```

传输选择逻辑改为：

```python
transport = (
    ScheduledAllToAllTransport(
        wave_executor,
        split_into_micro_ops=(transport_granularity == "atomic"),
    )
    if execution_mode == "scheduled_transport"
    else NativeAllToAllTransport(wave_executor)
)
```

### 2. `RS/experiments/distributed/exp_wave_execution.py`

新增 CLI 参数：

```python
parser.add_argument(
    "--transport-granularity",
    choices=["wave", "atomic"],
    default="wave",
    help="Transport granularity: 'wave' = one all_to_all per wave, "
         "'atomic' = one all_to_all per transfer op. Independent of strategy.",
)
```

调用 `execute_scheduled_inference` 时透传：

```python
transport_granularity=args.transport_granularity,
```

结果 JSON 中 `run` 块追加字段 `"transport_granularity": args.transport_granularity`。

### 3. `RS/experiments/distributed/exp_scheduled_execution.py`

该文件走 `p2p_matching` 模式，不经过 wave 路径，无需改动。

## 实验矩阵（改后跑法示例）

```bash
# 同一策略，两种粒度
for strat in U_gated_maxweight_matching_atomic birkhoff greedy; do
  for gran in wave atomic; do
    torchrun --nproc_per_node=1 --nnodes=2 \
      experiments/distributed/exp_wave_execution.py \
      --strategy $strat \
      --execution-mode scheduled_transport \
      --transport-granularity $gran \
      --prompt-file artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl \
      --sample-limit 64
  done
done
```

## 验收标准

- [ ] 所有策略 × {wave, atomic} = 2N 种组合均可正常运行
- [ ] `--transport-granularity=wave` 下，所有策略执行路径一致（每 wave 一次 `all_to_all_single`）
- [ ] 结果 JSON 中 `transport_granularity` 字段正确记录
- [ ] `native_baseline` 和 `wave_collective` 模式不受影响（不读取该参数）
