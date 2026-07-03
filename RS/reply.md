# N16：64-Sample 公平基准实验矩阵

## 背景

粒度解耦（N15）已落地，但验证跑只有 2 samples，scheduled comm 异常高（181ms vs 历史 11ms），不可比。
原因推测：token 数不同、NCCL 冷启动、样本量过小导致方差主导。

需要用同一 workload、同一样本量，跑完所有策略 × 粒度的公平矩阵，才能得出算法质量结论。

## 目标

在 64-sample 同一 prompt 集上，跑完以下矩阵，产出可对比的结果 JSON：

| 策略 | 粒度 | 状态 |
|------|------|------|
| U_gated_maxweight_matching_atomic | wave | 待跑 |
| U_gated_maxweight_matching_atomic | atomic | 待跑 |
| birkhoff | wave | 待跑 |
| birkhoff | atomic | 待跑 |
| greedy | wave | 可选 |
| greedy | atomic | 可选 |

## 执行命令

```bash
cd /root/RouterSense/RS

STRATEGIES="U_gated_maxweight_matching_atomic birkhoff"
GRANULARITIES="wave atomic"
PROMPT_FILE="artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl"
SAMPLE_LIMIT=64
OUTPUT_ROOT="/tmp/rs_fair_benchmark"

mkdir -p $OUTPUT_ROOT

for strat in $STRATEGIES; do
  for gran in $GRANULARITIES; do
    OUT_DIR="$OUTPUT_ROOT/${strat}_${gran}"
    mkdir -p $OUT_DIR
    echo "=== Running: $strat + $gran ==="
    torchrun \
      --nproc_per_node=1 \
      --nnodes=2 \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      experiments/distributed/exp_wave_execution.py \
      --model allenai/OLMoE-1B-7B-0924-Instruct \
      --strategy $strat \
      --execution-mode scheduled_transport \
      --transport-granularity $gran \
      --prompt-file $PROMPT_FILE \
      --sample-limit $SAMPLE_LIMIT \
      --distributed-control-plane \
      --output-dir $OUT_DIR \
      2>&1 | tee "$OUT_DIR/run.log"
    echo "=== Done: $strat + $gran ==="
  done
done
```

## 结果汇总格式

跑完后，从各结果 JSON 提取以下字段，汇总到 `report.md`：

```
策略 + 粒度 | samples/s | tokens/s | mean scheduled comm (ms) | P50 (ms) | P95 (ms) | mean native (ms) | planner (ms) | 正确性
```

关注点：
- 同粒度（wave vs wave）下，策略间 scheduled comm 差异 → 算法质量
- 同策略（birkhoff vs birkhoff）下，wave vs atomic 差异 → 粒度代价
- native comm 应基本一致（控制变量验证）

## 控制变量检查

跑前确认：
- [ ] 同一 prompt 文件 `olmoe_oasst256_unique.jsonl`
- [ ] 同一 `--sample-limit 64`
- [ ] 同一 `--layer-index 0`（默认）
- [ ] 同一模型 `OLMoE-1B-7B-0924-Instruct`
- [ ] `--distributed-control-plane` 开启（矩阵聚合一致）

## 验收标准

- [ ] 4 个组合（2 策略 × 2 粒度）全部跑完，正确性均 pass
- [ ] native comm 在 4 个组合间差异 < 20%（硬件/网络控制）
- [ ] 结果写入 report.md，含上表
- [ ] 标注哪组是「算法质量对比」、哪组是「粒度代价对比」
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
