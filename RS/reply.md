# N17：结果诊断 + 代码重构 + 下一步实验

## Part 1：结果诊断——为什么分布式没体现 POC1 增益

### 现状

| 指标 | native baseline | scheduled（最优） | 倍率 |
|------|----------------|-----------------|------|
| mean comm | ~2 ms | 8.13 ms | **4x 更差** |
| samples/s | — | 5.2-5.3 | 4 组合几乎无差异 |

**核心问题**：调度后的通信比 native 一次 `all_to_all_single` 慢了 4 倍，所有策略表现几乎相同。

### 根因分析

1. **NCCL 调用延迟地板**：每次 `all_to_all_single` 即使传 0 行数据，也有 ~0.5-1ms 的 NCCL 内部开销（kernel launch + sync）。native 调 1 次 ≈ 2ms；wave 调度拆成 N 个 wave = N 次 NCCL 调用 → 开销线性叠加。

2. **Pack/Unpack 开销**：每个 wave 需要从 `token_buffer` 中提取对应行、拼接成连续 tensor、通信后再 unpack。这些 CPU+GPU 内存操作不在 POC1 的 makespan 模型里。

3. **POC1 makespan 模型是纯理论值**：它衡量的是「在给定流量矩阵下，最优排序的理论 makespan」，假设通信时间与数据量成正比、无启动开销。真实 NCCL 不满足这个假设。

4. **策略间差异被噪声淹没**：birkhoff 和 U_gated_atomic 的调度质量差异（在 makespan 上可能 10-20%）在 8ms 的通信总时间里只有 ~1ms 差异，被 pack/unpack/NCCL 噪声覆盖。

### 修正方向

调度增益在真实环境中的体现需要满足：

```
单波通信数据量 >> NCCL 启动开销
```

当前 2 节点 × 1 GPU、OLMoE-1B（64 tokens × 2048 hidden × fp16 = 128KB/token 层），流量矩阵规模太小。需要：
- 更大 batch（128-256 samples）→ 每 wave 数据量更大 → NCCL 开销占比下降
- 更大模型（Mixtral 8x7B）→ hidden_size 4096、更多 expert → 单 wave 数据量更大
- 更多 GPU（4+）→ 流量矩阵更大 → 调度空间更有意义

## Part 2：代码重构

### 2.1 `scheduler/` 目录重构

**问题**：
- `strategy.py`（基类 + 注册表）和 `strategies.py`（工厂）职责重叠
- 具体策略文件（birkhoff/greedy/oracle 等）和框架文件混在同一层

**目标结构**：

```
scheduler/
  __init__.py         ← 对外导出（SchedulingContext, get_strategy 等）
  _common.py          ← 通用工具（保持不变）
  strategy.py         ← 基类 SchedulingStrategy + SchedulingContext + SchedulingResult + 注册表 + get_strategy()
                       （合并 strategies.py 内容进来，删除 strategies.py）
  strategies/         ← 具体策略实现子目录
    __init__.py
    greedy.py
    birkhoff.py
    oracle.py
    local_search.py
    cross_phase.py
    global_matching.py
    multiphase_global.py
```

**操作**：
1. 将 `strategies.py` 的 `get_strategy()` 和注册逻辑合并到 `strategy.py`
2. 删除 `strategies.py`
3. 将 `greedy.py` / `birkhoff.py` / `oracle.py` / `local_search.py` / `cross_phase.py` / `global_matching.py` / `multiphase_global.py` 移入 `strategies/`
4. `fast.py`（仅 4 行 re-export）删除，其导出在 `__init__.py` 中处理
5. 更新 `__init__.py` 的 import 路径

### 2.2 `core/` 目录审查

**问题文件**：

| 文件 | 行数 | 现状 | 建议 |
|------|------|------|------|
| `correctness.py` | 29 | 只有 `summarize_dispatch_plans()`，不是正确性验证 | 移到 `evaluation/` 或合并到 `manifest.py` |
| `worker_loop.py` | 25 | 极小，只被 wave_executor 用 | 合并到 `wave_executor.py` |
| `collective.py` | 121 | 旧 P2P 执行记录，与 `nccl_executor.py` 职责重叠 | 合并到 `nccl_executor.py`，删除 |

**保留在 core/ 的文件**：

```
core/
  __init__.py
  manifest.py         ← 数据结构（DispatchPlan/RouteItem/WaveSpec）
  placement.py        ← 专家放置
  scheduler.py        ← facade
  nccl_executor.py    ← NCCL 操作 + collective 记录（合并后）
  wave_planner.py     ← 调度结果 → wave 转换
  wave_executor.py    ← wave 执行 + worker_loop（合并后）
```

### 2.3 `experiments/` 目录重构

**问题**：
- 命名不一致（有的 `exp_` 开头，有的 `distributed_` 开头，有的 `smoke` 在后）
- 目录名 `poc_line1` 和 `distributed` 不够直观

**目标结构**：

```
experiments/
  poc/                ← 离线 POC 实验（原 poc_line1）
    exp_trace.py
    exp_oracle.py
    exp_pairwise.py
    exp_pairwise_candidate_compare.py
    exp_pairwise_model_compare.py
    exp_multiphase_global_matching.py
    exp_ablation_fluid_vs_joint.py
    exp_cross_layer.py
    full_sequence_trace.py
    full_sequence_trace_qwen.py
    build_prompt_mix.py
    pairwise_scheduler.py

  dep/                ← 真实部署/分布式实验（原 distributed）
    exp_wave_execution.py          ← 主力实验
    exp_scheduled_execution.py     ← 老版调度注入
    exp_nccl_smoke.py              ← NCCL smoke
    exp_olmoe_ep.py                ← OLMoE EP smoke
    exp_link_smoke.py              ← 链路 smoke
    smoke_nccl.py                  ← NCCL 连通性 smoke（原 distributed_nccl_smoke）
    smoke_olmoe_ep.py              ← OLMoE EP smoke（原 distributed_olmoe_ep_smoke）
    smoke_multinode.py             ← 多节点 smoke（原 future_multinode_smoke）
    _bootstrap.py
```

**命名规则**：
- `exp_` = 正式实验（产出结果数据）
- `smoke_` = 快速验证（通过/不通过）
- `full_sequence_trace` = trace 采集工具（保留原名）

### 2.4 引用更新

所有 import 路径需要同步更新：
- `from rs.scheduler.strategies import get_strategy` → `from rs.scheduler.strategy import get_strategy`
- `from rs.scheduler.birkhoff import ...` → `from rs.scheduler.strategies.birkhoff import ...`
- `experiments/distributed/` → `experiments/dep/`
- `experiments/poc_line1/` → `experiments/poc/`

测试文件（`tests/`）和脚本（`scripts/`）中引用这些路径的也需一并修改。

## Part 3：下一步实验

### 3.1 扩大 batch size（验证增益是否随规模出现）

```bash
# 128-sample 和 256-sample，只跑 wave 粒度
for n in 128 256; do
  for strat in U_gated_maxweight_matching_atomic birkhoff greedy; do
    torchrun ... exp_wave_execution.py \
      --strategy $strat \
      --execution-mode scheduled_transport \
      --transport-granularity wave \
      --sample-limit $n
  done
done
```

关注：scheduled comm / native comm 的比值是否随 batch 增大而改善。

### 3.2 Greedy 基线加入对比

greedy 是最简单的调度策略，如果 U_gated/birkhoff 不能显著优于 greedy，说明调度本身在当前规模下没有增益。

### 3.3 native_baseline 模式补跑

为每个 batch size 都跑一次 `--execution-mode native_baseline`，作为真正的「无调度」基线。scheduled 要赢的不是其他调度策略，而是 native。

## 验收标准

### 代码重构
- [ ] scheduler/ 合并 strategy.py + strategies.py，策略文件移入 strategies/
- [ ] core/ 合并 correctness → manifest，合并 worker_loop → wave_executor，合并 collective → nccl_executor
- [ ] experiments/ 重命名 poc_line1 → poc，distributed → dep
- [ ] experiments/dep/ 内 smoke 文件统一 smoke_ 前缀
- [ ] 所有 import 更新，`python -c "from rs.scheduler import get_strategy"` 不报错
- [ ] `python -m pytest tests/` 全部通过

### 实验
- [ ] 128-sample + 256-sample wave 粒度矩阵完成
- [ ] native_baseline 各 batch size 补跑完成
- [ ] greedy 加入对比
- [ ] report.md 更新含新结果 + 规模效应分析
