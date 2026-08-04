# RS 主线重构与后续开发任务书

## 0. 背景与目标

### 0.1 当前状态

RS 主线处于 Phase 0B 完成、Phase 0C 空壳阶段：

| 模块 | 状态 | 评估 |
|---|---|---|
| `runtime/single_gpu.py` | 完整 | 可复用，需迁移到 `local_test/` |
| `trace/olmoe_router_trace.py` | 完整 | 可复用 |
| `topology/inventory.py` + `paths.py` | 完整 | 可复用 |
| `evaluation/artifacts.py` | 完整 | 可复用 |
| `runtime/distributed_ep/*.py`（8 个文件） | 全部空壳 | 需重写，按新结构拆分 |
| `scheduler/` | 空目录 | 需定义归属 |
| `experiments/deployment/*.py`（7 个脚本） | 完整 | 需迁移到 `experiments/prerun/` |
| `configs/` | 存在但不规范 | 需迁移到 `experiments/` 下 |
| `deploy/` | 存在但不规范 | 需重新定位为远端最小可运行环境 |

### 0.2 本次任务目标

1. **代码结构重组**：按分层决策重组现有代码
2. **前置验证归集**：smoke 脚本收拢到 `experiments/prerun/`
3. **分布式 EP 重写**：按 core/adapter 分离原则搭建框架
4. **deploy/ 规范化**：明确远端最小可运行环境的定位
5. **消融实验模块**：为后续消融实验预留模块位置

### 0.3 POC1/POC2 的结论对本次任务的影响

| POC | 结论 | 对本次任务的意义 |
|---|---|---|
| POC1 | proxy 分析不可靠，必须做真实消融 | 消融实验是优先级最高的后续任务，本次重组需为其预留位置 |
| POC2 | 同步 barrier 下调度杠杆太弱，dependency ≈ state | 分布式 EP 需避免同步 barrier 结构，考虑更细粒度的 overlap |

---

## 1. 目标代码结构

```
RS/
├── pyproject.toml
├── README.md
├── configs/                         # 保留顶层，仅放集群/模型级配置
│   ├── model/
│   │   └── olmoe_1b_7b_0125.yaml    # 模型配置
│   └── topology/
│       └── torchrun_2node_2gpu.yaml  # 集群拓扑
├── deploy/                           # 远端最小可运行环境
│   ├── scripts/
│   │   ├── prepare_node.sh
│   │   ├── sync_to_remote.sh
│   │   ├── launch_remote.sh
│   │   ├── prefetch_model.sh
│   │   └── collect_logs.sh
│   ├── env/
│   │   └── requirements-gpu.txt
│   └── README.md
├── src/routesense/
│   ├── __init__.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── local_test/               # 单卡推理
│   │   │   ├── __init__.py
│   │   │   └── inference.py          # 从 single_gpu.py 迁移
│   │   └── distributed_ep/
│   │       ├── __init__.py
│   │       ├── core/                 # 模型无关：通信 + 调度
│   │       │   ├── __init__.py
│   │       │   ├── collective.py     # NCCL all_to_all_single 封装
│   │       │   ├── worker_loop.py    # 每个 rank 的工作循环
│   │       │   ├── placement.py      # 专家放置策略
│   │       │   ├── scheduler.py      # 调度逻辑（从原 scheduler/ 合入）
│   │       │   ├── manifest.py       # 运行 manifest
│   │       │   └── correctness.py    # 分布式 vs 单卡正确性对比
│   │       └── adapter/              # 模型相关：OLMoE 适配
│   │           ├── __init__.py
│   │           ├── olmoe_adapter.py  # OLMoE MoE 层包装
│   │           ├── expert_store.py   # 本地专家权重管理
│   │           └── runner.py         # 分布式运行入口（调用 core + adapter）
│   ├── trace/
│   │   ├── __init__.py
│   │   └── olmoe_router_trace.py    # 保持不变
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── paths.py                  # 保持不变
│   │   └── inventory.py             # 保持不变
│   └── evaluation/
│       ├── __init__.py
│       └── artifacts.py             # 保持不变
├── experiments/
│   ├── prerun/                       # 前置验证脚本（从 deployment/ 迁移）
│   │   ├── probe_architecture.py
│   │   ├── single_gpu_smoke.py
│   │   ├── router_trace_smoke.py
│   │   └── text_inference.py
│   ├── ablation/                     # 消融实验（后续任务）
│   │   ├── configs/
│   │   │   ├── smoke.yaml
│   │   │   ├── poc.yaml
│   │   │   └── formal.yaml
│   │   └── scripts/
│   └── distributed/                  # 分布式实验（后续任务）
│       ├── configs/
│       └── scripts/
├── tests/
│   ├── test_inventory.py
│   └── test_artifacts.py
├── artifacts/
└── outputs/
```

---

## 2. 文件迁移计划

### 2.1 迁移：runtime/single_gpu.py → runtime/local_test/inference.py

**原文件**：`src/routesense/runtime/single_gpu.py`（170 行）
**新位置**：`src/routesense/runtime/local_test/inference.py`

迁移操作：
1. 创建 `runtime/local_test/__init__.py`
2. 将 `single_gpu.py` 全部内容复制到 `local_test/inference.py`
3. 更新 `runtime/__init__.py` 的 import 路径
4. 删除 `runtime/single_gpu.py`

不需要修改代码逻辑，只改路径。

### 2.2 迁移：experiments/deployment/ → experiments/prerun/

**原文件**：
| 原文件名 | 新文件名 | 备注 |
|---|---|---|
| `single_gpu_olmoe_smoke.py` | `single_gpu_smoke.py` | 简化命名 |
| `single_gpu_router_trace_smoke.py` | `router_trace_smoke.py` | 简化命名 |
| `probe_olmoe_architecture.py` | `probe_architecture.py` | 简化命名 |
| `single_gpu_text_infer.py` | `text_inference.py` | 简化命名 |
| `distributed_olmoe_reference.py` | 删除或保留为归档 | 功能与 single_gpu_smoke 重复 |
| `distributed_nccl_smoke.py` | 删除或移到 `experiments/distributed/` | 属于分布式实验 |
| `future_multinode_smoke.py` | 删除或移到 `experiments/distributed/` | 属于多节点实验 |

迁移操作：
1. 创建 `experiments/prerun/`
2. 迁移 4 个单卡 smoke 脚本，简化命名
3. 更新脚本内的 import 路径（`runtime.single_gpu` → `runtime.local_test.inference`）
4. 删除或归档 `experiments/deployment/` 目录
5. `distributed_olmoe_ep_smoke.py` 等分布式相关脚本暂移到 `experiments/distributed/`

### 2.3 迁移：configs/ → experiments/ 下

**当前位置**：`configs/model/olmoe_1b_7b_instruct.yaml`
**新位置**：保留顶层 `configs/`，但增加实验级配置目录

保留顶层 `configs/` 用于模型和拓扑配置（这些是项目级的）。
实验级配置（如消融实验的 smoke/poc/formal）放到 `experiments/ablation/configs/` 下。

### 2.4 迁移：scheduler/ → distributed_ep/core/scheduler.py

**原文件**：`src/routesense/scheduler/`（空目录）
**新位置**：`src/routesense/runtime/distributed_ep/core/scheduler.py`

调度逻辑属于 runtime 的 core 层，不需要独立的顶层模块。
删除 `scheduler/` 目录。

### 2.5 迁移：oracle/ 目录

**原文件**：`src/routesense/oracle/`（空目录，只有 .gitkeep）
**处理**：删除。Oracle 策略将来放在消融实验的 `policies.py` 中，不需要独立模块。

---

## 3. 分布式 EP 详细规格

### 3.1 core/ — 模型无关层

#### `collective.py` — NCCL 通信封装

```python
class CollectiveOps:
    """
    封装 NCCL all_to_all_single 的 dispatch 和 return。

    dispatch(hidden_states, send_counts, recv_counts) -> received_states:
        将 hidden_states 按 send_counts 分发到各 rank，
        接收来自各 rank 的数据。

    return_results(processed_states, return_send_counts, return_recv_counts) -> gathered_results:
        将处理后的结果发回原始 rank。

    约束：
    - 只封装 NCCL 调用，不做调度决策
    - 输入输出都是 torch.Tensor
    - 支持同步和异步两种调用模式（async_op=True/False）
    - 记录每次通信的 send/recv bytes 和时间戳
    """
```

#### `placement.py` — 专家放置策略

```python
class PlacementStrategy:
    """
    决定每个 expert 放在哪个 GPU 上。

    round_robin(num_experts, num_gpus) -> dict[expert_id, gpu_rank]:
        均匀分配：#0→GPU0, #1→GPU1, ..., #N→GPU(N%num_gpus)

    load_aware(expert_loads, gpu_capacities) -> dict[expert_id, gpu_rank]:
        根据负载分配，平衡各 GPU 工作量

    colocate(coactive_groups, num_gpus) -> dict[expert_id, gpu_rank]:
        将经常被同时选中的 expert 放在同一 GPU，减少通信
    """
```

#### `worker_loop.py` — Rank 工作循环

```python
class WorkerLoop:
    """
    每个 rank 的主循环：
    1. 接收 dispatch 数据
    2. 调用 adapter 执行专家计算
    3. return 结果
    4. 更新 runtime state

    支持两种模式：
    - 同步模式：dispatch → compute → return → barrier
    - 异步模式：dispatch/compute/return 可以 overlap
    """
```

#### `scheduler.py` — 调度逻辑

```python
class Scheduler:
    """
    决定 dispatch 顺序和 bucket 释放顺序。

    输入：bucket 列表（每个 bucket = 一组要发到同一个 expert 的 token）
    输出：release_order（按什么顺序发送这些 bucket）

    策略接口：
    - fifo: 先到先发
    - state_only: 看 bucket 大小、队列深度
    - dependency_aware: 看 bucket 间的共现关系
    - oracle: 看真实 delta_nll（需要消融实验数据）
    """
```

#### `manifest.py` — 运行 Manifest

```python
@dataclass
class DistributedManifest:
    """
    分布式运行的元信息：
    - 参与的 rank 列表
    - 每个 rank 的 GPU 信息
    - 专家放置映射
    - 使用的调度策略
    - NCCL 配置参数
    """
```

#### `correctness.py` — 正确性对比

```python
def verify_distributed_vs_single(
    single_gpu_output: torch.Tensor,
    distributed_output: torch.Tensor,
    tolerance: float = 1e-3,
) -> CorrectnessResult:
    """
    对比单卡推理和分布式推理的输出是否一致。
    使用 allclose + 最大差异 + 相对误差。
    """
```

### 3.2 adapter/ — 模型相关层

#### `olmoe_adapter.py` — OLMoE 适配

```python
class OLMoEAdapter:
    """
    包装 OLMoE 的 MoE 层为 EP dispatch 接口。

    职责：
    1. 从 model.layers[i].mlp 提取 gate 和 experts
    2. 给定 hidden_states + routing 决策，执行 expert 计算
    3. 支持消融操作（将指定 expert 的权重置零）

    接口：
    - extract_routing_info(hidden_states) -> router_logits, top_k_indices, top_k_weights
    - compute_experts(hidden_states, expert_ids, weights) -> output
    - ablate_expert(hidden_states, target_expert_id) -> output_with_ablation
    """
```

#### `expert_store.py` — 专家权重管理

```python
class ExpertStore:
    """
    管理本地 GPU 上的专家权重。

    职责：
    1. 从完整模型中提取指定 expert 的权重
    2. 支持热加载/卸载（用于专家迁移实验）
    3. 记录每个 expert 的显存占用
    """
```

#### `runner.py` — 分布式运行入口

```python
def run_distributed_ep(
    model, tokenizer, placement, scheduler,
    input_texts: list[str],
    num_gpus: int,
) -> DistributedResult:
    """
    分布式 EP 的顶层入口。

    流程：
    1. 初始化 NCCL 进程组
    2. 用 placement 分配专家到 GPU
    3. 对每个 input：
       a. forward 到目标 MoE 层
       b. 用 adapter 提取 routing 信息
       c. 用 scheduler 决定 dispatch 顺序
       d. 用 collective 做 dispatch
       e. 各 rank 执行 expert 计算
       f. 用 collective 做 return
       g. 合并结果，继续 forward
    4. 对比分布式 vs 单卡正确性
    """
```

---

## 4. deploy/ 详细规格

### 4.1 定位

```
deploy/ = 远端节点的最小可运行环境
```

远端节点不需要完整的 RS 代码仓库。主程序负责：
1. 将 deploy/ 内容打包
2. 通过 scp 推送到远端节点
3. 在远端节点执行 prepare_node.sh 初始化环境
4. 通过 launch_remote.sh 启动实验

### 4.2 目录结构

```
deploy/
├── scripts/
│   ├── prepare_node.sh         # 远端环境初始化（pip install、目录创建）
│   ├── sync_to_remote.sh       # 从本地 scp deploy/ 到远端
│   ├── launch_remote.sh        # 在远端启动 torchrun 进程
│   ├── prefetch_model.sh       # 远端模型预下载
│   ├── collect_logs.sh         # 收集远端日志
│   └── stop_rs_jobs.sh         # 停止远端任务
├── env/
│   └── requirements-gpu.txt    # 远端 Python 依赖
├── inventory/
│   ├── hosts.local.yaml        # 本地开发用
│   ├── hosts.example.yaml      # 模板
│   └── hosts.cluster.yaml      # 生产集群（gitignore）
└── README.md
```

### 4.3 sync_to_remote.sh 的设计

```bash
#!/bin/bash
# 用法：./sync_to_remote.sh <node_name>
# 将 deploy/ 目录 scp 到远端节点的指定路径

# 步骤：
# 1. 从 inventory 读取远端节点的 host、port、ssh_user、remote_path
# 2. rsync -avz deploy/ ${ssh_user}@${host}:${remote_path}/deploy/
# 3. 验证远端文件完整性（checksum）
# 4. 在远端执行 prepare_node.sh
```

### 4.4 requirements-gpu.txt

```
torch>=2.1.0
transformers>=4.45.0
accelerate
safetensors
numpy
pyyaml
```

远端只需要这些依赖，不需要 pytest、matplotlib 等开发依赖。

---

## 5. trace/ 模块评估与增强计划

### 5.1 当前状态

`olmoe_router_trace.py`（157 行）已经完整实现了：
- 真实 model forward → router_logits 采集
- softmax + topk → expert 选择 + 权重
- 每条 trace 记录：expert_id, routing_weight, expert_rank_within_topk, topk

### 5.2 消融实验需要的增强

当前 trace 缺少以下路由上下文特征（消融实验需要）：

| 缺失特征 | 含义 | 是否需要添加 |
|---|---|---|
| `top1_top2_gap` | Top-1 和 Top-2 的概率差 | 消融实验需要 |
| `routing_entropy` | 全部 expert 概率分布的熵 | 消融实验需要 |
| 全部 expert 的 probability | 完整概率分布 | 消融实验需要（用于特征工程） |
| `router_logit`（原始值） | softmax 之前的原始 logit | 消融实验需要 |

### 5.3 增强方案

不修改 `olmoe_router_trace.py` 的现有逻辑，增加一个辅助函数：

```python
def compute_routing_features(
    router_logits: torch.Tensor,  # shape: (num_experts,)
    topk: int,
) -> dict[str, float]:
    """
    从单个 token 的 router logits 计算路由上下文特征。

    返回：
    - router_probabilities: 全部 expert 的 softmax 概率（list[float]）
    - top_k_expert_ids: 选中的 expert IDs（list[int]）
    - top_k_weights: Top-K 的实际权重（list[float]）
    - top1_top2_gap: 第一和第二 expert 的概率差
    - routing_entropy: -Σ p*log(p)
    - router_logits: 原始 logit 值（list[float]）
    """
```

这样消融实验可以调用 `compute_routing_features` 获取完整特征，不影响现有的 trace schema。

---

## 6. runtime/__init__.py 的接口暴露

当前的 `runtime/__init__.py` 直接暴露 `single_gpu` 的函数。重组后需要更新：

```python
# runtime/__init__.py
from .local_test.inference import (
    SingleGPUInferenceResult,
    gpu_environment_snapshot,
    load_model_and_tokenizer,
    run_single_gpu_text_inference,
)
```

保持对外接口不变，只是内部路径变了。

---

## 7. topology/ 和 evaluation/ 的保留

这两个模块保持不变，不需要迁移。但需要确认：

### 7.1 topology/ 的使用者

| 使用者 | 怎么用 |
|---|---|
| `experiments/prerun/*.py` | 从 inventory 读取 model_path |
| `experiments/distributed/*.py` | 从 inventory 读取远端节点信息 |
| `deploy/scripts/*.sh` | 从 inventory 读取 SSH 信息 |
| `distributed_ep/core/placement.py` | 未来可能用 GPU 数量信息 |

### 7.2 evaluation/ 的使用者

| 使用者 | 怎么用 |
|---|---|
| 所有实验脚本 | 写 artifact bundle（summary + config + environment + goal） |
| `gpu_environment_snapshot()` | 被 artifacts.py 调用 |

---

## 8. 迁移执行顺序

### Step 1：创建新目录结构

```
创建：
  src/routesense/runtime/local_test/
  src/routesense/runtime/distributed_ep/core/
  src/routesense/runtime/distributed_ep/adapter/
  experiments/prerun/
  experiments/ablation/configs/
  experiments/distributed/scripts/
```

### Step 2：迁移 runtime

1. 复制 `single_gpu.py` → `local_test/inference.py`
2. 更新 import 路径
3. 更新 `runtime/__init__.py`
4. 删除 `single_gpu.py`

### Step 3：迁移 experiments

1. 复制 4 个 smoke 脚本 → `experiments/prerun/`
2. 简化文件名
3. 更新脚本内 import 路径
4. 移动分布式相关脚本 → `experiments/distributed/`
5. 删除 `experiments/deployment/`

### Step 4：重组 distributed_ep

1. 创建 `core/` 和 `adapter/` 下的空文件
2. 将现有空文件内容移到对应位置
3. 删除旧的空文件（如果内容已迁移）
4. 删除 `scheduler/` 和 `oracle/` 目录

### Step 5：规范 deploy/

1. 整理现有 deploy 脚本
2. 创建 `sync_to_remote.sh`
3. 更新 `requirements-gpu.txt`
4. 更新 README.md

### Step 6：验证

1. 运行 `pytest tests/` 确保测试通过
2. 运行 `experiments/prerun/` 下的脚本确认路径正确
3. 确认所有 import 都能正确解析

---

## 9. 需要更新的测试

### 9.1 test_inventory.py

```python
def test_load_inventory_from_yaml():
    # 加载 hosts.local.yaml，验证 NodeSpec 字段正确

def test_inventory_paths():
    # 验证 resolve_node_rs_root 等路径解析正确
```

### 9.2 test_artifacts.py

```python
def test_write_artifact_bundle():
    # 写入 artifact bundle，验证 4 个文件都存在

def test_environment_snapshot():
    # 收集环境快照，验证必要字段
```

### 9.3 新增：test_import_paths.py

```python
def test_runtime_local_test_imports():
    # 验证 runtime.local_test.inference 可正常 import

def test_runtime_distributed_ep_imports():
    # 验证 core/ 和 adapter/ 下所有模块可正常 import

def test_trace_imports():
    # 验证 trace.olmoe_router_trace 可正常 import
```

---

## 10. 后续开发预留

### 10.1 消融实验模块（优先级最高）

消融实验将在 `experiments/ablation/` 下开发，代码结构见 `routesense-poc-task.md`。
本次重组只需要：
- 确保 `experiments/ablation/configs/` 目录存在
- 确保 `runtime/local_test/inference.py` 的 `load_model_and_tokenizer` 可被消融实验直接调用
- 确保 `trace/olmoe_router_trace.py` 的 `compute_routing_features`（新增）可被消融实验调用

### 10.2 分布式 EP 开发（优先级次高）

分布式 EP 将在 `runtime/distributed_ep/` 下开发。本次重组只需要：
- 确保 `core/` 和 `adapter/` 的空文件已创建
- 确保 `collective.py` 的接口定义已明确
- 确保 `adapter/olmoe_adapter.py` 的接口定义已明确

### 10.3 模型版本对齐

当前代码使用 `OLMoE-1B-7B-0924-Instruct`，消融实验需要 `OLMoE-1B-7B-0125`（基础模型）。
本次重组需要在 `configs/model/` 下新增 `olmoe_1b_7b_0125.yaml`。

---

## 11. 非目标

- 不实现消融实验逻辑（在 `routesense-poc-task.md` 中）
- 不实现分布式 EP 的核心逻辑（后续任务）
- 不修改 POC1/POC2 legacy 代码
- 不创建新的模型加载逻辑
- 不引入 DeepSpeed/Megatron 等框架

---

## 12. 验收检查点

### 结构验收

- [ ] `runtime/local_test/inference.py` 存在且可 import
- [ ] `runtime/distributed_ep/core/` 下有 6 个空文件
- [ ] `runtime/distributed_ep/adapter/` 下有 3 个空文件
- [ ] `experiments/prerun/` 下有 4 个 smoke 脚本
- [ ] `experiments/deployment/` 已删除或归档
- [ ] `scheduler/` 和 `oracle/` 已删除
- [ ] `deploy/sync_to_remote.sh` 存在
- [ ] `configs/model/olmoe_1b_7b_0125.yaml` 存在

### 功能验收

- [ ] `pytest tests/` 全部通过
- [ ] `python experiments/prerun/probe_architecture.py --help` 不报错
- [ ] `python experiments/prerun/single_gpu_smoke.py --help` 不报错
- [ ] `from routesense.runtime.local_test.inference import load_model_and_tokenizer` 可正常 import
- [ ] `from routesense.trace.olmoe_router_trace import collect_olmoe_router_trace` 可正常 import

### 路径验收

- [ ] 无循环 import
- [ ] 无遗留的 `runtime.single_gpu` 引用
- [ ] 无遗留的 `experiments.deployment` 引用

---

## 13. 现有代码逐文件评审

### 13.1 src/routesense/runtime/single_gpu.py（170 行）

**保留部分**：
- `SingleGPUInferenceResult` dataclass — 结构清晰，字段完整
- `gpu_environment_snapshot()` — 环境采集逻辑完善
- `load_model_and_tokenizer()` — 加载逻辑正确，支持 precision/device_map
- `_precision_to_dtype()` — 工具函数，无需改动

**需要修改**：
- `run_single_gpu_text_inference()` 使用了 `use_cache=True` 做 decode，但消融实验只用 `use_cache=False` 做 teacher-forcing。两者不冲突，但需要确保消融实验不复用这个函数。

**代码质量**：良好。无冗余注释，类型标注完整，错误处理到位。

### 13.2 src/routesense/trace/olmoe_router_trace.py（157 行）

**保留部分**：
- `RouterTraceRecord` dataclass — 结构清晰
- `_resolve_layer_id()` — 层解析逻辑健壮
- `collect_olmoe_router_trace()` — 真实 trace 采集完整
- `decode_metadata_tensor_rows()` — 元数据解码
- `summarize_router_trace()` — 汇总逻辑

**需要新增**：
- `compute_routing_features()` — 见第 5.3 节

**代码质量**：良好。唯一瑕疵是 `RouterTraceRecord.expert_rank` 字段跟 `expert_id` 含义重复（都是 expert ID），但改名会影响下游，暂不处理。

### 13.3 src/routesense/topology/inventory.py（174 行）

**保留部分**：全部保留，无需修改。

**代码质量**：良好。`NodeSpec` / `RendezvousSpec` / `Inventory` 结构清晰。`render_torchrun_dry_run()` 生成 torchrun 命令的逻辑完善。

### 13.4 src/routesense/topology/paths.py（47 行）

**保留部分**：全部保留。

**代码质量**：良好。`resolve_rs_root()` 使用 `Path(__file__).resolve().parents[3]` 定位，依赖包结构。如果移动文件位置需要更新 parents 索引。

### 13.5 src/routesense/evaluation/artifacts.py（38 行）

**保留部分**：全部保留。

**代码质量**：良好。`write_artifact_bundle()` 写入 4 个标准文件（summary/config/environment/goal），结构规范。

---

## 14. 命名规范

### 14.1 模块命名

| 规则 | 示例 |
|---|---|
| 使用 snake_case | `worker_loop.py` 不是 `WorkerLoop.py` |
| 文件名反映职责，不用 `utils` / `helpers` / `common` | `inference.py` 不是 `utils.py` |
| 模块名跟主类名对应 | `collective.py` 包含 `CollectiveOps` |

### 14.2 函数命名

| 规则 | 示例 |
|---|---|
| 动词开头 | `load_model()` 不是 `model_loader()` |
| 返回数据用 `compute_` / `collect_` | `compute_routing_features()` |
| 检查/验证用 `verify_` / `check_` | `verify_distributed_vs_single()` |
| 工厂函数用 `create_` / `build_` | `create_scheduler()` |

### 14.3 配置键命名

| 规则 | 示例 |
|---|---|
| 使用 snake_case | `model_id` 不是 `modelId` |
| 布尔值用 `enable_` / `is_` | `enable_calibration` |
| 数量用 `num_` | `num_windows` |
| 路径用 `_path` / `_dir` | `model_path`, `output_dir` |

---

## 15. 错误处理规范

### 15.1 GPU 相关错误

```python
# CUDA 不可用：立即报错，不做 fallback
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this host")

# OOM：打印当前配置和建议，不自动降级
except torch.cuda.OutOfMemoryError:
    print(f"CUDA OOM. Current config: context_length={context_length}")
    print("Suggestion: reduce context_length or num_moe_layers")
    raise

# BF16 在 T4 上：强制转为 FP16 并打印警告
if precision == "bf16" and "T4" in gpu_name:
    print("WARNING: T4 does not support BF16, forcing FP16")
    precision = "fp16"
```

### 15.2 模型加载错误

```python
# 模型路径不存在：打印路径和建议
if not Path(model_path).exists():
    raise FileNotFoundError(f"model path does not exist: {model_path}")

# Router logits 未返回：打印模型 class 和配置
if not router_logits:
    raise RuntimeError(
        f"model {model.__class__.__name__} did not return router_logits. "
        f"Check model.config.output_router_logits = True"
    )
```

### 15.3 分布式错误

```python
# NCCL 初始化失败：打印 rank 和网络信息
except RuntimeError as e:
    raise RuntimeError(
        f"NCCL init failed on rank {rank}. "
        f"master={master_addr}:{master_port}. Error: {e}"
    )

# all_to_all shape 不匹配：打印期望和实际 shape
if send_tensor.shape != expected_shape:
    raise RuntimeError(
        f"all_to_all shape mismatch: got {send_tensor.shape}, "
        f"expected {expected_shape}. Check placement mapping."
    )
```

---

## 16. Git 操作规范

### 16.1 提交信息格式

```
restructure: move smoke scripts to experiments/prerun/
restructure: split runtime into local_test and distributed_ep
restructure: split distributed_ep into core and adapter
restructure: normalize deploy/ as minimal remote environment
restructure: add olmoE_1b_7b_0125 config
feat: add compute_routing_features to trace module
```

### 16.2 提交顺序

建议按 Step 1-6 的顺序逐步提交，每个 Step 一个 commit：

1. `restructure: create new directory structure`
2. `restructure: migrate runtime/single_gpu.py to local_test/inference.py`
3. `restructure: migrate smoke scripts to experiments/prerun/`
4. `restructure: reorganize distributed_ep into core/adapter`
5. `restructure: normalize deploy/ directory`
6. `restructure: add model config and update imports`
7. `chore: remove deprecated directories and update tests`
