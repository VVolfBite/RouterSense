# RouteSense PoC 消融实验实现任务书

## 0. 背景与目标

### 0.1 这个项目要回答什么问题

MoE（混合专家）模型中，路由器为每个 token 选择 Top-K 个专家来处理任务。路由器的分数代表"这个专家跟当前任务有多匹配"，但**不等于**"撤掉这个专家后输出质量会下降多少"。

本 PoC 要回答：**路由器的输出信号（分数、熵、gap 等）能否准确预测单个专家被移除后对 next-token 预测质量的真实影响？**

### 0.2 核心实验方法

对每个 `(文档, 窗口, MoE层, 目标token, 选中的expert)`：

1. 正常前向 → 得到 baseline NLL
2. 只将该 expert 的有效门控权重置零（不重新选择、不重新归一化）
3. 再次完整前向 → 得到 ablated NLL
4. `delta_nll = ablated_nll - baseline_nll`
5. 记录该 expert 的路由上下文特征
6. 最终检验：路由上下文特征能否预测 delta_nll

### 0.3 POC1 做了什么、没做什么

| 已完成 | 未完成 |
|---|---|
| 真实 OLMoE 模型加载（双 T4，balanced） | **真实消融实验（route patch 是空壳）** |
| Router trace 采集（logits/probabilities/topk） | 多窗口 × 多层消融循环 |
| 单 token 路由特征提取 | 路由上下文特征的系统采集 |
| NLL 计算函数 | 策略对比与指标计算 |
| 模型结构发现 | 校准模型训练 |
| CLI 框架 | 数据管道（wikitext） |
| JSON 序列化 | Parquet 输出、断点续跑 |
| | 绘图、报告生成 |

**关键结论**：POC1 只完成了 trace 和 proxy 分析，没有做真正的反事实消融。`route_patch.py` 中的 `RouteAblationContext` 和 `OLMoEAdapter` 都是未完成的骨架。

### 0.4 硬件约束

```
Alibaba Cloud ECS gn6i-c24g1.12xlarge
48 vCPU, 186 GiB RAM
2 × NVIDIA T4 16GB
```

硬约束：
- FP16（不是 BF16），T4 不支持 BF16
- `device_map="balanced"` 或 `device_map="auto"` + `max_memory={0:"14GiB",1:"14GiB","cpu":"140GiB"}`
- 不使用 DDP、FSDP、DeepSpeed
- 不使用 `model.generate()`，只用 `model.forward()`
- `model.eval()` + `torch.inference_mode()`
- batch size = 1（代码必须显式断言）

---

## 1. 目标模型与数据

### 1.1 主模型

```
allenai/OLMoE-1B-7B-0125
```

基础模型（非 Instruct），原因：本实验使用 teacher-forcing 的 next-token NLL，基础语言模型更适合作为统计对象。

### 1.2 数据集

```
Salesforce/wikitext
config: wikitext-2-raw-v1
split: test
```

数据处理规则：
1. 删除空行和无法形成完整窗口的短文档
2. tokenizer 使用 `add_special_tokens=False`
3. 每个文档以固定 seed 抽取或切分窗口
4. 窗口长度 = `context_length + 1`
5. 倒数第二个 token 是目标位置（要预测的 token），最后一个 token 是 ground truth（真实答案）
6. 保存 `document_id`，校准模型的 train/validation/test 切分必须按 document 做，禁止数据泄漏

### 1.3 配置文件

```yaml
# configs/smoke.yaml
model_id: allenai/OLMoE-1B-7B-0125
precision: fp16
num_windows: 8
context_length: 64
num_moe_layers: 3
seed: 42
enable_calibration: false

# configs/poc.yaml
model_id: allenai/OLMoE-1B-7B-0125
precision: fp16
num_windows: 32
context_length: 96
num_moe_layers: 4
seed: 42
enable_calibration: true

# configs/formal.yaml
model_id: allenai/OLMoE-1B-7B-0125
precision: fp16
num_windows: 64
context_length: 96
num_moe_layers: 6
seed: 42
enable_calibration: true
```

不要自动运行 formal。先 smoke，再 POC。

---

## 2. 最终代码结构

```
routesense-poc/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── smoke.yaml
│   ├── poc.yaml
│   └── formal.yaml
├── scripts/
│   ├── bootstrap.sh
│   └── run_smoke.sh
├── src/routesense_poc/
│   ├── __init__.py
│   ├── cli.py              # CLI 入口
│   ├── config.py           # 配置加载
│   ├── environment.py      # 环境检查与架构自检
│   ├── data.py             # 数据加载与窗口切分
│   ├── model_loader.py     # 模型加载
│   ├── moe_inspect.py      # MoE 层自动发现
│   ├── trace.py            # 路由上下文特征采集
│   ├── intervention.py     # 核心：精确消融 hook
│   ├── ablation.py         # 消融循环主逻辑
│   ├── metrics.py          # NLL 计算
│   ├── calibration.py      # 校准模型训练与评估
│   ├── policies.py         # 五种策略定义
│   ├── analysis.py         # 三个核心指标计算
│   ├── plotting.py         # 绘图
│   └── utils.py            # 工具函数
├── tests/
│   ├── test_metrics.py
│   ├── test_intervention.py
│   ├── test_data.py
│   └── test_smoke.py
├── data/
└── outputs/
```

---

## 3. POC1 代码评审：保留 / 修改 / 删除

### 3.1 保留（可直接复用，少量修改）

#### `model_loader.py` → 保留

现有实现质量良好。`load_olmoe_model(config)` 正确处理了：
- precision → dtype 映射
- device_map 设置
- max_memory 过滤
- `model.config.output_router_logits = True`

**需要的修改**：加载后增加架构自检调用（打印 model class、num_layers、num_experts 等），但不改核心逻辑。

#### `metrics.py` → 完全保留

`next_token_nll(logits, input_ids)` 实现正确：
- 支持 torch.Tensor 和 numpy fallback
- 使用 `logits[:, -2, :]` 预测 `input_ids[:, -1]`
- cross_entropy 计算正确

**不需要修改。**

#### `config.py` → 保留框架，扩展字段

`load_config` / `parse_args` / `build_config` / `ensure_output_dir` 框架可复用。

**需要的修改**：
- `RunConfig` 增加字段：`num_windows`, `context_length`, `num_moe_layers`, `enable_calibration`, `dataset_id`, `dataset_config`, `dataset_split`
- CLI 增加 `--resume`, `--verbose`, `--run-id` 参数

#### `serialization.py` → 保留

`save_json` / `load_json` 无需修改。

#### `utils.py` → 保留

`stable_hash` / `ensure_directory` 无需修改。

### 3.2 需要重写

#### `router_trace.py` → 拆分为 `trace.py`

现有的 `collect_router_trace` 做了正确的事情（真实 logits 采集、softmax、topk），但：
- 它采集的特征不够完整（缺少 `top1_top2_gap` 的全 expert 版本、缺少按 expert 拆分的特征）
- `build_batch_routing_summary` 是 POC1 的 bucket 聚合逻辑，**完全不需要**

**重写方向**：新 `trace.py` 只关注目标 token 的路由上下文特征采集，不做 batch 级聚合。具体字段见第 5.6 节。

#### `route_patch.py` → 重写为 `intervention.py`

现有代码的问题：
- `RouteAblationContext` 是空壳，adapter 直接报错
- `OLMoEAdapter` 尝试从 `layer.router_probs` / `layer.topk_weights` 等属性读取状态，但 HuggingFace OLMoE 的 MoE 层**不暴露这些属性**——它在 forward 内部临时计算
- `MockRouteAblationContext` 依赖 mock layer 属性，没有参考价值

**必须从零重写**，具体方案见第 5.7 节。

#### `cli.py` → 重写

现有只有 inspect/trace/ablate 三个命令，且 ablate 返回 `not_implemented`。

**需要重写为完整的 8 个命令**，见第 5.1 节。

#### `schemas.py` → 重写

现有 `RouterTrace` / `AblationResult` 可作为参考但需要调整：
- 删除 `BucketSummary`, `BatchRoutingSummary`（POC1 专用）
- `RouterTrace` 改名为路由上下文记录，增加缺失字段
- `AblationResult` 增加路由上下文特征字段
- 增加 `WindowRecord`, `AblationRecord` 等新结构

#### `model_inspector.py` → 重写为 `moe_inspect.py`

现有实现的评分逻辑过于通用（通过名字猜测 MoE 层），且没有验证 gate 的真实接口。

**重写方向**：明确遍历模型找到包含 `gate` + `experts` 的 MoE block，验证 gate 的 forward 签名，输出层数/专家数/Top-K 等结构化信息。

### 3.3 删除（不需要迁移）

| 文件/模块 | 删除理由 |
|---|---|
| `adapter_registry.py` | 旧 adapter 注册，从未工作 |
| `output_archive.py` | 旧输出归档 |
| `critical_bucket_proxy.py`（实验脚本） | 伪关键性定义，本实验不需要 |
| `proxy_compare.py`（实验脚本） | proxy 策略对比，本实验不需要 |
| `verify_proxy_results.py`（实验脚本） | proxy 结果复核 |
| `BatchRoutingSummary` / `BucketSummary` 数据结构 | POC1 bucket 聚合逻辑 |
| state-only / dependency-only / full 打分体系 | proxy 策略，非本实验内容 |
| `MockRouterTraceProvider` | mock 模式，本实验只跑真实模型 |
| `MockRouteAblationContext` | mock 消融，本实验只跑真实模型 |

---

## 4. 新增模块清单

| 模块 | 职责 | 优先级 |
|---|---|---|
| `environment.py` | GPU 环境检查、CUDA 版本、架构自检打印 | P0 |
| `data.py` | wikitext 加载、窗口切分、document_id 管理 | P0 |
| `moe_inspect.py` | 遍历模型找 MoE 层、验证 gate 接口 | P0 |
| `trace.py` | 目标 token 路由上下文特征采集 | P0 |
| `intervention.py` | OLMoE MoE forward hook + 精确消融 | P0（最关键） |
| `ablation.py` | 多窗口 × 多层 × 多 expert 消融循环 + 断点续跑 | P0 |
| `policies.py` | Full/Random/Raw/Calibrated/Oracle 五种策略 | P1 |
| `analysis.py` | 三个核心指标计算 + 策略对比 | P1 |
| `calibration.py` | HistGradientBoostingRegressor 训练与评估 | P1 |
| `plotting.py` | 三张核心图 | P2 |

---

## 5. 各模块详细规格

### 5.1 `cli.py` — CLI 入口

命令列表：

```bash
routesense-poc doctor              # 环境检查
routesense-poc prepare-data        # 数据准备
routesense-poc inspect-model       # 模型架构自检
routesense-poc collect-trace       # 路由 trace 采集
routesense-poc ablate              # 消融实验
routesense-poc analyze             # 指标计算与分析
routesense-poc calibrate           # 校准模型训练
routesense-poc run-all             # 全流程串联
```

所有命令支持的公共参数：
- `--config`：配置文件路径
- `--run-id`：运行 ID（默认自动生成 timestamp）
- `--output-dir`：输出目录（默认 `outputs/<run_id>`）
- `--seed`：随机种子
- `--resume`：断点续跑（仅 ablate 有效）
- `--verbose`：详细日志

`run-all` 的执行顺序：
```
doctor → prepare-data → inspect-model → collect-trace → ablate → analyze → calibrate（配置开启时）→ analyze（含校准策略）
```

### 5.2 `environment.py` — 环境检查

```python
def run_doctor() -> dict:
    """
    检查并打印：
    - CUDA 是否可用
    - GPU 数量、名称、显存
    - CUDA 版本
    - Python 版本
    - PyTorch 版本
    - transformers 版本
    - 模型文件是否可访问（检查 model_path 是否存在）
    - 数据集是否可加载（dry-run 加载 wikitext 前几条）

    返回 environment dict，保存到 outputs/<run_id>/environment.json

    如果 CUDA 不可用，打印明确错误并退出。
    如果 T4 检测到 BF16 请求，打印警告并强制 FP16。
    """
```

### 5.3 `data.py` — 数据加载与窗口切分

```python
@dataclass
class Window:
    document_id: int
    window_id: int
    input_ids: list[int]        # 长度 = context_length + 1
    target_pos: int             # 倒数第二个 token 的位置
    ground_truth_token_id: int  # 最后一个 token（正确答案）

def load_and_prepare_data(config: RunConfig) -> list[Window]:
    """
    1. 从 HuggingFace 加载 wikitext-2-raw-v1 test split
    2. 过滤空行和短文档
    3. 对每个文档用 tokenizer(add_special_tokens=False) 编码
    4. 用固定 seed 抽取或切分出 num_windows 个窗口
    5. 每个窗口长度 = context_length + 1
    6. target_pos = context_length - 1（倒数第二个 token）
    7. ground_truth_token_id = input_ids[context_length]（最后一个 token）

    返回 Window 列表，保存到 outputs/<run_id>/selected_windows.parquet

    重要：必须保存 document_id，用于后续校准模型的 train/test 切分。
    """
```

### 5.4 `model_loader.py` — 模型加载

基于 POC1 的 `load_olmoe_model` 修改：

```python
def load_model(config: RunConfig) -> tuple[model, tokenizer]:
    """
    与 POC1 相同逻辑，增加：
    1. 加载后调用 print_architecture_summary(model, tokenizer)
    2. 断言 model.config.output_router_logits 可设为 True
    3. 返回 (model, tokenizer)
    """

def print_architecture_summary(model, tokenizer):
    """
    打印并保存到 outputs/<run_id>/model_inspection.json：
    - model class name
    - transformers version
    - num_hidden_layers
    - num_experts（从 config 或遍历得到）
    - num_experts_per_tok (Top-K)
    - norm_topk_prob
    - all MoE layer paths
    - GPU names / memory / CUDA version
    - device_map
    """
```

### 5.5 `moe_inspect.py` — MoE 层发现

```python
@dataclass
class MoELayerSpec:
    layer_index: int          # 在全部 MoE 层中的序号（0, 1, 2, ...）
    module_path: str          # 模型中的完整路径，如 "model.layers.3.mlp"
    num_experts: int          # 该层的专家数量
    top_k: int                # Top-K 值
    norm_topk_prob: bool      # 是否对 Top-K 权重归一化
    gate_module_path: str     # gate 子模块路径
    experts_module_path: str  # experts 子模块路径

def discover_moe_layers(model) -> list[MoELayerSpec]:
    """
    遍历 model.named_modules()，找到所有包含 gate + experts 的 MoE block。

    对于每个候选：
    1. 检查 hasattr(module, 'gate') 和 hasattr(module, 'experts')
    2. 获取 gate 的 forward 签名，确认它接受 hidden_states 并返回 router_logits
    3. 获取 experts 的数量（len(module.experts) 或 module.num_experts）
    4. 从 model.config 获取 top_k 和 norm_topk_prob

    如果 gate 的接口不符合预期，报错并输出：模块路径、类名、forward 签名、修复建议。

    选择 num_moe_layers 个层用于实验：
    - 如果 num_moe_layers <= 总 MoE 层数：均匀选取（首、尾、中间）
    - 如果 num_moe_layers > 总 MoE 层数：使用全部

    保存到 outputs/<run_id>/moe_layers.json
    """
```

### 5.6 `trace.py` — 路由上下文特征采集

```python
@dataclass
class RoutingContext:
    """单个 expert 在特定 (window, layer) 下的路由上下文"""
    run_id: str
    document_id: int
    window_id: int
    layer_id: int             # MoE 层序号
    layer_path: str           # MoE 层路径
    token_pos: int            # 目标 token 位置
    expert_id: int            # 该 expert 的 ID
    topk_rank: int            # 该 expert 在 Top-K 中的名次（0 或 1）
    router_logit: float       # 该 expert 的原始 logit
    router_probability: float # softmax 后该 expert 的概率（全部 expert 中）
    effective_gate_weight: float  # Top-K 后的实际门控权重
    top1_top2_gap: float      # 第一和第二 expert 的概率差
    routing_entropy: float    # 全部 expert 概率分布的熵
    baseline_nll: float       # 正常前向的 NLL（后续由 ablation 填充）
    ablated_nll: float        # 消融后的 NLL（后续由 ablation 填充）
    delta_nll: float          # ablated - baseline（后续由 ablation 填充）

def collect_routing_context(
    model, tokenizer, window: Window, moe_layer: MoELayerSpec
) -> list[RoutingContext]:
    """
    对目标 token 在指定 MoE 层采集路由上下文。

    步骤：
    1. model forward（output_router_logits=True, use_cache=False）
    2. 从 outputs.router_logits[moe_layer.layer_index] 取出目标 token 的 logits
       shape: (seq_len, num_experts)，取 logits[target_pos, :]
    3. softmax → 全部 expert 的概率分布
    4. 计算 routing_entropy = -Σ p*log(p)
    5. topk(k=top_k) → 选中的 expert IDs + 权重
    6. top1_top2_gap = probs[0] - probs[1]（概率最高的两个之差）
    7. 对每个选中的 expert，记录：
       - expert_id
       - topk_rank（0 或 1）
       - router_logit（原始 logit 值）
       - router_probability（softmax 后的概率）
       - effective_gate_weight（Top-K 后的权重；如果 norm_topk_prob=True，则归一化后的值）

    返回 list[RoutingContext]，长度 = top_k（通常是 2）
    """
```

### 5.7 `intervention.py` — 精确消融 hook（最关键模块）

这是整个实验的基石。必须精确、安全、可验证。

```python
@contextmanager
def ablate_route(
    model,
    moe_layer_spec: MoELayerSpec,
    token_pos: int,
    expert_rank: int,          # 0 或 1（在 Top-K 中的名次）
    expected_expert_id: int,   # 校验用：该 rank 对应的 expert ID
):
    """
    Context manager：进入时安装消融 hook，退出时恢复原始 forward。

    实现方案：包装目标 MoE block 的 forward 方法。

    进入时（__enter__）：
    1. 通过 moe_layer_spec.module_path 找到目标 MoE 模块
    2. 保存该模块的原始 forward 方法
    3. 安装包装后的 forward：

       def wrapped_forward(hidden_states):
           # 1. 用原始 gate 计算 router_logits
           router_logits = original_gate(hidden_states)

           # 2. softmax + topk（与原始逻辑完全一致）
           routing_weights = softmax(router_logits, dim=-1)
           topk_weights, topk_indices = topk(routing_weights, k=top_k, dim=-1)
           if norm_topk_prob:
               topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

           # 3. 校验
           assert topk_indices[flat_target_pos, expert_rank] == expected_expert_id

           # 4. 消融：只修改目标 token 的目标 expert 的权重
           topk_weights = topk_weights.clone()
           topk_weights[flat_target_pos, expert_rank] = 0.0
           # 不改 topk_indices（不重新选择 expert）
           # 不归一化其他 expert 的权重
           # 不改其他 token 的权重

           # 5. 用修改后的 weights + 原始 indices 继续执行
           #    调用原有的 expert 计算逻辑
           result = compute_expert_output(hidden_states, topk_weights, topk_indices)
           return result

    退出时（__exit__）：
    1. 恢复原始 forward

    关键约束：
    - 只改一个 token 的一个 expert 的权重
    - 不改 Top-K 选择（topk_indices 不变）
    - 不归一化剩余权重
    - 不改其他 token 的路由
    - 不改其他 MoE 层
    """
```

**正确性检查（必须全部通过）**：

```python
def verify_intervention_correctness(model, tokenizer, window, moe_layer_spec):
    """
    8 项检查：

    1. 无消融时（不进入 context），wrapped logits 与原始 logits torch.allclose
    2. selected expert IDs 与基线完全一致
    3. 只有一个目标 route weight 变为 0
    4. 其他 weights 与基线一致（torch.allclose）
    5. 未重归一化（其他 weights 之和 != 1，除非原本就恰好为 1）
    6. 其他 MoE 层的 gate 输出与基线一致
    7. context 退出后，下一次正常前向恢复基线（logits allclose）
    8. 如果架构接口无法精确 patch，必须失败并报错，不做近似消融
    """
```

**实现注意**：
- HuggingFace OLMoE 的 MoE forward 内部结构需要阅读源码确认变量名和 shape
- `flat_target_pos` 是目标 token 在 flatten 后的索引（batch_size=1 时 = token_pos）
- 必须对 batch size 做显式断言：`assert batch_size == 1`
- 包装的 forward 必须能正确传递 gradient（虽然 inference_mode 下不需要）

### 5.8 `ablation.py` — 消融循环

```python
@dataclass
class AblationRecord:
    """单条消融记录"""
    run_id: str
    document_id: int
    window_id: int
    layer_id: int
    layer_path: str
    token_pos: int
    expert_id: int
    topk_rank: int
    router_logit: float
    router_probability: float
    effective_gate_weight: float
    top1_top2_gap: float
    routing_entropy: float
    baseline_nll: float
    ablated_nll: float
    delta_nll: float

def run_ablation(
    config: RunConfig,
    model, tokenizer,
    windows: list[Window],
    moe_layers: list[MoELayerSpec],
    output_dir: Path,
    resume: bool = False,
) -> list[AblationRecord]:
    """
    主循环：

    total_tasks = len(windows) × len(moe_layers) × top_k
    打印任务量和预估时间。

    对每个 (window, moe_layer)：
    1. 先跑一次正常 forward → baseline_nll（用 metrics.next_token_nll）
    2. 采集路由上下文特征（用 trace.collect_routing_context）
    3. 对每个选中的 expert（top_k 个）：
       a. with ablate_route(model, moe_layer, target_pos, rank, expert_id):
              ablated_output = model(...)
          ablated_nll = metrics.next_token_nll(ablated_logits, input_ids)
       b. delta_nll = ablated_nll - baseline_nll
    4. 写入 checkpoint（原子写入，防止中断丢失）
    5. 打印进度：已完成 / 总数 / 平均每次耗时 / 预计剩余时间

    resume=True 时：
    - 读取已有的 checkpoint 文件
    - 跳过已完成的 (window_id, layer_id) 组合
    - 打印 "跳过 X 个已完成任务"

    输出：
    - outputs/<run_id>/ablation_results.parquet（全部记录）
    - outputs/<run_id>/ablation_checkpoint.jsonl（逐条追加写入）

    OOM 处理：
    - 捕获 CUDA OOM 异常
    - 打印当前配置（context_length、num_moe_layers）和建议
    - 不自动切换量化
    - 不自动跳过
    """
```

### 5.9 `policies.py` — 五种策略

```python
def select_deferrable_expert(
    records: list[AblationRecord],  # 同一个 (window_id, layer_id) 组
    strategy: str,
    calibrator=None,               # 校准模型（仅 calibrated 策略使用）
) -> int | None:
    """
    给定一个 (window, layer) 组的全部 expert 消融记录，选择"最适合取消"的 expert。

    策略定义：

    "full"：
        不取消任何 expert。返回 None。质量基线。

    "random"：
        在同组 Top-K routes 中等概率随机选一个。
        实际计算时用该组 delta_nll 的平均值代表随机期望。

    "raw_routing"：
        选择 effective_gate_weight 最小的 route。
        tie-breaking：选 topk_rank 更低者（即名次更靠后的）。

    "calibrated"：
        用校准模型预测每个 expert 的 delta_nll。
        选择预测 delta_nll 最小的 route。
        如果 calibrator 为 None，报错。

    "oracle"：
        选择真实 delta_nll 最小的 route。
        理论上限，不可在实际部署中使用。

    返回被选中的 expert_id，或 None（full 策略）。
    """
```

### 5.10 `analysis.py` — 三个核心指标

```python
def compute_metrics(
    all_records: list[AblationRecord],
    calibrator=None,
) -> dict:
    """
    按 (window_id, layer_id) 分组，对每组应用五种策略。

    === 指标 A: Pairwise Ranking Accuracy ===

    对同一个 (window, layer) 中任意一对已选 expert：
    - 真实顺序由 delta_nll 给出
    - 预测顺序由策略的关键性估计给出
    - 两者一致则为正确

    Pairwise Ranking Accuracy = 正确排序的 expert 对数量 / 有效 expert 对总数

    随机基线约为 0.5。
    当 abs(delta_nll(A) - delta_nll(B)) < threshold（如 1e-6）时，
    跳过该 pair，但必须报告跳过比例。

    === 指标 B: Mean ΔNLL 与 P95 ΔNLL ===

    对每个 (window, layer)，每个策略选择一个最适合取消的 route：
    - Full → 不取消，delta_nll = 0
    - Random → 该组 delta_nll 的平均值
    - Raw Routing → 选 effective_gate_weight 最小的那个 expert 的 delta_nll
    - Calibrated → 选校准模型预测最小的那个 expert 的 delta_nll
    - Oracle → 选真实 delta_nll 最小的那个 expert 的 delta_nll

    报告：Mean ΔNLL, Median ΔNLL, P95 ΔNLL

    主比较：
    - Raw Routing vs Random
    - Calibrated vs Random
    - Calibrated vs Raw Routing
    - Oracle vs Calibrated

    === 指标 C: Quality-Constrained Deferrable Ratio ===

    对每个策略选出的 route，给定质量预算 ε：
    safe = (delta_nll <= ε)
    Deferrable Ratio = safe groups / all (window, layer) groups

    ε ∈ {0.01, 0.05, 0.10}

    注意：当前每个 group 只尝试取消一个 route，
    所以这是"可安全取消一个 expert request 的 group 比例"，不是通信量节省比例。
    报告必须明确这一限制。

    === 附加指标（保存但不作为主结论）===
    - KL divergence（消融前后 logits 分布的 KL）
    - top-1 prediction flip（消融后 top-1 预测是否改变）
    - logit L2 distance（消融前后 logits 的 L2 距离）

    输出：outputs/<run_id>/summary.json
    """
```

### 5.11 `calibration.py` — 校准模型

```python
def train_calibrator(
    records: list[AblationRecord],
    config: RunConfig,
    output_dir: Path,
) -> HistGradientBoostingRegressor:
    """
    特征：
    - effective_gate_weight
    - router_probability
    - topk_rank
    - top1_top2_gap
    - routing_entropy
    - layer_id
    - expert_id（必须 one-hot 或类别编码，不要当成连续数值）

    目标：delta_nll

    数据切分：
    - 按 document_id 切分，train/validation/test = 70/15/15
    - 禁止按 record 级别切分（同一文档的不同窗口不能分到 train 和 test）

    保存：
    - outputs/<run_id>/calibrator.joblib
    - outputs/<run_id>/calibration_metrics.json（MAE、R² 等）
    - outputs/<run_id>/calibration_feature_importance.csv
    """

def evaluate_calibrator(
    calibrator, test_records: list[AblationRecord]
) -> dict:
    """
    在 test split 上评估校准模型的预测质量。
    返回 MAE、R²、Spearman 相关等指标。
    """
```

### 5.12 `plotting.py` — 绘图

```python
def plot_ranking_accuracy(summary: dict, output_dir: Path):
    """
    柱状图：各策略的 Pairwise Ranking Accuracy
    X 轴：策略名称（Random, Raw Routing, Calibrated, Oracle）
    Y 轴：Accuracy
    随机基线画一条水平虚线在 0.5
    保存：outputs/<run_id>/figures/ranking_accuracy.png
    """

def plot_policy_delta_nll(summary: dict, output_dir: Path):
    """
    箱线图或分组柱状图：各策略的 ΔNLL 分布
    X 轴：策略名称
    Y 轴：ΔNLL
    标注 Mean 和 P95
    保存：outputs/<run_id>/figures/policy_delta_nll.png
    """

def plot_deferrable_ratio(summary: dict, output_dir: Path):
    """
    折线图：不同 ε 下各策略的 Deferrable Ratio
    X 轴：ε 值（0.01, 0.05, 0.10）
    Y 轴：Deferrable Ratio
    每条线一个策略
    保存：outputs/<run_id>/figures/deferrable_ratio.png
    """
```

---

## 6. 输出文件规范

每次运行在 `outputs/<run_id>/` 下创建：

```
outputs/<run_id>/
├── manifest.json               # 运行元信息
├── environment.json            # GPU/CUDA/Python/模型信息
├── model_inspection.json       # MoE 架构信息
├── moe_layers.json             # 选中的 MoE 层列表
├── selected_windows.parquet    # 选中的数据窗口
├── router_trace.parquet        # 路由上下文特征
├── ablation_results.parquet    # 消融结果
├── ablation_checkpoint.jsonl   # 断点续跑用
├── summary.json                # 三个核心指标
├── report.md                   # 可读报告
├── calibrator.joblib           # 校准模型（可选）
├── calibration_metrics.json    # 校准评估指标（可选）
├── calibration_feature_importance.csv
├── figures/
│   ├── ranking_accuracy.png
│   ├── policy_delta_nll.png
│   └── deferrable_ratio.png
└── logs/
```

### manifest.json 内容

```json
{
    "run_id": "20250101_120000",
    "timestamp": "2025-01-01T12:00:00Z",
    "git_commit": "abc1234 or null",
    "model_id": "allenai/OLMoE-1B-7B-0125",
    "model_revision": "...",
    "dataset_id": "Salesforce/wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "dataset_split": "test",
    "config": { /* 完整 config */ },
    "python_version": "3.10.x",
    "torch_version": "2.x.x",
    "transformers_version": "4.x.x",
    "cuda_version": "12.x",
    "gpu_info": ["NVIDIA T4", "NVIDIA T4"],
    "device_map": "balanced",
    "seed": 42
}
```

### report.md 内容要求

- 清楚区分"观察到的现象"与"未验证的假设"
- 列出三个核心指标的完整数值
- 各策略对比表格
- 引用三张图
- 明确说明：当前 PoC 每个 group 只取消一个 expert，不代表最终通信节省比例
- 明确说明：当前"取消 route"代表"请求暂缓后最终不释放"的质量上限近似

---

## 7. 单元测试要求

### `tests/test_metrics.py`

```
- test_nll_basic: 已知 logits 和 target，NLL 值正确
- test_nll_perfect_prediction: logits 集中在 target 上，NLL 接近 0
- test_nll_uniform_prediction: logits 均匀分布，NLL = log(vocab_size)
- test_nll_batch_mismatch: batch 维度不匹配时报错
- test_nll_short_sequence: sequence_length < 2 时报错
```

### `tests/test_intervention.py`

```
- test_ablate_zeros_target_weight: 消融后目标 expert 的权重为 0
- test_ablate_preserves_other_weights: 其他 expert 的权重不变
- test_ablate_preserves_other_tokens: 其他 token 的路由不变
- test_ablate_preserves_selected_experts: Top-K 选择不变（只改权重，不改选谁）
- test_ablate_no_renormalization: 其他权重之和不等于 1（未归一化）
- test_ablate_context_exit_restores: context 退出后恢复正常 forward
- test_ablate_wrong_expert_id_fails: expected_expert_id 不匹配时报错
- test_ablate_out_of_range_rank_fails: expert_rank 超范围时报错
- test_baseline_logits_match_without_ablation: 不消融时 wrapped forward 输出与原始 allclose

注意：这些测试需要加载真实模型。如果 CI 环境无 GPU，标记为 @pytest.mark.skipif。
可以只跑 1 个窗口 × 1 层 × 1 expert 的快速验证。
```

### `tests/test_data.py`

```
- test_window_length: 每个窗口长度 = context_length + 1
- test_target_position: target_pos = context_length - 1
- test_no_empty_windows: 不含空窗口
- test_document_id_preserved: 每个窗口有 document_id
- test_seed_reproducibility: 同 seed 同 config 产出相同窗口
- test_different_seeds_differ: 不同 seed 产出不同窗口
```

### `tests/test_smoke.py`

```
- test_smoke_end_to_end: 用 1 个窗口 × 1 层 × Top-K 跑完整流程
  （需要 GPU，标记 @pytest.mark.gpu）
- test_manifest_created: 运行后 manifest.json 存在且字段完整
- test_ablation_results_non_empty: ablation_results.parquet 非空
- test_summary_has_all_metrics: summary.json 包含三个核心指标
```

---

## 8. 实现阶段与验收检查点

### Phase 0：环境与结构自检

完成 `doctor`、`inspect-model`。

验收：
- [ ] 双 T4 被正确识别
- [ ] OLMoE 可下载、加载
- [ ] MoE 层、专家数、Top-K、gate 接口可发现
- [ ] 可拿到 Router logits（做一次 forward + output_router_logits=True）
- [ ] environment.json 和 model_inspection.json 生成正确

### Phase 1：最小 Trace

对 1 个窗口、1 个 MoE 层、目标 token 输出：

验收：
- [ ] router_trace.parquet 包含正确字段
- [ ] 每条记录有：expert_id, topk_rank, router_probability, effective_gate_weight, top1_top2_gap, entropy
- [ ] 数值合理（概率之和 ≈ 1，entropy > 0）

### Phase 2：单 route 精确消融

对同一窗口、层、一个 route：

验收：
- [ ] intervention.py 通过全部 8 项正确性检查
- [ ] 输出 baseline_nll、ablated_nll、delta_nll
- [ ] context 退出后恢复基线（验证第 7 项检查）
- [ ] delta_nll 不为 0（消融确实影响了输出）

### Phase 3：Smoke 运行

运行 `8 windows × 3 MoE layers × actual Top-K`。

验收：
- [ ] ablation_results.parquet 非空，行数 = 8 × 3 × top_k
- [ ] summary.json 包含三个核心指标
- [ ] 三张图生成
- [ ] report.md 可读
- [ ] `pytest` 全部通过

### Phase 4：PoC 运行

运行 `32 windows × 4 MoE layers × actual Top-K`。

验收：
- [ ] 先跑 raw routing（无校准）
- [ ] 成功后再跑 calibration
- [ ] 校准模型的 MAE 和 R² 合理
- [ ] 五个策略的指标对比表格完整
- [ ] Calibrated vs Raw Routing 的提升可量化

### Phase 5：Formal 准备

验收：
- [ ] formal.yaml 写好
- [ ] 命令可以运行
- [ ] **不得未经确认自动跑 formal**

---

## 9. 术语固定

在整个代码和文档中，术语必须一致：

| 术语 | 含义 | 禁止使用 |
|---|---|---|
| 路由上下文（routing context） | 专家选择、门控分布、相对偏好、路由不确定性和层位置等 | "路由特征"、"路由信号" |
| 实际关键性（realized criticality） | 反事实消融后测得的 delta_nll | "真实贡献"、"真实重要性" |
| 请求关键性估计（criticality estimate） | 由路由上下文预测得到的排序结果 | "关键性分数" |
| delta_nll | ablated_nll - baseline_nll | "损失变化"、"质量差" |
| effective_gate_weight | Top-K 后实际送入 expert 计算的权重 | "门控值" |
| router_probability | 对全部 expert softmax 后该 expert 的概率 | "路由概率" |

不要把 Router 的单一分数称为"真实贡献"。

---

## 10. 依赖

```
transformers>=4.45.0,<5
accelerate
torch
datasets
huggingface_hub
safetensors
numpy
pandas
pyarrow
scipy
scikit-learn
matplotlib
pyyaml
tqdm
pytest
```

---

## 11. 非目标（不要做的事）

- 不做网络 QoS、包优先级、P4、ECN、在网设备
- 不训练或微调 Router
- 不改 Top-K 选择
- 不将结果解释为端到端通信加速
- 不实现分布式、NCCL、SmartNIC、DPU
- 不做 proxy 分析或伪关键性定义
- 不用 mock 模式（只跑真实模型）
- 不用 model.generate()
- 不自动运行 formal 配置
