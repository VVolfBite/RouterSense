# POC1 Refine: 代码收口 + 信号发掘

## 1. 背景

POC1 消融链路已完成首版闭环：
- 消融流水线（intervention → ablation → analysis → calibration）可端到端运行
- 32 windows × 4 layers × top-8 = 1024 条消融记录已归档
- 参数搜索（HistGradientBoostingRegressor + 33 个特征子集）已跑完
- 结论：routing context 特征无法有效预测 delta_nll，topk_rank 是唯一不劣于随机的特征

当前问题：
1. 代码有遗留的死文件和结构问题需要收口
2. 实验结论的信号太弱，需要设计更有效的实验来发掘 routing 价值

---

## 2. 代码审查发现

### 2.1 必须删除的死文件

以下文件引用的 schema 类（`RouterTrace`、`BatchRoutingSummary`、`BucketSummary`、`MoELayerInfo`）已从 `schemas.py` 中移除，导入必然报错：

| 文件 | 原因 |
|------|------|
| `router_trace.py` (226 行) | 旧 mock trace 实现，引用已删除的 `RouterTrace`/`BatchRoutingSummary`/`BucketSummary` |
| `adapter_registry.py` (46 行) | 旧 adapter 协议，`UnsupportedOLMoEAdapter` 已被 `intervention.py` 替代 |
| `model_inspector.py` (77 行) | 旧模型探测，引用已删除的 `MoELayerInfo`，被 `moe_inspect.py` 替代 |
| `output_archive.py` (30 行) | 旧归档逻辑，不再被 `cli.py` 调用 |

以下文件虽然能用但属于遗留冗余：

| 文件 | 原因 |
|------|------|
| `utils.py` (18 行) | `stable_hash` 和 `ensure_directory` 未被任何活跃模块引用 |

### 2.2 必须修复的问题

**问题 1：`environment.py` 静默降级 bf16 → fp16**

```python
# 第 15-16 行
if config.precision.lower() in {"bf16", "bfloat16"}:
    config.precision = "fp16"
```

这是一个危险的隐式行为。用户配置 bf16 时，精度被悄悄降级，没有任何警告。
修复方案：改为抛出明确提示或直接删除这段降级逻辑（bf16 在 4090D 上可用）。

**问题 2：`ablation.py` 重复前向计算**

`trace.py` 的 `collect_routing_context()` 内部已经做了一次 `model forward`（第 21 行），
`ablation.py` 的 `run_ablation()` 又做了一次 baseline forward（第 52 行）。
同一 window 同一 layer 的前向传播被执行了两次。

修复方案：让 `collect_routing_context()` 返回 `(contexts, baseline_nll)` 或直接复用已有的 `baseline_nll`。
当前 trace 已经在 `RoutingContext` 里记录了 `baseline_nll`，但 ablation.py 没有复用。

**问题 3：`policies.py` 中 raw_routing 与 effective_gate_weight_min 完全重复**

```python
# 第 33-34 行
if strategy == "raw_routing":
    return min(records, key=lambda r: (r.effective_gate_weight, r.topk_rank)).expert_id
if strategy == "effective_gate_weight_min":
    return min(records, key=lambda r: (r.effective_gate_weight, r.topk_rank)).expert_id
```

完全相同的排序逻辑。`raw_routing` 本意是"路由器原始排序"，但实现时退化成了 effective_gate_weight 排序。
修复方案：要么让 `raw_routing` 使用真正的路由器原始排序（如按 router_probability），要么显式标注二者等价并只保留一个。

**问题 4：特征定义三处重复**

同一个特征到提取函数的映射出现在三个地方：
- `analysis.py` 的 `FEATURES` dict
- `calibration.py` 的 `FEATURE_COLUMNS` dict
- `policies.py` 的 `_records_to_features` 中的 `feature_map`

三处的 key 集合和 lambda 实现略有不同，容易不一致。

修复方案：统一到一个 `features.py` 模块，三处引用同一个定义。

**问题 5：`analysis.py` 硬编码策略名**

```python
# 第 192 行
raw_pairwise = factor_diagnostics["effective_gate_weight"]["group_pairwise"]
```

这行假设 `effective_gate_weight` 一定存在于 FEATURES dict 中。如果未来重命名特征，这里会静默失败。
修复方案：改为从 factor_diagnostics 取第一个或指定一个 primary_feature。

**问题 6：`config.py` 的 DEFAULT_CONFIG_PATH 引用不存在的 poc1.yaml**

```python
# 第 12 行
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poc1.yaml"
```

`poc1.yaml` 包含 `layer: auto` 和 `rank: 0` 等不在 `RunConfig` schema 中的字段，从 YAML 加载时会报 `unexpected keyword argument`。
修复方案：删除 `poc1.yaml`（已被 `poc.yaml` 替代），将默认路径指向 `poc.yaml`。

**问题 7：`poc1.yaml` 包含不在 schema 中的字段**

```yaml
layer: auto
rank: 0
```

这两个字段在 `RunConfig` dataclass 中不存在，`RunConfig.from_dict(payload)` 会抛出 `TypeError`。

### 2.3 可改进项（非阻塞）

**a. `calibration.py` 缺少交叉验证**

当前用固定 70/15/15 划分。1024 条记录下验证集只有 160 条，MAE 的方差很大。
建议：加一个简单的 3-fold 或 5-fold cross-validation 来得到更稳定的评估。

**b. `analysis.py` 中 pairwise accuracy 的 tie-breaking 逻辑**

```python
# 第 100-101 行
truth_order = sorted(group_records, key=lambda r: (r.delta_nll, r.topk_rank))
predicted_order = sorted(group_records, key=lambda r: (feature_fn(r), r.topk_rank), reverse=reverse)
```

当两个专家的 delta_nll 相同时用 topk_rank 做 tie-breaker，但当 feature 值也相同时仍用 topk_rank，
这可能在 rank=0 和 rank=1 之间引入系统性偏差。建议改为随机 tie-breaking 或跳过此类 pair。

**c. `ablation.py` 进度日志可以更丰富**

当前只输出 `avg` 和 `eta`。建议补充：当前 window 的 delta_nll 范围、已完成 groups 数。

**d. 测试覆盖**

- `test_intervention.py` 是 skip 状态（需要 GPU），应补充一个 CPU-side 的 mock test 验证 forward patching 逻辑
- 缺少 `policies.py` 的单元测试
- 缺少 `moe_inspect.py` 的单元测试

---

## 3. 代码结构收口

### 3.1 目标目录结构（分层）

当前 18 个文件平铺在一个目录下，职责混杂。按「数据定义 / 模型交互 / 实验执行 / 结果分析」四层重组：

```
routesense_poc1/
├── __init__.py                 # 包入口（re-export 各层公开接口）
├── cli.py                      # CLI 入口（唯一顶层脚本）
│
├── core/                       # 数据定义层（纯数据，零 GPU 依赖）
│   ├── __init__.py
│   ├── schemas.py              # 所有 dataclass（Window, AblationRecord, RoutingContext...）
│   ├── features.py             # 新增：统一特征定义（FEATURE_EXTRACTORS, FEATURE_ORIENTATION）
│   └── serialization.py        # JSON 读写工具
│
├── runtime/                    # 模型交互层（需要 GPU，与模型直接打交道）
│   ├── __init__.py
│   ├── model_loader.py         # 模型加载（precision, device_map, max_memory）
│   ├── moe_inspect.py          # MoE 层自动发现（named_modules 遍历）
│   ├── intervention.py         # forward patching（核心消融机制）
│   ├── trace.py                # routing context 采集（model forward → router_logits → features）
│   └── metrics.py              # NLL 计算（PyTorch + NumPy 双路径）
│
├── experiment/                 # 实验执行层（流水线编排，调度各层协作）
│   ├── __init__.py
│   ├── config.py               # 配置加载（YAML → RunConfig）
│   ├── data.py                 # 数据加载 + 窗口构建（JSONL/HF → Window 列表）
│   ├── ablation.py             # 消融主循环（复用 trace 的 baseline_nll）
│   └── environment.py          # GPU 环境探测（删除 bf16 降级）
│
└── analysis/                   # 结果分析层（纯 CPU，不依赖模型）
    ├── __init__.py
    ├── policies.py             # 策略选择（12 种策略 + calibrated）
    ├── analysis.py             # 策略对比 + 因子诊断 + 报告生成
    ├── calibration.py          # HistGradientBoostingRegressor 参数学习
    └── plotting.py             # 可视化（ranking_accuracy / policy_delta / deferrable_ratio）
```

**分层原则：**
- `core/` 只定义数据结构，不 import torch
- `runtime/` 是唯一 import torch + transformers 的地方
- `experiment/` 编排 runtime + core，是"胶水层"
- `analysis/` 纯 Python 数值计算，可在无 GPU 机器上独立运行

**import 方向严格单向：**
```
cli.py → experiment/ → runtime/ → core/
       → analysis/  → core/
```

禁止反向引用（如 `core/` 不能 import `runtime/`，`analysis/` 不能 import `runtime/`）。

### 3.2 删除清单

删除 5 个死文件 + 1 个废弃配置：

```
# 死文件（引用已删除的 schema 类，import 会直接报错）
routesense_poc1/router_trace.py       # 226 行，旧 mock trace
routesense_poc1/adapter_registry.py   # 46 行，旧 adapter 协议
routesense_poc1/model_inspector.py    # 77 行，旧模型探测
routesense_poc1/output_archive.py     # 30 行，旧归档逻辑
routesense_poc1/utils.py              # 18 行，无引用方

# 废弃配置
configs/poc1.yaml                     # 包含 layer/rank 等不在 schema 中的字段
```

### 3.3 迁移清单

所有文件只移动位置 + 修改 import 路径，**不改功能逻辑**（bug 修复在 Phase 1 单独做）。

| 原位置 | 新位置 | import 修改 |
|--------|--------|-------------|
| `schemas.py` | `core/schemas.py` | 无内部依赖，无需改 |
| `serialization.py` | `core/serialization.py` | 无内部依赖，无需改 |
| `features.py`（新增） | `core/features.py` | `from .schemas import AblationRecord` |
| `model_loader.py` | `runtime/model_loader.py` | `from .schemas import RunConfig` → `from ..core.schemas import RunConfig` |
| `moe_inspect.py` | `runtime/moe_inspect.py` | `from .schemas` → `from ..core.schemas` |
| `intervention.py` | `runtime/intervention.py` | 无内部依赖，无需改 |
| `trace.py` | `runtime/trace.py` | `from .metrics` → `from .metrics`; `from .schemas` → `from ..core.schemas` |
| `metrics.py` | `runtime/metrics.py` | 无内部依赖，无需改 |
| `config.py` | `experiment/config.py` | `from .schemas` → `from ..core.schemas` |
| `data.py` | `experiment/data.py` | `from .schemas` → `from ..core.schemas` |
| `ablation.py` | `experiment/ablation.py` | `from .intervention` → `from ..runtime.intervention`; `from .trace` → `from ..runtime.trace`; `from .schemas` → `from ..core.schemas` |
| `environment.py` | `experiment/environment.py` | `from .schemas` → `from ..core.schemas` |
| `policies.py` | `analysis/policies.py` | `from .schemas` → `from ..core.schemas`; `from .features` → `from ..core.features` |
| `analysis.py` | `analysis/analysis.py` | `from .schemas` → `from ..core.schemas`; `from .policies` → `from .policies`; `from .features` → `from ..core.features` |
| `calibration.py` | `analysis/calibration.py` | `from .schemas` → `from ..core.schemas`; `from .features` → `from ..core.features` |
| `plotting.py` | `analysis/plotting.py` | 无内部依赖，无需改 |
| `cli.py` | `cli.py`（不动） | `from .ablation` → `from .experiment.ablation` 等 |
| `__init__.py` | `__init__.py`（不动） | 更新所有 re-export 路径 |

每个子包的 `__init__.py` 做 re-export：

```python
# core/__init__.py
from .schemas import AblationRecord, MoELayerSpec, RoutingContext, RunConfig, Window
from .features import FEATURE_EXTRACTORS, FEATURE_ORIENTATION
from .serialization import save_json, load_json

# runtime/__init__.py
from .model_loader import load_olmoe_model, ModelLoadError
from .intervention import RouteAblationContext, RoutePatchError
from .trace import collect_routing_context
from .moe_inspect import discover_moe_layers
from .metrics import next_token_nll

# experiment/__init__.py
from .config import build_config, parse_args, finalize_config, ensure_output_dir
from .data import load_and_prepare_data
from .ablation import run_ablation
from .environment import run_doctor

# analysis/__init__.py
from .analysis import analyze_records, write_report
from .calibration import train_calibrator, evaluate_calibrator
from .policies import select_deferrable_expert
from .plotting import plot_ranking_accuracy, plot_policy_delta_nll, plot_deferrable_ratio
```

### 3.4 新增文件：`core/features.py`

```python
from __future__ import annotations
from typing import Callable
from .schemas import AblationRecord

FEATURE_EXTRACTORS: dict[str, Callable[[AblationRecord], float]] = {
    "effective_gate_weight": lambda r: float(r.effective_gate_weight),
    "router_probability": lambda r: float(r.router_probability),
    "router_logit": lambda r: float(r.router_logit),
    "abs_router_logit": lambda r: abs(float(r.router_logit)),
    "topk_rank": lambda r: float(r.topk_rank),
    "top1_top2_gap": lambda r: float(r.top1_top2_gap),
    "routing_entropy": lambda r: float(r.routing_entropy),
    "layer_id": lambda r: float(r.layer_id),
    "expert_id": lambda r: float(r.expert_id),
}

# 方向：True 表示该特征越大 → 预测 delta_nll 越大（越不关键）
FEATURE_ORIENTATION: dict[str, bool] = {
    "effective_gate_weight": False,
    "router_probability": False,
    "router_logit": False,
    "abs_router_logit": False,
    "topk_rank": True,
    "top1_top2_gap": False,
    "routing_entropy": True,
    "layer_id": False,
    "expert_id": False,
}
```

### 3.5 测试目录调整

测试也按层分子目录：

```
legacy/poc1/tests/
├── test_smoke.py               # → 移到 tests/analysis/ （测 analyze_records）
├── test_analysis.py            # → 移到 tests/analysis/
├── test_calibration.py         # → 移到 tests/analysis/
├── test_data.py                # → 移到 tests/experiment/
├── test_intervention.py        # → 移到 tests/runtime/
├── test_metrics.py             # → 移到 tests/runtime/
└── conftest.py                 # 新增：共享 fixture（fake AblationRecord 工厂等）
```

### 3.6 configs 清理

保留 `configs/poc.yaml`、`configs/smoke.yaml`、`configs/formal.yaml`。
删除 `configs/poc1.yaml`。

---

## 4. 实验结论分析

### 4.1 当前数据说了什么

1024 条消融记录的分布：

```
rank=0: mean_delta_nll = +0.0308  （移除 top-1 专家后 NLL 上升最多 → 最重要）
rank=1: mean_delta_nll = -0.0088  （移除 top-2 专家后 NLL 反而下降 → 该专家有时"帮倒忙"）
rank=2: mean_delta_nll = +0.0060
rank=3: mean_delta_nll = +0.0149
rank=4: mean_delta_nll = +0.0062
rank=5: mean_delta_nll = -0.0022
rank=6: mean_delta_nll = +0.0035
rank=7: mean_delta_nll = +0.0118  （最低权重专家移除后也有影响）
```

pairwise ranking accuracy: 0.478（低于随机的 0.5）
test R²: -0.038（模型在测试集上比预测均值还差）

### 4.2 关键观察

**观察 1：rank=0 的信号最强但反直觉**
移除 top-1 专家后 mean_delta_nll 最大（+0.0308），这本身是正确的——最重要的专家移除后影响最大。
但 pairwise accuracy 低于随机说明：在同一组内，按 router 权重排序并不能预测"谁被移除后影响更大"。

**观察 2：rank=1 专家平均是"有害"的**
移除 top-2 专家后 NLL 反而降低（-0.0088），说明这个专家的贡献有时是负面的。
这可能是因为 top-k 归一化后，rank=1 分到的权重过高，实际它的内容贡献不如权重暗示的那么大。

**观察 3：校准模型没有发现有用组合**
33 个特征子集的搜索结果全部收敛到 `topk_rank` 单特征。
所有组合特征的验证 MAE 都比单特征更差（过拟合）。
这说明在当前数据分布下，routing context 的细粒度数值（logit、probability、gap、entropy）不携带额外预测信号。

**观察 4：信号弱不等于信号不存在**

当前实验有几个固有的局限性：
1. **语料太窄**：50 条 theory prompts 属于同质领域，可能无法激发路由器的专业化分工
2. **context 太短**：24 tokens 几乎没有上下文，路由器可能还没来得及展现"根据上下文选择专家"的能力
3. **只看单 token NLL**：消融一个专家对下一个 token 的影响可能微弱，但对后续生成有级联效应
4. **消融方式**：zero-weight 消融改变了归一化后的权重分布，引入了间接效应
5. **数据量**：1024 条 = 128 groups，每组只有 8 个专家，组内统计噪声大

### 4.3 大白话解释

打个比方：公司里有 64 个专家，路由器是分配任务的经理。
当前实验就像只观察了 32 天的工作日志，每天只看了经理分配任务后一秒钟的结果。
你发现"经理把最重要的活给头号专家"这件事是对的（rank=0 移除影响最大），
但你没法从经理的分配记录里预测"具体哪个专家今天干活不好"。

为什么？可能因为：
- 看的场景太少（只有 32 天）
- 看的任务太单一（全是同一个领域的问题）
- 看的时间太短（只看下一秒的输出，没看整篇文章的质量）

---

## 5. 下一步实验：发掘 routing 价值

### 5.1 实验矩阵

核心思路：当前实验的"探测镜头"不够好，需要换不同的镜头来看。

| 实验 | 变量 | 目的 | 预估成本 |
|------|------|------|----------|
| E1: 长上下文 | context_length: 24 → 256 | 看路由器在长文本中是否展现更明确的分工 | 4x 计算量 |
| E2: 多领域语料 | theory → code/math/wiki/dialog | 看不同领域的路由器行为差异 | 2-4x 计算量 |
| E3: 多层级对比 | 4 层 → 全部 ~16 个 MoE 层 | 看浅层 vs 深层的路由器是否行为不同 | 4x 计算量 |
| E4: 多 token 消融 | 单 token NLL → 生成 50 tokens 的 NLL | 看消融的级联效应 | 50x 计算量 |
| E5: 基础模型对比 | 0924 → 0125（base model） | 看 instruct 微调是否改变了路由器行为 | 2x 计算量 |
| E6: 大规模验证 | 32 windows → 500 windows | 用更大样本验证前几个实验的最佳发现 | 15x 计算量 |

### 5.2 优先级排序

推荐顺序：

1. **E1（长上下文）**: 最可能暴露信号。24 tokens 的上下文太短，路由器几乎没有"选择"的空间。
   256 tokens 下，同一段文本的不同部分可能激活不同的专家组合，此时路由器的"专业性"才会体现。

2. **E2（多领域语料）**: 第二优先。如果路由器确实在不同领域选择不同专家，那么混合领域的 delta_nll 分布
   应该比单一领域更有结构。建议准备 4 类语料各 50 条：
   - 代码（Python 函数实现）
   - 数学/逻辑推理
   - 百科知识问答
   - 日常对话

3. **E3（多层级对比）**: 快速实验。不需要重新采集消融数据，只需分析时按 layer 分层看。
   浅层（layer 0-5）可能做语法/格式分工，深层（layer 10-15）可能做语义/知识分工。
   当前采样了 layer 0, 5, 10, 15，可以扩展为全部 MoE 层。

4. **E4（多 token 消融）**: 最有信息量但最贵。当前只看消融后下一个 token 的 NLL 变化，
   但 MoE 专家的影响可能体现在后续多个 token 上。
   建议对 top-50 个最极端的 group（oracle 选出 delta_nll 最大的），重新跑 50 token 生成对比。

5. **E5（基础模型对比）**: 等新语料下载完再做。instruct 微调可能让路由器变得更"确定性"，
   对比 base model 可以看微调是否压缩了路由器的信息量。

6. **E6（大规模验证）**: 等前面实验确定了最佳配置后再跑。

### 5.3 每个实验的具体操作

#### E1: 长上下文实验

1. 修改 `poc.yaml`：`context_length: 256`
2. 确保 prompt file 中有足够长的文本（≥ 300 tokens）
3. 跑 `run-all`，收集 32 windows × 4 layers × 8 = 1024 条记录
4. 分析：对比 24 vs 256 下的 pairwise accuracy、oracle mean_delta_nll、各 rank 的 delta_nll 分布

预期：256 下 oracle 和 random 的差距应该更大（路由器有更多上下文来做决策）。

#### E2: 多领域语料实验

1. 准备 4 个 prompt file：`code_prompts_50.jsonl`, `math_prompts_50.jsonl`, `wiki_prompts_50.jsonl`, `dialog_prompts_50.jsonl`
2. 每个领域单独跑 `run-all`
3. 分析：
   - 按领域看 pairwise accuracy（是否某个领域的路由器更可预测？）
   - 按领域看 rank 的 delta_nll 分布（是否某个领域的 top-1 更重要？）
   - 跨领域混合后再做 calibration（是否混合数据能让模型学到更好的特征？）

预期：代码和数学领域的路由器可能更"确定性"（expert 分工更明确），pairwise accuracy 更高。

#### E3: 多层级对比

1. 修改 `poc.yaml`：`num_moe_layers: 16`（OLMoE-1B-7B 有 ~16 个 MoE 层）
2. 跑 `run-all`，收集 32 × 16 × 8 = 4096 条记录
3. 分析：按 layer_id 分层，看每层的 delta_nll 分布和 pairwise accuracy
4. 画出 heatmap：x 轴 = layer_id，y 轴 = topk_rank，颜色 = mean_delta_nll

预期：浅层路由器可能信号弱（做的是低级的语法分工），深层路由器信号强（做高级语义分工）。

#### E4: 多 token 级联消融

1. 从当前 1024 条记录中，选出 delta_nll 绝对值最大的 50 条
2. 对每条，不只计算 next-token NLL，而是生成 50 tokens，对比 baseline vs ablated 的：
   - 50-token 累积 NLL
   - 文本相似度（ROUGE-L / BLEU）
   - 是否生成了不同的 token 序列（分歧点在哪）
3. 新增模块 `cascade.py` 用于多 token 生成和对比

预期：单 token 看不到信号的 case，在多 token 中可能暴露出来（级联放大效应）。

### 5.4 新增分析维度

除了上面的实验，在现有数据上还可以做：

**a. 条件分析：按 delta_nll 分桶**

当前把所有 group 混在一起看 pairwise accuracy。但如果只看"oracle 认为差异很大"的 group
（即 oracle 的 delta_nll << random 的 delta_nll），路由特征在这些 group 中是否更可预测？

操作：把 128 个 group 按 `oracle_delta_nll` 排序，取 top-32（oracle 选到了明显更差的 expert），
在这 32 个 group 中重新计算 pairwise accuracy。

**b. expert 身份分析**

当前按 topk_rank 聚合。但如果按 expert_id 聚合（64 个专家），看每个专家的平均 delta_nll，
是否有些专家"天生重要"（无论被分配到哪里，移除它都会造成大的质量下降）？

操作：对每个 expert_id 计算 mean_delta_nll，看分布是否不均匀（某些专家显著重要）。
如果是，那路由器的"选对人"可能不如"选对专家身份"重要。

**c. 权重归一化效应分析**

当前 zero-weight 消融后，剩余 7 个专家的权重被重新归一化。
这引入了一个间接效应：即使被消融的专家贡献为 0，归一化后的权重分布也变了。

操作：尝试"equal-weight"消融——把被消融专家的权重平分给其他专家（而不是让 softmax 自动归一化）。
对比两种消融方式的 delta_nll 分布差异。

---

## 6. 执行计划

### Phase 1a: 清理 + 迁移（1-2 小时）

1. 删除 5 个死文件 + poc1.yaml
2. 创建 4 个子目录：`core/`、`runtime/`、`experiment/`、`analysis/`
3. 新建 `core/features.py`（统一特征定义）
4. 按 3.3 迁移表移动文件 + 修改 import 路径
5. 创建 4 个子包的 `__init__.py`（re-export）
6. 更新顶层 `__init__.py` 和 `cli.py` 的 import 路径
7. 移动测试文件到子目录 + 新建 `conftest.py`
8. `pytest` 全通过确认

### Phase 1b: Bug 修复（1 小时）

迁移完成后，在原 2.2 节的 7 个问题逐个修复：

1. 修复 `experiment/environment.py` 的 bf16 降级
2. 修复 `experiment/config.py` 的默认配置路径
3. 修复 `analysis/policies.py` 的 raw_routing 重复问题
4. 修复 `experiment/ablation.py` 的重复前向计算（复用 trace 中的 baseline_nll）
5. 修改 `analysis/analysis.py` 引用 `core.features`
6. 修改 `analysis/calibration.py` 引用 `core.features`
7. 修复 `analysis/analysis.py` 硬编码策略名

### Phase 2: 快速分析（30 分钟，不需要 GPU）

在现有 1024 条数据上做：
1. 按 layer_id 分层看 pairwise accuracy
2. 按 expert_id 聚合看 delta_nll 分布
3. 按 oracle_delta_nll 排序后看 top-32 group 的 pairwise accuracy
4. 画出 rank × layer 的 delta_nll heatmap

这些分析可以揭示当前数据中是否已经存在被淹没的信号。

### Phase 3: 新实验准备

1. 准备长文本 prompts（E1）
2. 准备多领域 prompts（E2）
3. 修改 config 支持全层采样（E3）

### Phase 4: 执行实验（GPU 服务器）

按优先级跑 E1 → E2 → E3。
每个实验跑完后立即做 Phase 2 的分析，决定是否继续。

### Phase 5: 级联消融（E4）

等 E1-E3 确定最佳配置后，跑 E4 作为最终验证。

---

## 7. 文件变更汇总

### 删除（6 个）

```
routesense_poc1/router_trace.py
routesense_poc1/adapter_registry.py
routesense_poc1/model_inspector.py
routesense_poc1/output_archive.py
routesense_poc1/utils.py
configs/poc1.yaml
```

### 新增（9 个）

```
routesense_poc1/core/__init__.py
routesense_poc1/core/features.py          # 统一特征定义
routesense_poc1/runtime/__init__.py
routesense_poc1/experiment/__init__.py
routesense_poc1/analysis/__init__.py
tests/conftest.py                         # 共享 fixture
tests/analysis/                            # 测试子目录
tests/runtime/                             # 测试子目录
tests/experiment/                          # 测试子目录
```

### 移动 + 修改 import（16 个）

```
schemas.py          → core/schemas.py
serialization.py    → core/serialization.py
model_loader.py     → runtime/model_loader.py
moe_inspect.py      → runtime/moe_inspect.py
intervention.py     → runtime/intervention.py
trace.py            → runtime/trace.py
metrics.py          → runtime/metrics.py
config.py           → experiment/config.py
data.py             → experiment/data.py
ablation.py         → experiment/ablation.py
environment.py      → experiment/environment.py
policies.py         → analysis/policies.py
analysis.py         → analysis/analysis.py
calibration.py      → analysis/calibration.py
plotting.py         → analysis/plotting.py
cli.py              → cli.py（位置不动，import 路径全改）
```

---

## 8. 验收标准

### Phase 1a 验收（迁移）
- [ ] 5 个死文件 + poc1.yaml 已删除
- [ ] 4 个子目录已创建（core/runtime/experiment/analysis）
- [ ] 16 个文件已移动到新位置
- [ ] 所有 import 路径已更新
- [ ] `pytest` 全通过（现有测试不 break）
- [ ] `python -m routesense_poc1.cli doctor` 可正常运行

### Phase 1b 验收（Bug 修复）
- [ ] `core/features.py` 已创建，三个消费者引用统一源
- [ ] `raw_routing` 和 `effective_gate_weight_min` 行为不再重复
- [ ] `environment.py` 不再静默降级精度
- [ ] `config.py` 默认指向 `poc.yaml`
- [ ] `ablation.py` 复用 trace 的 baseline_nll（无重复前向）
- [ ] `analysis.py` 不再硬编码特征名

### Phase 2 验收
- [ ] 按 layer 分层的 pairwise accuracy 表格
- [ ] 按 expert_id 的 delta_nll 分布图
- [ ] top-32 oracle group 的 pairwise accuracy
- [ ] rank × layer 的 delta_nll heatmap
- [ ] 分析结论：信号是否存在于某个子群中

### Phase 3 验收
- [ ] 4 个新 prompt file 已准备（code/math/wiki/dialog 各 50 条，长度 ≥ 300 tokens）
- [ ] 长文本 config 可运行
- [ ] 全层采样 config 可运行

---

## 9. 补充：新增分析模块规格

### 9.1 `analysis/layer_analysis.py`（Phase 2 用）

放在 `analysis/` 层，用于在现有数据上做分层分析，不需要 GPU：

```python
def analyze_by_layer(records: list[AblationRecord]) -> dict:
    """按 layer_id 分层计算 pairwise accuracy 和 delta_nll 分布。"""
    # 返回 {layer_id: {pairwise_accuracy, mean_delta, median_delta, num_groups}}

def analyze_by_expert(records: list[AblationRecord]) -> dict:
    """按 expert_id 聚合，看每个专家的 delta_nll 分布。"""
    # 返回 {expert_id: {mean_delta, std_delta, count, rank_distribution}}

def analyze_oracle_subset(records: list[AblationRecord], top_k: int = 32) -> dict:
    """只看 oracle 选到明显更差 expert 的 group，看这些 group 中 pairwise accuracy 是否更高。"""
    # 按 oracle_delta_nll 排序，取 top_k 个 group
    # 在这些 group 中重新计算 pairwise accuracy

def compute_rank_layer_heatmap(records: list[AblationRecord]) -> list[list[float]]:
    """生成 rank × layer 的 delta_nll 矩阵，用于画 heatmap。"""
    # 返回 matrix[layer_id][topk_rank] = mean_delta_nll
```

### 9.2 `runtime/cascade.py`（Phase 5 用）

放在 `runtime/` 层，多 token 级联消融，需要 GPU：

```python
def run_cascade_ablation(
    model, tokenizer, window: Window, moe_layer: MoELayerSpec,
    expert_rank: int, num_generate_tokens: int = 50,
) -> CascadeResult:
    """对指定 expert 做消融，然后生成 num_generate_tokens 个 token。
    
    对比 baseline vs ablated 的：
    - 累积 NLL（所有生成 token 的交叉熵之和）
    - 首个分歧点（从第几个 token 开始生成不同内容）
    - ROUGE-L 相似度
    """

@dataclass
class CascadeResult:
    baseline_text: str
    ablated_text: str
    baseline_cumulative_nll: float
    ablated_cumulative_nll: float
    delta_cumulative_nll: float
    first_divergence_token: int  # 第几个 token 开始不同
    rouge_l: float
```

### 9.3 `analysis/conditional_analysis.py`（Phase 2 用）

放在 `analysis/` 层，条件分析，发现被淹没的信号：

```python
def split_by_context_entropy(records: list[AblationRecord]) -> dict[str, list[AblationRecord]]:
    """按 routing_entropy 高低分组。"""
    # 高熵 = 路由器犹豫不决，低熵 = 路由器很确定
    # 分别看两组的 pairwise accuracy

def split_by_top1_dominance(records: list[AblationRecord]) -> dict[str, list[AblationRecord]]:
    """按 top1_top2_gap 大小分组。"""
    # gap 大 = 路由器非常确定首选专家，gap 小 = 首选和二选差不多
    # 分别看两组的 pairwise accuracy

def split_by_delta_magnitude(records: list[AblationRecord], threshold: float = 0.05) -> dict:
    """只看 delta_nll 绝对值大于阈值的 group。"""
    # 如果只关注"影响大的消融"，路由特征是否更可预测？

def analyze_expert_specialization(records: list[AblationRecord]) -> dict:
    """看同一个 expert_id 在不同 layer 和不同 window 中的 delta_nll 是否一致。"""
    # 如果某个 expert 在所有层中都"重要"，说明它是通用重要专家
    # 如果只在特定层重要，说明路由器确实在做"按层分工"
```

---

## 10. 补充：当前实验数据的深层问题

### 10.1 零权重消融的归一化陷阱

当前消融方式是把被选中专家的 weight 设为 0，然后模型内部的 `experts()` 函数会用新的权重分布计算输出。

问题在于：OLMoE 使用 `norm_topk_prob=True`，消融一个专家后，剩余专家的权重会被重新归一化。

假设 top-8 权重为 [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04]，归一化后总和为 1。

消融 rank=0 后：原始权重 [0, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04]
归一化后：[0, 0.286, 0.214, 0.143, 0.114, 0.100, 0.086, 0.057]

注意 rank=1 的权重从 0.20 涨到 0.286（涨了 43%），这意味着即使 rank=1 专家的内容贡献不变，
它在输出中的影响力也大幅增加。这个间接效应可能比直接效应更大。

消融 rank=7 后：原始权重 [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0]
归一化后：[0.313, 0.208, 0.156, 0.104, 0.083, 0.073, 0.063, 0]

rank=0 只从 0.30 涨到 0.313（涨了 4%），间接效应很小。

这解释了为什么 rank=0 的 mean_delta_nll 最大（+0.0308）而 rank=7 的也不小（+0.0118）：
消融 rank=0 时，间接效应巨大（剩余权重大幅重分配），而消融 rank=7 时间接效应小。

但间接效应掩盖了真实的内容贡献差异。这需要在分析中控制。

### 10.2 建议的改进消融方式

**equal-weight 消融**：把被消融专家的权重均匀分配给其他专家，而不是让归一化自动处理。

消融 rank=0 后：
- 原始权重 [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04]
- 把 0.30 平分给 7 个：每个加 0.043
- 新权重 [0, 0.243, 0.193, 0.143, 0.123, 0.113, 0.103, 0.083]
- 归一化后总和仍为 1

这样间接效应被均匀分散，不会集中在 rank=1 上。

实现方式：修改 `intervention.py` 的 `_wrapped_forward`：

```python
# 当前：直接置零
top_k_weights[flat_target_pos, self.expert_rank] = 0.0

# 改进：均分权重
ablated_weight = top_k_weights[flat_target_pos, self.expert_rank].item()
top_k_weights[flat_target_pos, self.expert_rank] = 0.0
redistribute = ablated_weight / (moe_layer.top_k - 1)
for rank in range(moe_layer.top_k):
    if rank != self.expert_rank:
        top_k_weights[flat_target_pos, rank] += redistribute
```

对比两种消融方式的结果差异，可以揭示"归一化间接效应"对分析结论的影响程度。

---

## 11. 补充：语料准备指南

### 11.1 语料格式

所有 prompt file 使用 JSONL 格式：

```json
{"document_id": 0, "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
{"document_id": 1, "text": "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None"}
```

约束：
- 每条 text 至少 300 tokens（用 tokenizer 验证）
- 50 条 per domain
- document_id 唯一且在领域内连续

### 11.2 具体语料推荐（国内可直接下载）

以下所有语料都可通过三种方式在国内下载：
- **HF Mirror**：`hf-mirror.com`（HuggingFace 公益镜像，直接替换 huggingface.co 域名）
- **ModelScope**：`modelscope.cn`（阿里达摩院托管，pip install modelscope）
- **Gitee 镜像**：部分数据集有 Gitee 副本

---

#### 代码领域：`code_prompts_50.jsonl`

**推荐数据源：HumanEval + MBPP**

| 数据集 | 规模 | 内容 | 下载方式 |
|--------|------|------|----------|
| HumanEval | 164 题 | Python 函数实现（含 docstring + 测试） | `modelscope` → `opencompass/humaneval` |
| MBPP | 974 题 | Python 基础编程题（含描述 + 参考解） | `modelscope` → `modelscope/mbpp` |

**准备方式：**
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("opencompass/humaneval", split="test")
# 每条记录包含 prompt（docstring + 函数签名）
# 拼接 prompt + canonical_solution 作为完整文本
# 筛出 token 数 ≥ 300 的条目
```

**备选：如果 HumanEval 太短（平均 < 100 tokens），改用 The Stack**
- 数据集：`bigcode/the-stack`（可通过 hf-mirror 下载）
- 筛 Python 子集，取函数/类定义，每条 300-1000 tokens

---

#### 数学领域：`math_prompts_50.jsonl`

**推荐数据源：GSM8K**

| 数据集 | 规模 | 内容 | 下载方式 |
|--------|------|------|----------|
| GSM8K | 8,500 题 | 小学数学应用题（多步推理） | `modelscope` → `AI-ModelScope/gsm8k` 或 Gitee `hf-datasets/gsm8k` |
| MATH | 12,500 题 | 竞赛级数学（代数/几何/概率/数论） | hf-mirror → `hendrycks/competition_math` |

**准备方式：**
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("AI-ModelScope/gsm8k", subset_name="main", split="test")
# 每条记录有 question + answer（含推理步骤）
# 拼接 question + answer 作为完整文本
# GSM8K 平均长度 ~100 tokens question + ~100 tokens answer
# 筛选后应该能拿到 50+ 条 ≥ 300 tokens 的记录
```

**注意：GSM8K 问题本身可能不够长（~80 tokens），需要拼接 answer 部分。**

---

#### 百科知识领域：`wiki_prompts_50.jsonl`

**推荐数据源：英文 Wikipedia 精选段落**

| 数据集 | 规模 | 内容 | 下载方式 |
|--------|------|------|----------|
| wikipedia (20220301.en) | 6M+ 文章 | 英文维基百科全文 | hf-mirror → `wikimedia/wikipedia` |
| Simple Wikipedia | 200K+ 文章 | 简化版英文维基 | hf-mirror → `SoybeanZhu/Simple_Wikipedia` |

**准备方式：**
```python
# 推荐用 Simple Wikipedia（句子更简单，适合单 GPU 推理）
# 或通过 Wikipedia API 直接下载精选段落：
import requests
# 下载 "Featured articles" 或 "Good articles" 的 intro 段落
# 每个段落通常 200-500 tokens
```

**备选更轻量方式（无需下载大数据集）：**
```python
# 直接用 Wikipedia REST API 获取随机文章摘要
# https://en.wikipedia.org/api/rest_v1/page/random/summary
# 循环 200 次，筛出长度 ≥ 300 tokens 的
```

---

#### 对话领域：`dialog_prompts_50.jsonl`

**推荐数据源：ShareGPT + OpenAssistant**

| 数据集 | 规模 | 内容 | 下载方式 |
|--------|------|------|----------|
| ShareGPT (Vicuna unfiltered) | 53K 对话 | ChatGPT 真实用户对话 | hf-mirror → `anon8231489123/ShareGPT_Vicuna_unfiltered` 或 ModelScope → `swift/sharegpt` |
| OpenAssistant Conversations | 160K 对话 | 开源助手对话 | hf-mirror → `OpenAssistant/oasst1` |

**准备方式：**
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("swift/sharegpt", split="train")
# 每条记录是完整对话（多轮 user/assistant 交互）
# 取前 3-4 轮（截断到 ~400 tokens）作为 context
# 筛出总长 ≥ 300 tokens 的对话
```

**关键注意事项：**
- ShareGPT 对话长度分布很不均匀，需过滤掉过短的闲聊
- 优先选择"技术咨询"或"知识问答"类对话（更有信息密度）
- 每条对话取前 N 轮拼接，确保总长 300-500 tokens

---

#### 额外推荐：长文本语料（用于 E1 长上下文实验）

如果 context_length 扩展到 256 或 512，需要更长的文本：

| 数据集 | 规模 | 内容 | 下载方式 |
|--------|------|------|----------|
| PG-19 (Project Gutenberg) | 28K 本书 | 公版英文书籍全文 | hf-mirror → `deepmind/pg19` |
| BookCorpus | 17K 本书 | 免费书籍 | hf-mirror → `bookcorpus/bookcorpus` |
| arXiv 论文摘要 | 2M+ 篇 | 学术论文 abstract | hf-mirror → `arxiv_dataset` |

**PG-19 特别适合长上下文实验**：每本书平均 50K tokens，可以随意截取任意长度的窗口。

---

### 11.3 下载脚本模板

```python
# download_datasets.py
"""国内环境友好的数据集下载脚本。"""
from pathlib import Path
import json

# 方式 1：通过 ModelScope SDK 下载（推荐，国内速度最快）
def download_via_modelscope(dataset_id: str, split: str, output_path: Path):
    from modelscope.msdatasets import MsDataset
    ds = MsDataset.load(dataset_id, split=split)
    records = list(ds)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 方式 2：通过 HF Mirror 下载（设置环境变量即可）
# export HF_ENDPOINT=https://hf-mirror.com
# 然后正常使用 huggingface_hub 或 datasets 库

def download_via_hf_mirror(dataset_id: str, split: str, output_path: Path):
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    ds.to_json(str(output_path))

# 推荐下载清单
DOWNLOAD_LIST = [
    # (dataset_id, split, output_file, method)
    ("opencompass/humaneval", "test", "raw/humaneval.jsonl", "modelscope"),
    ("modelscope/mbpp", "test", "raw/mbpp.jsonl", "modelscope"),
    ("AI-ModelScope/gsm8k", "test", "raw/gsm8k.jsonl", "modelscope"),
    ("swift/sharegpt", "train", "raw/sharegpt.jsonl", "modelscope"),
]
```

### 11.4 语料筛选与准备脚本

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
min_tokens = 300  # 确保 context_length=256 时有足够余量

for line in open("raw_prompts.jsonl"):
    text = json.loads(line)["text"]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) >= min_tokens:
        # 保留
```

---

## 12. 总结

当前 POC1 已经证明了消融链路的可行性，但实验条件不足以揭示 routing signal。

短期行动（Phase 1-2）：
1. 代码收口：删 6 个死文件，统一特征定义，修复 6 个 bug
2. 深层分析：在现有 1024 条数据上做分层/分桶分析，看信号是否被全局均值淹没

中期行动（Phase 3-5）：
1. 长上下文 + 多领域语料 + 全层采样 → 用更好的"探测镜头"重新看 routing signal
2. 多 token 级联消融 → 看单 token 看不到的级联效应
3. equal-weight 消融 → 消除归一化间接效应的干扰

核心判断：routing signal 大概率存在，但当前实验设计的探测能力不足以揭示它。
通过改变语料多样性、上下文长度、消融方式和观测粒度，应该能找到一个条件下 routing 特征确实有用的证据。
