# Codex 执行任务：仅导出 RouterSense 真实 Trace 与 Traffic 数据包

你在本轮只充当**确定性实验材料执行器**。不要分析调度算法，不要调参，不要比较 Local/Joint，不要修改 scheduler、runtime、predictor、executor、traffic 语义或论文结论。

本轮唯一目标：

> 在最新合并后的 `convergence/m123-integration` clean commit 上，使用真实 OLMoE 模型和现有正式采集入口，生成一份自包含、可校验的 Router Trace 与 TrafficInstance ZIP，供后续独立算法分析使用。

## 1. 开始条件

在仓库根目录执行并保存输出：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse origin/convergence/m123-integration
git status --short
git remote -v
```

必须满足：

```text
branch = convergence/m123-integration
HEAD = origin/convergence/m123-integration
git status = clean
```

如果不满足，停止并报告 `TRACE-DATA-BLOCKED`。不要自行合并、重置或修改源码。

## 2. 禁止事项

禁止修改或执行以下工作：

```text
src/rs/scheduling/**
experiments/paper/family_evaluation.py
任何 Local/Joint policy
任何算法权重
任何算法选择、胜负比较、调参或结论分析
正式 predictor
正式 executor
exact oracle
runtime lifecycle
stable token ownership
expert placement policy
```

允许做的只有：

```text
使用现有正式 trace capture 入口
使用现有正式 traffic builder
新增纯数据 corpus/config（仅在确有必要时）
编写不改变语义的打包脚本
运行数据一致性检查
```

若现有 CLI 已支持所需操作，优先只生成配置和调用 CLI，不新增代码。

## 3. 模型和环境

真实模型路径：

```text
D:\models\OLMoE-1B-7B-0924-Instruct
```

模型不存在、加载失败或 Router Hook 不可用时，停止并报告 `TRACE-DATA-BLOCKED`。不得使用 synthetic routing、随机 expert IDs 或旧 fixture 冒充真实 trace。

记录：

```text
OS
Python 版本
PyTorch 版本
CUDA 版本
GPU 型号与显存
CPU 型号与物理核心数
内存
模型 ID、revision、dtype
MoE 层数、expert 数、top-k
```

## 4. 固定 Prompt Corpus

创建仓库外的运行目录，不把采集结果提交到 Git。

构造 48 个固定 prompt：

```text
development: 32
validation: 16
```

固定种子：

```text
20260717
```

覆盖输入 token bucket：

```text
8, 16, 32, 64, 128, 256
```

覆盖类型：

```text
中文问答
英文问答
代码理解/生成
数学与逻辑
技术说明
摘要
多轮对话风格
中英混合
```

每个样本保存：

```text
sample_id
split
prompt_text
category
language
requested_length_bucket
actual_input_token_count
token_ids_digest
seed
```

Development 和 validation 不得有相同 prompt、相同 token IDs 或相同 digest。

本轮不生成 frozen-test 算法数据。后续 frozen test 将另行采集，避免在调优阶段泄漏。

## 5. 真实 Router Trace 捕获

只调用仓库现有正式 trace capture 路径。对每个 prompt 捕获所有正式支持的 MoE 层。

每条 compact trace 至少包含：

```text
sample_id
split
layer_id
token_position
request/batch identity（若正式入口提供）
selected expert IDs
top-k weights（若正式入口提供）
source token identity
model ID/revision
```

必须输出并验证：

```text
sample 数
每个 sample 的 token 数
MoE 层数
expert 数
top-k
trace record 数
理论 record 数
实际/理论一致性
缺失 layer/sample 统计
```

若当前 harness 只支持 prefill 或固定 forward 模式，按真实能力采集并在 manifest 中写明。不得伪造 decode trace。

## 6. TrafficInstance 构造

使用当前代码中的正式：

```text
stable_token_owner_v1
当前 expert placement policy
current-layer mapping
target-layer mapping
```

对 development 和 validation 的 trace 构造：

```text
vEP = 2, 4, 8, 16
```

每个实例必须保存：

```text
traffic_instance_id
sample_id
split
layer_id
target_layer_id
virtual_ep_size
P0 dispatch matrix
P1 return matrix
P2 next-layer dispatch matrix（若下一层存在）
p2_available
source_ownership_policy_id
placement_policy_id
current_layer_mapping_digest
target_layer_mapping_digest
P0/P1/P2 matrix digests
```

一致性检查：

```text
P1 == transpose(P0)
P2(layer L) == P0(layer L+1)
target_mapping_digest(L) == current_mapping_digest(L+1)
所有矩阵维度 == vEP × vEP
所有值为非负整数
同一 token 的 virtual source owner 跨层稳定
```

最后一个 MoE 层若无下一层，必须写 `p2_available=false`，不得复制当前层矩阵。

## 7. 实例描述特征

只计算与算法无关的输入特征，不运行任何 policy：

```text
total volume
remote volume
self volume
nonzero density
row max/mean/CV
column max/mean/CV
hotspot ratio
largest-flow ratio
matrix entropy
P0/P1/P2 volume
P0/P1/P2 matrix digests
```

不得生成 Local/Joint objective、win/tie/loss 或算法结论。

## 8. 可选时间材料

仅当现有正式 harness 已经能够无修改导出时，附带：

```text
单层 router trace capture 时间
D2H 时间
traffic build 时间
每层 forward 时间
```

不得为了时间采集修改模型 Hook 或 runtime。无法获得时标记 `not_available`。

## 9. 数据包结构

生成唯一 ZIP：

```text
RouterSense_trace_traffic_data_<commit8>_<timestamp>.zip
```

结构：

```text
README.md
run_manifest.json
checksums.sha256

git/
  branch.txt
  commit.txt
  remote_commit.txt
  status.txt
  remote.txt

environment/
  hardware.json
  software.json
  model_metadata.json

corpus/
  development_prompts.json
  validation_prompts.json
  split_manifest.json

trace/
  development_trace.jsonl
  validation_trace.jsonl
  trace_summary.json
  architecture_probe.json
  paper_trace_bundle_manifest.json

traffic/
  development_instances.json
  validation_instances.json
  traffic_summary.json
  feature_statistics.json
  ownership_and_placement_summary.json

commands/
  commands.txt
  generated_configs/

logs/
  stdout.log
  stderr.log

tests/
  data_validation.json
  compileall.txt
```

不要包含：

```text
模型权重
完整 logits
hidden states
venv
.git
__pycache__
*.pyc
.pytest_cache
算法运行结果
算法调参记录
绝对临时路径
```

## 10. 可移植性和 fresh-unpack

要求：

```text
ZIP 成员全部使用 POSIX 相对路径
所有 JSON/JSONL/TXT/MD/YAML 为 UTF-8 无 BOM + LF
checksums.sha256 使用相对 POSIX 路径
artifact 路径不得包含 C:\、C:/、D:\、D:/、/tmp 或 %TEMP%
```

Fresh-unpack 后重新验证：

```text
checksums
commit identity
split 无重叠
trace count
P1 transpose
P2 next-layer consistency
mapping digest consistency
矩阵维度与非负性
无缓存/模型文件
无绝对路径
```

输出 `data_validation.json`。

## 11. Git 处理

本轮数据不提交 Git。

只有在为了固定 prompt corpus 或纯打包脚本而确实新增源码文件时，先停止并报告需要修改的文件，不要自行提交；优先使用仓库外临时配置完成任务。

最终仓库仍必须：

```text
git status = clean
HEAD = remote commit
```

## 12. 最终回复

只汇报：

```text
branch
commit
remote commit
git clean

模型和环境
prompt count by split/token bucket
trace sample/record count
traffic instance count by split/vEP
数据一致性检查
fresh-unpack 检查

唯一 ZIP 路径
ZIP SHA-256
final status
```

最终状态只允许：

```text
TRACE-DATA-READY
TRACE-DATA-PARTIAL
TRACE-DATA-BLOCKED
```

不要评价任何调度算法，不要提出算法修改建议，不要运行 Local/Joint 实验。
