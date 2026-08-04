# ReleaseFrontier 图表脚本与论文图形计划

## 叙事原则

论文动机不是“某个匹配算法更快”，而是：两个顺序可以具有相同的当前 phase 完成时间，却产生不同的跨阶段释放时间和通信关键路径。因此结果图应优先展示：

1. Local 与 Joint 的严格同核心配对；
2. Zero、FATE、Perfect 对信息价值的分解；
3. 通信等待、尾部和完整通信窗口三个层次；
4. EP2/4/8 实测与 EP16/32 projected 的证据边界；
5. 规划、预测、绑定开销是否低于通信收益。

Overview 图继续采用两条水平链：上方控制面提前预测/规划，下方数据面等待真实路由、绑定并执行。数据图不复刻 overview 的框图风格，而保持简洁的 INFOCOM 曲线、柱状图和表格。

## 建议图表

### Fig. 1 Motivation

使用现有 `motivation_joint_vs_local.tex` 重画或转为 draw.io：Local 与 Joint 当前 Combine 的结束时间相同，但 Joint 提前释放 rank，使下一层长 Dispatch 与剩余 Combine 重叠。

### Fig. 2 System overview

使用现有 `overview_pipeline.tex` 作为内容蓝本：Observe → frontier → FATE hint → prepared order；真实 route 到达后 reconcile/bind → execute。

### Fig. 3 Main performance

三联柱状图：Mean Communication Stall、P95 Communication Stall、Communication Makespan，相对 FIFO-Local 的 paired reduction。主图只保留 5–8 个核心策略，完整算法矩阵放表格。

### Fig. 4 Local-to-Joint genericity

每个 ordering core 一组柱：Mean Stall、P95 Stall、Communication Makespan 的 Joint over matched Local 改善。

### Fig. 5 Scaling

横轴 EP2/4/8/16/32。EP2/4/8 使用实线和实心 marker；EP16/32 使用虚线和空心 marker，并明确标注 projected。

### Fig. 6 Prediction value

Zero / FATE / Perfect 三柱，附 Predicted-to-Perfect gap 与 gain recovered。

### Fig. 7 Rank-stall CDF

代表性模型/规模下 FIFO-Local、RSCF-Local、RSCF-Joint-FATE、Perfect/Oracle 的 rank communication stall CDF。

### Fig. 8 Overhead

Prediction、planning/control、binding/repair 的 visible overhead 堆叠柱；hidden overhead 放附表或正文。

## 一条命令生成图表

```bash
pip install -e '.[plot]'
python scripts/plot_results.py \
  --input-csv /path/to/results.csv \
  --output-dir /path/to/figures \
  --baseline FIFO-Local \
  --width double
```

按模型/规模筛选：

```bash
python scripts/plot_results.py \
  --input-csv results.csv \
  --output-dir figures/olmoe_ep8 \
  --models OLMoE-1B-7B-0924 \
  --eps 8 \
  --sequences 128 \
  --treatments FIFO-Local,Birkhoff-Local,RSCF-Local,RSCF-Joint-FATE,Oracle-Joint
```

脚本同时输出 PDF、SVG、PNG，以及规范化数据 CSV 和 LaTeX 表格。
