# Offline Runtime

这个目录负责离线分析和重放。

负责：

- trace 读取与重建
- 预测分析
- logical schedule replay
- theoretical / oracle 对比
- control replay trace 的离线统计

不负责：

- 真实 GPU 执行
- Megatron hook
- NCCL executor

原则：

- 可以重
- 可以慢
- 但输入 trace 和流量矩阵必须真实可审计
