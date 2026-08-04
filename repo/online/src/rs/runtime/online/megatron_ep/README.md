# Megatron EP Online Runtime

这个目录负责真实在线执行路径。

负责：

- host hook 安装
- P0/P1 生命周期编排
- root 协商与 wave-level 计划
- 本地 materialize
- executor 触发
- 最小观测 / audit / timing

不负责：

- 离线理论分析
- 大规模报告生成
- 论文绘图
- 调度算法研究本体

当前最重要入口：

- `host.py`：外部接入入口
- `lifecycle.py`：P0/P1 生命周期主线
- `control/plan_agreement.py`：phase 协商
- `pending_window/adapter.py`：prepared-window / fast path 接口
- `execution/async_p2p_executor.py`：真实 `batch_isend_irecv` 路径
- `async_release/runtime_projection.py`：runtime-safe-U host projection

当前状态：

- B2 lifecycle 已在 `4x RTX 4090 D` 上通过，确认了真实 `PhaseReadyContext` 流量采集、P1 plan reuse 和 `batch_isend_irecv` 调用。
- C2 correctness 还需要下一次 GPU 会话的短 gate。
- A2 performance 还需要下一次 GPU 会话的正式性能矩阵。
