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
