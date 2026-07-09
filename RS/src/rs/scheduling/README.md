# Scheduling Layer

这个目录是论文算法层。

负责：

- phase-local 策略
- multiphase logical scheduler
- reference / oracle / replay
- 计划合同与验证

不负责：

- torch / Megatron / NCCL
- 实验入口编排
- artifact 文件系统写入

依赖边界：

- 可以依赖 `rs.core`
- 不能依赖 online runtime、Megatron、experiments

当前最重要入口：

- `registry.py`
- `phase_execution.py`
- `phase_local/`
- `multiphase/`
