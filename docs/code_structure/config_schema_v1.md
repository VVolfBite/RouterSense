# Config Schema v1

正式 v1 配置分组：

- `run`
- `model`
- `topology`
- `workload`
- `runtime`
- `traffic`
- `policy`
- `prediction`
- `evaluation`
- `replay`
- `oracle`
- `regime_analysis`
- `strategies`（仅在线对比矩阵入口）

兼容策略：

- `schema_version: 0` 旧配置通过 `normalize_run_config` 桥接到 v1
- 冲突字段会直接报错，不静默降级
- public entrypoints 只读取 canonical 字段，再生成 legacy 内部 runner payload
