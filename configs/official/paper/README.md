# Paper evaluation configs

This directory only contains frozen paper-evaluation entry configs. The evaluator must consume these configs directly and emit `consumed_config.json` with the resolved values.

- `capability_audit.yaml`
  - 用途：论文评估能力审计入口。
  - 环境：CPU 即可；若提供真实 trace bundle 或模型路径，可额外验证对应能力。
  - 结论范围：只审计 public entrypoint、最小 smoke 和 contract test 是否闭环，不给论文结论。

- `trace_capture.yaml`
  - 用途：真实模型 trace capture。
  - 环境：外部模型路径，默认通过 `RS_MODEL_PATH` 指向 `D:\models\...`。
  - 结论范围：只生成独立 trace bundle，不在 Git clone 内伪造 artifact。

- `scheduling_value.yaml`
  - 用途：offline scheduling evaluator。
  - 环境：CPU。
  - 结论范围：只做 paired/oracle fail-closed 语义，不把 oracle-like 伪装成 exact comparable oracle。

- `prediction_value.yaml`
  - 用途：prediction evaluator。
  - 环境：CPU。
  - 结论范围：perfect/zero/shuffled baseline 可运行；正式 predicted path 缺失时必须显式保留缺失字段。

- `hiding_timeline.yaml`
  - 用途：hiding timeline evaluator。
  - 环境：CPU。
  - 结论范围：当前只允许报告 public API 可见的 timeline 能力状态。

- `runtime_correctness_gloo.yaml`
  - 用途：真实 4-rank Gloo runtime correctness wrapper。
  - 环境：CPU/Gloo，物理 world size = 4。
  - 结论范围：只有正式 runner 返回 executed-plan identity、任务完成和 tensor parity 证据时，才能写 `RUNTIME_CORRECTNESS`。

- `scheduling_family_pilot.yaml`
  - 用途：严格同核 `Local(f)`/`Joint(f)` 算法族小规模 pilot。
  - 结论范围：同时记录 makespan 与 planning overhead，只验证采数合同，不形成论文普遍性结论。

- `scheduling_prediction_closure.yaml`
  - 用途：统一 Local/Joint P01/P012、P2 信息阶梯、预测误差与 exact tiny-control。
  - 环境：CPU；输入为可移植 trace/traffic 包。
  - 结论范围：严格区分 rolling exact P01-reactive、predicted P012 与 clairvoyant O-Joint(P012-perfect)；预测字节绝不作为执行真值。
