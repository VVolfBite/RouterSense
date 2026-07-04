# Policy Library v1 Handoff

## A. 当前已验证状态

- `P0/P1` pre-transport hook 已冻结。
- 双端 `TransferLayout` 已冻结。
- `sync_before_phase` + `phase_sync_wave` executor 已冻结。
- `bucketed_fifo` 与 `trivial_reverse_bucket` 已在真实 NCCL bucket/wave 提交顺序上产生变化，并保持数值正确。
- policy 与 executor 已分离，统一通过：

```text
PhaseReadyContext
-> SchedulingPolicy
-> PhaseExecutionPlan
-> phase_sync_wave executor
```

- 当前 policy library：
  - `bucketed_fifo`
  - `trivial_reverse_bucket`
  - `aurora_order_fixed`
  - `fast_bvn_single_tier`
  - `routersense_p0p1_reservation`
  - `routersense_p0p1p2_hint`

## B. 当前真实证据边界

- 已验证范围：
  - 单机 `EP=2`
  - 真实 `NCCL`
  - selected-layer real-executor injection validation
  - policy injection / layout / logits correctness
- 当前结果不等于：
  - 性能收益结论
  - `EP=4/8` 真机结果
  - 多机结果
  - `default_continue` 真异步执行结果
  - 真实 `P2` predictor 结果

## C. 冻结边界

后续禁止改动以下语义，除非明确重开 runtime 底座：

- `native_runtime.py` 中的 hook 语义
- `integrations/megatron_ep/routersense/phase/contracts.py` 中的 `TransferLayout` 语义
- `MegatronPhaseTransportAdapter`
- `sync_wave_executor`
- `P0` `hidden_states + routing_probs` bundle atomicity
- `P1` bundle contract

## D. 允许后续改动的位置

- `integrations/megatron_ep/routersense/policy/`
- policy registry
- policy diagnostics
- synthetic policy cases
- benchmark / evaluation config
- 实验 runner 的参数和 summary 字段

## E. 后续性能实验的公平性约束

- 若某个 `P0-only` baseline 可以看到完整当前 `P0` matrix，则必须允许其推导 reverse `P1` demand。
- `P2` 只能来自非 oracle、可校验的 predictor artifact。
- `deterministic_stub` 永远 `evaluation_eligible = false`。
- `aurora_order_fixed` 与 `fast_bvn_single_tier` 是 fixed-placement adaptation，不得写成完整 Aurora / FAST。
- `routersense_p0p1p2_hint` 只有在 `calibrated_artifact` 下才允许进入正式效果表。

## F. 关键命令

完整 unit test：

```bash
PYTHONPATH=. python -m pytest -q integrations/megatron_ep/tests
```

source archive self-check：

```bash
bash tools/archive/package_source_only.sh --scope mainline <source-archive>
mkdir -p /tmp/routersense-policy-library-archive-check
tar -xzf <source-archive> -C /tmp/routersense-policy-library-archive-check
cd /tmp/routersense-policy-library-archive-check/RS
PYTHONPATH=. python -m pytest -q integrations/megatron_ep/tests
```

已有 real-GPU policy validation artifact：

```text
artifacts/megatron_ep/phase_executor/
  native-selected-reference/
  aurora-selected-no-stop-v1/
  fastbvn-selected-no-stop-v1/
  rsp0p1-selected-no-stop-v1/
  rsp0p1p2-selected-no-stop-v1/
```

后续 `EP=4/8` 评估入口：

```text
integrations/megatron_ep/exp_phase_executor.py
```

建议只在现有 ABI 上增加 policy，不要再改 executor / adapter。

## Archive Record

Source archive：

- filename: `routersense_policy_library_src_20260705_v1.tar.gz`
- sha256: `a21315e00818d123e6fb9b2d218d6180e4741e5603bee962493ca161c19617d1`

Results archive：

- filename: `routersense_policy_library_results_20260705_v1.tar.gz`
- sha256: `01ad283205c498cab6b32d3462e03a4855f91ee732319c24ff8ca20625e64551`

Archive self-check：

- unpack root: `/tmp/routersense-policy-library-archive-check/RS`
- command: `PYTHONPATH=. python -m pytest -q integrations/megatron_ep/tests`
- result: `72 passed in 27.15s`

## Commit-Snapshot Notes

- This snapshot freezes the pluggable policy runtime v1 line.
- It includes synthetic validation and `EP=2` selected-layer real-executor validation.
- It does not include any approved performance claim.
