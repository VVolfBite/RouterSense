# POC2 Stress Suite Contract

POC2.7 不修改真实 router route item，不修改 token、payload、expert compute，也不修改 origin-aware NCCL 语义。
唯一允许的扰动是：

- trace-preserving expert-to-rank placement skew
- trace-derived microbatch replay / episode ordering
- scheduler release order 和 release-round composition

所有 strategy 在同一个 scenario / repetition 中必须保持：

- same trace bundle
- same token-origin map
- same route-item set
- same payload checksum
- same metadata checksum
- same expert-weight checksum
- same total dispatch matrix
- same total return matrix

唯一允许变化的是：

- release_order
- release_rounds
- per-round source->destination submatrix

## Stage Order

1. Stage 1：生成 calibration / evaluation split、placement-skew ladder、burst episode schedule
2. Stage 2：先做 FIFO preflight，验证 skew 强度、manifest conservation、remote traffic
3. Stage 3：先做 dependency predictiveness gate 和 oracle round diagnostic
4. Stage 4：只做 FIFO / strong-state / full 的短机制诊断
5. Stage 5：只有 mechanism lever present 时才启动完整四策略长窗口 benchmark

## Scenario Levels

- Scenario 0：balanced negative control
- Scenario 1：mild-skew
- Scenario 2：moderate-skew
- Scenario 3：high-skew
- Scenario 4：trace-derived temporal burst episode

强度不是按“移动几个 expert”定义，而是按 replay 后实际观测到的：

- target_rank_inbound_byte_share
- target_rank_compute_row_share
- target_rank_peak_expert_share

## Dependency Gate

先比较：

- Model S：只用 strong-state 可见特征预测 t+1 hotspot
- Model S+D：strong-state 特征 + dependency 特征

若 S+D 对 S 没有明确 held-out 增益，则：

- `DEPENDENCY_PREDICTIVENESS_GATE = FAIL`

此时仍可运行 placement-skew 的 FIFO/random/strong-state 诊断，但不得把 full 的潜在优势写成 dependency 证据。

## Oracle Diagnostic

`oracle-minimax-round-packer` 只用于回答：

- 当前 synchronous release-round action space 是否存在物理杠杆

它可以看完整 scenario 的未来 flow，但不能修改：

- token origin
- destination
- expert
- total matrix
- payload bytes

它不是论文 baseline。
