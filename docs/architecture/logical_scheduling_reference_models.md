# Logical scheduling and reference models

Formal RouterSense scheduling uses the same fixed-placement traffic matrices,
canonical bucket semantics and full-duplex matching constraints across Local
and Joint comparisons. Only the scope axis changes in a strict scope test.

Offline paper references live under `rs.reference.baselines`:

- `fast_stage_reference`
- `aurora_order_reference`
- `islip_reference`
- `birkhoff_fluid_reference`

They are comparison models, not online runtime policies. Exact references are
`oracle_local_exact` and `oracle_joint_exact`; unsupported exact problem sizes
fail closed rather than returning an uncertified value.
