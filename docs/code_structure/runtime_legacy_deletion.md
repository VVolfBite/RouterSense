**Runtime Legacy Deletion**

已完成：
- `RouterSensePolicy` / `RouterSensePhasePolicy` / 旧 `SchedulingPolicy` 已从 [base.py](/root/autodl-tmp/RouterSense/RS/src/rs/scheduling/base.py) 抽离到 [legacy_interfaces.py](/root/autodl-tmp/RouterSense/RS/src/rs/scheduling/legacy_interfaces.py)。
- [base.py](/root/autodl-tmp/RouterSense/RS/src/rs/scheduling/base.py) 现在只保留 deprecated bridge。

仍保留的 legacy bridge：
- `pending_window/policy_adapter.py`
- `resolve_phase_policy()` 的 legacy phase-local builder 路径
- `async_release/*` 中的 shadow/state-only 研究模块
