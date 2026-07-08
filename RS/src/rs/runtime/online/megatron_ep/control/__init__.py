"""控制面子包入口。

控制面当前分三类：
- 正式热路径：agreement_wire、plan_agreement、p2_provider
- shadow / native-order 兼容路径：shadow_policy/*
- 早期异步 control-plane 试验残留：mailbox / state_machine / timeline
"""

from .p2_contracts import P2HintMetadata, P2HintRequest
from .p2_provider import build_p2_hint_provider, extract_prepared_plan_priority
from .plan_agreement import run_phase_plan_agreement

__all__ = [
    "P2HintMetadata",
    "P2HintRequest",
    "build_p2_hint_provider",
    "extract_prepared_plan_priority",
    "run_phase_plan_agreement",
]
