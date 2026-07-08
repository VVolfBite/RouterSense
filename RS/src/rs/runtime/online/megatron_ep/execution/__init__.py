"""执行面子包入口。

这一层放：
- transport_adapter：把计划落成真实 collectives
- sync_wave_executor：按 wave 顺序驱动执行
- audit / layout helpers：执行后校验
"""

from .bucketizer import bucketize_transfer_layouts
from .layout_validation import row_digest, validate_phase_execution_plan
from .sync_wave_executor import PhaseExecutionResult, execute_scheduled_phase_tensor
from .transport_adapter import HostAPIDriftError, MegatronPhaseTransportAdapter

__all__ = [
    "HostAPIDriftError",
    "MegatronPhaseTransportAdapter",
    "PhaseExecutionResult",
    "bucketize_transfer_layouts",
    "execute_scheduled_phase_tensor",
    "row_digest",
    "validate_phase_execution_plan",
]
