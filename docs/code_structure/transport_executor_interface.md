# Transport Executor Interface

正式 transport facade 位于 [src/rs/runtime/online/megatron_ep/execution/executor_facade.py](/root/autodl-tmp/RouterSense/RS/src/rs/runtime/online/megatron_ep/execution/executor_facade.py)。

统一外部接口：

```python
class TransportExecutor(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...
```

当前 backend：

- `PhaseSyncTransportExecutor`
- `AsyncReleaseTransportExecutor`

统一输入：

- `execution_plan`
- `phase_context`
- `tensor_role`
- `input_tensor`
- `process_group`
- `rank_context`

统一输出：

- `output_tensor`
- `backend_id`
- `execution_plan_digest`
- `send_op_count`
- `recv_op_count`
- `local_copy_task_count`
- `local_copy_row_count`
- `enqueue_us`
- `wait_us`
- `total_us`
- `fallback_used`
- `timeout`

本阶段没有修改底层 phase-sync 或 async P2P 的执行顺序，只是把 runtime adapter 的外部调用统一到 `execute_transport(...)`。
