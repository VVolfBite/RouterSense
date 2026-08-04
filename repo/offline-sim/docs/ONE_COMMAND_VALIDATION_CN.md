# 一键实验入口本地验收

## 验收命令

```bash
python scripts/run_formal_experiment.py \
  --trace-root /path/to/measured/RouterSense_FATE_COLLECTION_V0712_20260731/OLMoE-1B-7B-0924/EP8/SEQ128 \
  --output-csv /path/to/results.csv \
  --preset smoke \
  --workers 2 \
  --task-bytes 4194304 \
  --max-fixtures 1
```

## 已验证行为

1. 无需 YAML；只需要 trace 根目录与输出 CSV。
2. 两个 isolated workers 并行运行时，由父进程独占写 CSV。
3. 每个 treatment 完成后立即 append、flush、fsync。
4. 完成 fixture 后追加 `TRACE_COMPLETE`。
5. 完整 OLMoE EP8 step0 smoke：3 treatments、15 个 P12 windows，43.104 秒，PASS。
6. 直接指向统一 trace 包顶层目录：自动发现 DeepSeek-V2-Lite / EP2 / SEQ1024 的首个完整 fixture；首次两条 Local 已落盘后恢复，补跑 RSCF-Joint，40.854 秒完成并追加 `TRACE_COMPLETE=PASS`。
7. 中断恢复：已有 FIFO 与 Birkhoff 两行后终止；相同命令重启仅补跑 RSCF-Joint，跳过 2 条已完成记录，并追加 `TRACE_COMPLETE=PASS`。
8. 完成后再次执行相同命令会按 trace complete key 跳过整个 fixture，约 0.22 秒返回。
9. parent 对 fixture 做一次完整校验；worker 使用 `PARENT_VALIDATED_TRUSTED_WORKER`，该 EP8 fixture 每个 worker 的验证阶段约 96–106 ms，而不是重新执行完整 invariants walk。

## CSV 事务边界

权威提交边界是 CSV 中完整的 `RUNTIME`、`FAILURE`、`ORACLE_VALIDATION` 或 `TRACE_COMPLETE` 行。`.progress.json` 仅用于观察进度；恢复逻辑以 CSV run key 为准。

10. worker 提交权威 status/result 后由父进程按 PID 清理残留数值运行时，不调用 `Popen.poll()` 或 `waitpid()`；结果行记录 `worker_forced_exit_after_status=true` 以便审计。
