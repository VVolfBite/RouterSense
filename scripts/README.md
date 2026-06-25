# Scripts Layer

`scripts/` is now a compatibility and platform-utility layer.

Primary POC1 experiment entrypoints live in:

```text
experiment/poc1/
```

Use `scripts/` for:

- legacy wrappers such as `trace_one_token.py` and `run_action.py`
- platform smoke scripts such as `run_poc1_linux.sh` and `run_poc1_windows.ps1`
- environment and preflight utilities such as `setup_env.sh` and `preflight_gpu.py`
- implementation helpers still reused by thin experiment wrappers, such as `summarize_trace_batch.py` and `analyze_proxy_scores.py`

Do not add new main experiment entrypoints here. Add new POC1 experiment commands under `experiment/poc1/`, and keep `scripts/` as thin compatibility glue unless a platform utility really belongs here.
