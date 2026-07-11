# Experiment Cleanup Report

- Official public experiment entrypoints are now limited to:
  - `experiments/run_offline_replay.py`
  - `experiments/run_online_phase_sync.py`
  - `experiments/run_online_async_release.py`
- Validation now has one public entrypoint:
  - `experiments/dev/run_validation.py`
- Reporting now has one public entrypoint:
  - `experiments/reporting/build_report.py`
- `experiments/dev/run_gpu_validation.py` remains as a compatibility forwarder.

