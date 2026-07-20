# Official Workflows

## Public Entrypoints

- `python experiments/run_offline_replay.py --config configs/official/offline_replay.yaml`
- `torchrun ... experiments/run_online_phase_sync.py --config configs/official/online_phase_sync.yaml`
- `torchrun ... experiments/run_online_async_release.py --config configs/official/online_async_release.yaml`

## Validation

- `python experiments/dev/run_validation.py --suite offline-smoke`
- `python experiments/dev/run_validation.py --suite config`
- `python experiments/dev/run_validation.py --suite catalog`
- `python experiments/dev/run_validation.py --suite compiler`
- `python experiments/dev/run_validation.py --suite transport`
- `python experiments/dev/run_validation.py --suite gloo`
- `python experiments/dev/run_validation.py --suite b2`
- `python experiments/dev/run_validation.py --suite c2`
- `python experiments/dev/run_validation.py --suite a2`

## Reporting

- `python experiments/reporting/build_report.py --input <run_dir> --report-type offline`
- `python experiments/reporting/build_report.py --input <run_dir> --report-type runtime_audit`
- `python experiments/reporting/build_report.py --input <run_dir> --report-type comparison`
- `python experiments/reporting/build_report.py --input <run_dir> --report-type c2`
- `python experiments/reporting/build_report.py --input <run_dir> --report-type a2`

