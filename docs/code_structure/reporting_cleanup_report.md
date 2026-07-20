# Reporting Cleanup Report

- Structured reporting is unified behind `experiments/reporting/build_report.py`.
- Reporting logic is moved to `src/rs/reporting/`.
- Official reports read structured metrics from run directories rather than re-running experiments.
- Legacy root-level report files remain readable, but new official reports are emitted under `reports/`.
