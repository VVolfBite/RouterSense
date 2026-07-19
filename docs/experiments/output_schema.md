# Output Schema

All official entrypoints now initialize a unified output root:

```text
outputs/<run_id>/
  manifest.json
  environment.json
  config_snapshot.yaml
  status.json
  raw/
  metrics/
  reports/
  logs/
  failures/
```

Compatibility files such as legacy `summary.json` or `comparison_report.json` may still be written at the run root for historical tooling, but structured consumers should prefer:

- `manifest.json`
- `status.json`
- `metrics/*.json`
- `reports/*.json`

