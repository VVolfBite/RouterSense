# Result Eligibility

Formal offline, C2, and A2 reports are fail-closed.

`validate_report_eligibility(...)` rejects runs when any of the following is true:

- `manifest.status` or `status.json` is not `completed` / `success`;
- `git_dirty=true`;
- manifest and runtime SHA disagree;
- `valid_for_evaluation=false`;
- fallback count is non-zero;
- timeout count is non-zero;
- offline `audit_invalid_count` is non-zero;
- legacy compiler bridge or compiler shadow compare was used;
- required summary metrics are missing;
- A2 is requested without C2 eligibility.

`experiments/reporting/build_report.py --allow-invalid-diagnostic` can still render a diagnostic report, but it is explicitly marked `NOT VALID FOR PERFORMANCE COMPARISON`.
