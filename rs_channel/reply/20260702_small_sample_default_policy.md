## Small-Sample Validation Policy

Effective immediately for POC-line1 scheduler experiments:

- Default validation should use small prompt/sample caps first.
- Recommended first-pass sizes:
  - `sample_limit = 32`
  - `sample_limit = 64`
- Do **not** start with batch500 by default.

## Reason

Recent runs showed that weak candidates can be rejected quickly on small samples:

- `phase_aware_greedy` underperformed immediately
- `lagrangian` underperformed immediately

That means large batch500 runs are unnecessary until a candidate already looks promising on:

- 32 samples
- then 64 samples

## Code Changes

Updated defaults:

- `RS/experiments/poc_line1/pairwise_scheduler.py`
  - `--sample-limit` default is now `32`
  - help text explicitly recommends `32/64` first

- `RS/experiments/poc_line1/exp_pairwise_model_compare.py`
  - `--sample-limit` default is now `32`
  - help text explicitly recommends `32/64` first

Also updated `pairwise_scheduler.py` strategy summary to reflect the current fast candidate set:

- `phase_aware_greedy`
- `birkhoff`
- `lagrangian`
- `iterated_greedy`

## Current Recommended Workflow

1. Run `sample_limit=32`
2. If promising, rerun with `sample_limit=64`
3. Only then consider larger batch evaluation
