# Validation Workflows

`experiments/dev/run_validation.py` is the single public validation entrypoint.

## Suites

- `offline-smoke`: small official offline replay run
- `config`: canonical config normalization tests
- `catalog`: public catalog / entrypoint smoke
- `compiler`: unified scheduling/compiler contract tests
- `transport`: async runtime closure tests
- `gloo`: low-memory runtime-integrated Gloo gate
- `b2`: GPU B2 workflow dry-run by default in no-GPU environments
- `c2`: GPU C2 workflow dry-run by default in no-GPU environments
- `a2`: GPU A2 workflow dry-run by default in no-GPU environments

