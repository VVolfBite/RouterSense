# Archive

This directory is for small milestone manifests, SHA256 records, environment
fingerprints, and milestone summaries only.

Do not store:

- raw tensors
- model checkpoints
- full NCCL logs
- large trace dumps

Those remain under `artifacts/` or external object storage.
The Round 1 removed-source snapshot is intentionally excluded from deployment source archives.
Only small migration manifests and fingerprints belong in a formal handoff.
