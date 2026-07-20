# Archive Index

The formal public surface no longer points users to the large historical runner set under:

- `experiments/offline/`
- `experiments/online/`
- `experiments/distributed/`

Those directories remain as internal implementation, historical reproduction, or diagnostic modules until each script is either:

- moved to `experiments/archive/`,
- reduced to a compatibility wrapper, or
- safely deleted after reference audit.

Current explicit compatibility wrapper:

- `experiments/dev/run_gpu_validation.py` -> forwards to `experiments/dev/run_validation.py`

