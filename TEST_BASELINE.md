## RouterSense Frozen Test Baseline

- Baseline branch: `convergence/m123-integration`
- Baseline commit: `b782893721e7069bc38d90a9b12bd3454012b402`
- Clone root: `RS/repo`
- RouterSense source root: `RS/repo/RS`
- Remote: `https://github.com/VVolfBite/RouterSense.git`
- Git clean requirement: `git status --short` must be empty before and after every formal test run

## Test machine

- Date: 2026-07-16
- OS: Windows 11
- Python: 3.12.6
- PyTorch: 2.7.0+cu118
- CUDA runtime: 11.8
- Visible GPU count: 1

## Model path policy

- Model weights stay outside `RS/repo` and outside `RS/output`
- Formal runs must reference model assets through config or environment variables
- No model checkpoint, tokenizer cache, or weight shard is copied into the repository or final evidence ZIP

## Formal configs

- `RS/configs/official/offline_replay.yaml`
- `RS/configs/official/online_phase_sync.yaml`
- `RS/configs/official/online_async_release.yaml`
- `RS/configs/official/gpu_c2_correctness.yaml`
- `RS/configs/official/gpu_first_bringup.yaml`
- `RS/configs/official/gpu_hotpath_iteration.yaml`
- `RS/configs/official/gpu_runtime_diag.yaml`
- `RS/configs/official/gpu_runtime_timeline.yaml`
- `RS/configs/official/gpu_runtime_attribution.yaml`
- `RS/configs/official/gpu_shadow_retire_check.yaml`
- `RS/configs/official/gpu_a2_performance.yaml`
- `RS/configs/official/evaluation_matrix.yaml`

## Formal runners

- `RS/experiments/run_offline_replay.py`
- `RS/experiments/run_online_phase_sync.py`
- `RS/experiments/run_online_async_release.py`
- `RS/experiments/distributed/run_gpu_first_bringup.py`
- `RS/experiments/distributed/run_gpu_c2_async_correctness.py`
- `RS/experiments/distributed/run_gpu_a2_strategy_compare.py`

## Output delivery contract

- Temporary run outputs go to a system temp directory or an explicit staging directory outside the Git working tree
- Final user delivery goes only to `RS/output`
- Each test round produces one final ZIP
- The final ZIP must include:
  - `README.txt`
  - `run_manifest.json`
  - `git/branch.txt`
  - `git/commit.txt`
  - `git/status.txt`
  - `git/remote.txt`
  - `commands/commands.txt`
  - `results/result_bundle.json`
  - `checksums.sha256`

## Code modification discipline

- Default state from this baseline: code frozen
- Test failures are recorded, reproduced, classified, and reported before any code change is considered
- A code change is allowed only for a confirmed software bug with:
  - a stable reproduction command
  - clear expected vs actual behavior
  - a bounded code path
  - a minimal regression test
  - no algorithm-semantics change
