Official RouterSense test configs

This directory contains the only formal configs used for frozen correctness testing.

- `offline_replay.yaml`
  - Purpose: official offline replay smoke and contract replay.
  - Environment: CPU.
  - Formal correctness: yes.
  - Timing eligible: no.
  - Runner: `experiments/run_offline_replay.py`.

- `online_phase_sync.yaml`
  - Purpose: official online phase-sync correctness path.
  - Environment: CPU or distributed runtime with the appropriate backend.
  - Formal correctness: yes.
  - Timing eligible: no by default.
  - Runner: `experiments/run_online_phase_sync.py`.

- `online_async_release.yaml`
  - Purpose: official online async-release correctness path.
  - Environment: CPU or distributed runtime with the appropriate backend.
  - Formal correctness: yes.
  - Timing eligible: no by default.
  - Runner: `experiments/run_online_async_release.py`.

- `gpu_c2_correctness.yaml`
  - Purpose: formal 4-GPU correctness gate.
  - Environment: 4 GPU, NCCL, model assets.
  - Formal correctness: yes.
  - Timing eligible: no.
  - Runner: `experiments/distributed/run_gpu_c2_async_correctness.py`.

- `gpu_first_bringup.yaml`
  - Purpose: first GPU environment bring-up.
  - Environment: GPU.
  - Formal correctness: diagnostic.
  - Timing eligible: no.
  - Runner: `experiments/distributed/run_gpu_first_bringup.py`.

- `gpu_hotpath_iteration.yaml`
  - Purpose: hotpath iteration and compact validation.
  - Environment: GPU.
  - Formal correctness: diagnostic/supporting.
  - Timing eligible: no.
  - Runner: `experiments/distributed/run_gpu_a2_strategy_compare.py`.

- `gpu_runtime_diag.yaml`
  - Purpose: runtime diagnostic capture.
  - Environment: GPU.
  - Formal correctness: diagnostic.
  - Timing eligible: no.
  - Runner: GPU diagnostic workflows.

- `gpu_runtime_timeline.yaml`
  - Purpose: runtime timeline capture.
  - Environment: GPU.
  - Formal correctness: diagnostic.
  - Timing eligible: no.
  - Runner: GPU timeline workflows.

- `gpu_runtime_attribution.yaml`
  - Purpose: runtime attribution capture.
  - Environment: GPU.
  - Formal correctness: diagnostic.
  - Timing eligible: no.
  - Runner: GPU attribution workflows.

- `gpu_shadow_retire_check.yaml`
  - Purpose: shadow-path retirement verification.
  - Environment: GPU.
  - Formal correctness: diagnostic/supporting.
  - Timing eligible: no.
  - Runner: shadow-retire validation workflows.

- `gpu_a2_performance.yaml`
  - Purpose: post-correctness GPU performance gate.
  - Environment: 4 GPU, NCCL, model assets.
  - Formal correctness: supporting only.
  - Timing eligible: yes, after correctness closure.
  - Runner: `experiments/distributed/run_gpu_a2_strategy_compare.py`.

- `evaluation_matrix.yaml`
  - Purpose: matrix of official evaluation combinations.
  - Environment: depends on referenced scenario.
  - Formal correctness: orchestrates official runs.
  - Timing eligible: depends on referenced scenario.
  - Runner: matrix/validation entrypoints.
