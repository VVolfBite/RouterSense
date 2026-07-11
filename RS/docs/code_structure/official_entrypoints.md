# Official Entrypoints

Public entrypoints are fixed to three scripts:

1. `experiments/run_offline_replay.py`
2. `experiments/run_online_phase_sync.py`
3. `experiments/run_online_async_release.py`

Validation entrypoints are fixed to three scripts:

1. `experiments/distributed/run_stage1_gloo_e2e_gate.py`
2. `experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py`
3. `experiments/dev/run_gpu_validation.py`

`run_gpu_validation.py` dispatches to the existing `B2/C2/A2` workers and is the only validation surface that should be documented publicly for GPU correctness/performance gates.
