# Config Inventory

## Official Configs

- `configs/official/offline_replay.yaml`
- `configs/official/online_phase_sync.yaml`
- `configs/official/online_async_release.yaml`
- `configs/official/gpu_c2_correctness.yaml`
- `configs/official/gpu_a2_performance.yaml`
- `configs/official/evaluation_matrix.yaml`

## Reusable Components

- `configs/components/models/olmoe_1b_7b_instruct.yaml`
- `configs/components/topologies/local_4gpu.yaml`
- `configs/components/workloads/comparison_64_prompts.json`

## Legacy / Historical Config Roots

- `configs/comparison/`
- `configs/experiment/`
- `configs/offline/`
- `configs/model/`
- `configs/topology/`
- `configs/workload/`

Those roots remain in-tree for historical experiments and compatibility, but formal public workflows should resolve through `configs/official/` and `configs/components/`.

