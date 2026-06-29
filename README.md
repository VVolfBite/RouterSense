# RouteSense

RouteSense is a distributed MoE scheduling and deployment project. The current mainline lives under `RS/` and focuses on real single-GPU deployment bring-up, router trace capture, and the path toward controlled multi-GPU execution.

## Mainline

- `RS/src/routesense/`: formal runtime, trace, topology, oracle, and evaluation code.
- `RS/deploy/`: inventory, dry-run launch contracts, and deployment scripts.
- `RS/experiments/prerun/`: single-GPU smoke, architecture probe, and router trace entrypoints.
- `RS/experiments/distributed/`: distributed bring-up scripts.
- `RS/experiments/ablation/`: reserved real ablation configs and scripts.
- `RS/configs/`: model and topology configuration.
- `RS/tests/`: deployment-side regression tests.

## Legacy

- `legacy/poc1/`: router observability and proxy-era experiments.
- `legacy/poc2/`: single-node NCCL correctness and harness diagnostics.
- `legacy/shared/`: shared historical docs.

The legacy trees are preserved for reference and reproducibility. They are not the formal deployment mainline.

## Current Phase

The current RS mainline is being refactored toward real distributed OLMoE expert-parallel bring-up. Legacy POC1/POC2 are archived references only and are not part of deployment or runtime mainline execution.
