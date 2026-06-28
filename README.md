# RouteSense

RouteSense is a distributed MoE scheduling and deployment project. The current mainline lives under `RS/` and focuses on real single-GPU deployment bring-up, router trace capture, and the path toward controlled multi-GPU execution.

## Mainline

- `RS/src/routesense/`: formal runtime, trace, topology, oracle, and evaluation code.
- `RS/deploy/`: inventory, dry-run launch contracts, and deployment scripts.
- `RS/experiments/deployment/`: single-GPU smoke and trace entrypoints.
- `RS/configs/`: model and topology configuration.
- `RS/tests/`: deployment-side regression tests.

## Legacy

- `legacy/poc1/`: router observability and proxy-era experiments.
- `legacy/poc2/`: single-node NCCL correctness and harness diagnostics.
- `legacy/shared/`: shared historical docs.

The legacy trees are preserved for reference and reproducibility. They are not the formal deployment mainline.

## Current Phase

Phase 0B validates real single-GPU OLMoE loading and router trace capture on the executor host, while the control host maintains inventory and dry-run launch contracts. No scheduler benchmark is part of this phase.

