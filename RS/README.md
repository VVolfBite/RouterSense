# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/routesense/`: main runtime, trace, topology, oracle, and evaluation modules.
- `deploy/`: inventory, environment requirements, log handling, and remote launch scripts.
- `experiments/deployment/`: single-GPU smoke tests and real router-trace capture.
- `configs/`: model and topology configuration.
- `docs/`: formal deployment and trace schema notes.
- `tests/`: mainline regression tests.

## Phase 0B

Phase 0B is the deployment bring-up stage. It validates real single-GPU OLMoE inference and router trace export on the executor host. It does not claim multi-node expert parallel or scheduler benchmarking.

