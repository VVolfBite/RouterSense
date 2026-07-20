# RouterSense authoritative documentation

Only the documents listed here are active. Historical handoffs, recovery notes,
round checkpoints, migration inventories, and stage-specific GPU command sheets
were removed from the deployment mainline because they described retired code
paths and could mislead an execution agent.

## Deployment

- [`../task-test-deploy.md`](../task-test-deploy.md): the only operational
  handoff for Codex/PPIO execution.
- [`../deploy/README.md`](../deploy/README.md): inventory fields, dry-run, apply,
  failure handling, and result locations.
- [`architecture/current_code_structure.md`](architecture/current_code_structure.md):
  current package and runtime boundaries.

## Runtime and planning contracts

- `P012_ORTHOGONAL_PLANNER_AXES.md`
- `P012_P0123_FUTURE_P012.md`
- `runtime_joint_async_design.md`
- `runtime_online_hotpath_contract.md`
- `async_release_runtime_contract.md`
- `predictor_contract.md`
- `architecture/runtime_contracts.md`
- `architecture/scheduling_policy_contract.md`
- `architecture/scheduling_prediction_closure.md`

## Evaluation contracts

- `evaluation/README.md`
- `experiments/official_workflows.md`
- `experiments/output_schema.md`
- `experiments/result_eligibility.md`
- `router_trace_schema.md`
- `runtime_replay_trace_contract.md`

Documents not listed above are research references, not deployment instructions.
