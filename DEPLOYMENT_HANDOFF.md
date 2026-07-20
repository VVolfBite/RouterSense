# RouterSense deployment handoff

This is the only operational instruction for Codex/PPIO execution.

## Role boundary

Codex/PPIO is an execution agent. It may create/start/stop instances, configure
PPIO networking, mount the cloud model volume, obtain internal and SSH
endpoints, fill one inventory file, and run the commands below.

It must not modify Python source, planner parameters, experiment YAML, model
files, or result JSON. On failure it stops and returns the pipeline report and
logs. Bug diagnosis and configuration interpretation remain with the primary
RouterSense analysis workflow.

## Local-controller model

- The user's local checkout is authoritative and must be clean and committed.
- Source is transferred to temporary servers as a Git bundle.
- Servers only install support dependencies and run experiments.
- Raw and summarized results are copied back to the local checkout.
- The model snapshot remains on PPIO cloud storage and is referenced by an
  absolute mounted path on every node.

## Select one inventory

For the first single-node validation:

```bash
cp deploy/inventory/hosts.1x4.example.yaml deploy/inventory/hosts.local.yaml
```

For the first multi-node validation:

```bash
cp deploy/inventory/hosts.2x2.example.yaml deploy/inventory/hosts.local.yaml
```

Fill only the placeholder values. `host` is the PPIO internal/data-plane IP.
`ssh_host` is the management endpoint when different. `model_cache` is the
mounted model snapshot directory or its parent. The same model content must be
visible on every node.

## Dry-run

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-preflight
```

The dry-run must return `DRY_RUN_PASS` before applying.

## Apply

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --apply \
  --run-id rs-1x4-smoke
```

For 2x2 use a distinct run ID such as `rs-2x2-smoke`.

The pipeline performs: access check, clean source distribution, source parity,
Python support dependency installation, model parity, mounted-model load
preflight, CUDA/NCCL/Megatron runtime preflight, measured pairwise GPU link
calibration, experiment launch, result collection, and local result validation.

## Return artifacts

Return these files without editing them:

- `outputs/deployment_pipeline/<run-id>/pipeline_report.json`
- `outputs/deployment_pipeline/<run-id>/logs/`
- `outputs/deployment/<run-id>/collection_manifest.json`
- `outputs/deployment/<run-id>/deployment_result_summary.json`
- the complete `outputs/deployment/<run-id>/` directory
- `outputs/deployment_profiles/<run-id>/link_cost_profile.json`

A missing profile, model mismatch, source mismatch, nonzero rank exit, fallback,
invalid audit, or missing master result is a hard failure.
