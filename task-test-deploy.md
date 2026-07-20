# RouterSense task-test-deploy

This is the sole execution task for Codex using the PPIO MCP tools.

## 1. Role boundary

Codex is an execution agent. It may:

- create/start/stop PPIO instances;
- enable or inspect the private network;
- mount the prepared cloud-storage model volume;
- obtain internal IPs, SSH endpoints and users;
- fill `deploy/inventory/hosts.local.yaml`;
- run the commands in this document;
- return logs and results.

Codex must not:

- modify Python source;
- change planner/core parameters;
- edit official experiment YAML;
- alter model files or result JSON;
- mask, delete or reinterpret errors;
- continue to a larger run after a failed smaller run.

When a run fails, stop. The RouterSense analysis workflow diagnoses the cause.

## 2. Prepare the local checkout

From the extracted release bundle, run:

```bash
bash bootstrap_from_gitbundle.sh
cd RouterSense
```

Confirm `git status --short` is empty. Do not deploy directly from the standalone
source ZIP because it intentionally has no `.git` metadata for source transfer.

## 3. Provisioning order

Run only these stages in order:

1. **1×4 validation**: one server, four GPUs;
2. **2×2 validation**: two servers, two GPUs per server, only after 1×4 passes.

Do not start 1×8, 2×6, 2×8 or 3×8 in this task.

For every PPIO node, confirm before running code:

- the requested GPU count is visible;
- nodes in a multi-node run can reach each other over internal IPs;
- SSH from the local controller works;
- the cloud-storage model volume is mounted;
- the same model snapshot is visible on every node.

## 4. Create the inventory

For 1×4:

```bash
bash scripts/deploy/init_inventory.sh 1x4
```

For 2×2 after the 1×4 run is accepted:

```bash
bash scripts/deploy/init_inventory.sh 2x2 --force
```

Edit only the placeholder values in
`deploy/inventory/hosts.local.yaml`:

```yaml
nodes:
  - host: INTERNAL_IP
    ssh_host: SSH_ENDPOINT
    gpu_count: 4
    model_path: /ABSOLUTE/MOUNTED/MODEL/PATH
```

For two nodes, fill the same four values for both entries.

Rules:

- delete `ssh_host` when SSH uses the internal IP;
- add `ssh_user` only when it is not `root`;
- add `ssh_port` only when it is not `22`;
- do not add node ranks, rendezvous, remote source paths or artifact paths;
- `model_path` must refer to the mounted local filesystem view, not an HTTP URL.

## 5. Dry-run

Use a unique run ID:

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-preflight
```

Acceptance condition:

```text
DRY_RUN_PASS
```

If dry-run fails, stop and return:

```text
outputs/deployment_pipeline/rs-preflight/failure_summary.txt
outputs/deployment_pipeline/rs-preflight/pipeline_report.json
outputs/deployment_pipeline/rs-preflight/logs/
```

Do not run with `--apply`.

## 6. Apply 1×4

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-1x4-smoke \
  --apply
```

The pipeline itself performs:

1. inventory and SSH checks;
2. clean Git-bundle source distribution and commit/tree parity;
3. support dependency installation;
4. mounted-model structure, shard, tokenizer and local-only load checks;
5. CUDA, NCCL, Megatron Core/Bridge and Transformer Engine checks;
6. directed intra-node GPU link calibration;
7. immutable link-cost profile validation and planner injection;
8. torchrun experiment launch;
9. remote exit-marker capture;
10. local result collection and fail-closed validation.

Do not manually skip link calibration or model/runtime preflight.

## 7. Apply 2×2

Only after 1×4 has final `PASS`, recreate the 2×2 inventory and repeat the
dry-run. Then run:

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-2x2-smoke \
  --apply
```

The two `host` values must be internal IPs used by torchrun/NCCL. Public SSH
endpoints belong only in `ssh_host`.

## 8. Failure rule and logs

On any nonzero exit, timeout, missing result, fallback, model mismatch, source
mismatch, profile mismatch or execution-audit failure:

1. stop the run;
2. do not edit code or configuration beyond correcting a plainly mistyped
   inventory placeholder;
3. return the following without modification:

```text
outputs/deployment_pipeline/<run-id>/failure_summary.txt
outputs/deployment_pipeline/<run-id>/pipeline_report.json
outputs/deployment_pipeline/<run-id>/logs/
outputs/deployment/<run-id>/
outputs/deployment_profiles/<run-id>/link_cost_profile.json   # when generated
```

`failure_summary.txt` is the first file to read. It records why the pipeline
failed and includes the relevant log tails.

## 9. Success artifacts

For each successful applied run, return:

```text
outputs/deployment_pipeline/<run-id>/run_summary.txt
outputs/deployment_pipeline/<run-id>/pipeline_report.json
outputs/deployment_pipeline/<run-id>/logs/
outputs/deployment/<run-id>/collection_manifest.json
outputs/deployment/<run-id>/deployment_result_summary.json
outputs/deployment/<run-id>/
outputs/deployment_profiles/<run-id>/link_cost_profile.json
```

A run is not accepted merely because torchrun started. The final
`deployment_result_summary.json` must report `PASS` and no fallback, timeout,
missing rank result or failed execution audit.

## 10. Completion report

Codex should report only:

- PPIO instance IDs and topology used;
- filled inventory values with model path but no secrets;
- run ID;
- dry-run status;
- applied-run status;
- exact artifact paths returned;
- first failed stage and `failure_summary.txt` path when not successful.
