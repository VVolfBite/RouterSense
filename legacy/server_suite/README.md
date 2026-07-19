# Archived server-suite prototype

This directory is a recovered prototype from the pre-mainline
`routersense_sched` package. That package and its overlay driver are not part of
the current `rs` mainline, so these commands are intentionally excluded from
the active deployment surface.

Use the maintained inventory-driven entrypoints instead:

```bash
bash scripts/deploy/launch_remote.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_cluster_access.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_repo_parity.sh deploy/inventory/hosts.local.yaml
bash deploy/remote/run_formal_experiment.sh --help
```

The files remain here only for historical comparison and must not be used as
formal deployment evidence.
