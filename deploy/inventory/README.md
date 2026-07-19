# Deployment inventory

Copy `hosts.example.yaml` to the git-ignored `hosts.local.yaml` and replace all
placeholder values.  `remote_rs_root` must be the actual RouterSense `RS/`
repository root on that node, not its parent directory.

`host` is the address used by the distributed rendezvous/data plane.
`ssh_host` may be added when SSH uses a different management address.  The
formal two-node smoke expects two target GPUs per node.

The private inventory is not committed.  `scripts/deploy/sync_repo.py` copies it
to the same relative path in each remote checkout after synchronizing the clean
Git commit.
