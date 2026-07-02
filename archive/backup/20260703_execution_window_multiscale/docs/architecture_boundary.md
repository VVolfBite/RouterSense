# Architecture Boundary

`RS/` is the formal mainline.

## Legacy Scope

- `legacy/poc1/`: router observability and proxy verification history.
- `legacy/poc2/`: single-node NCCL harness history.
- `legacy/shared/`: shared historical docs.

## Rule

`RS/` must not depend on legacy code paths for runtime or scheduler behavior. If a utility is reused, it must be copied into `RS/` and retested there.

