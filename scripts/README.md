# Scripts Layer

`scripts/` is a compatibility and utility layer for the repository root.

Use it for:

- source packaging and archive verification;
- environment preflight and install helpers;
- thin wrappers that keep historical entrypoints runnable during migration.

Do not add new mainline RouteSense code here. Formal implementation lives under `RS/`.
Legacy POC1/POC2 wrappers stay here only if they are needed for reproducibility or compatibility.
