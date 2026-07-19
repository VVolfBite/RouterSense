from __future__ import annotations

"""Cluster topology utilities.

This package is used by deployment-oriented tooling to parse inventory files,
derive per-node paths, and render multi-node launch commands. It is part of the
deployment/link-smoke path rather than the scheduler simulation stack.
"""

from .inventory import (
    Inventory,
    NodeSpec,
    RendezvousSpec,
    inventory_cli_summary,
    inventory_paths,
    inventory_summary,
    load_inventory,
    render_torchrun_dry_run,
)
from .paths import (
    resolve_inventory_path,
    resolve_model_path_for_node,
    resolve_node_artifact_root,
    resolve_node_model_cache,
    resolve_node_rs_root,
    resolve_preferred_model_path,
    resolve_rs_root,
)

from .model_cache import (
    DEFAULT_DEPLOYMENT_MODEL_ID,
    ModelCacheInspection,
    inspect_model_cache,
    model_name,
    resolve_model_directory,
)
