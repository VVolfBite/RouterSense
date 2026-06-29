"""Legacy PoC1 package for real route ablation experiments."""

from .config import build_config, ensure_output_dir, finalize_config, load_config, parse_args
from .intervention import RouteAblationContext, RoutePatchError, verify_intervention_correctness
from .metrics import next_token_nll
from .model_loader import ModelLoadError, load_olmoe_model
from .schemas import AblationRecord, MoELayerSpec, RoutingContext, RunConfig, Window

__all__ = [
    "AblationRecord",
    "ModelLoadError",
    "MoELayerSpec",
    "RouteAblationContext",
    "RoutePatchError",
    "RoutingContext",
    "RunConfig",
    "Window",
    "build_config",
    "ensure_output_dir",
    "finalize_config",
    "load_config",
    "load_olmoe_model",
    "next_token_nll",
    "parse_args",
    "verify_intervention_correctness",
]
