"""routesense_poc1 package."""

from .adapter_registry import MoERouteAdapter, UnsupportedOLMoEAdapter, get_adapter_for_layer
from .config import build_config, ensure_output_dir, load_config, parse_args
from .metrics import next_token_nll
from .model_inspector import inspect_model, select_middle_moe_layer
from .model_loader import ModelLoadError, load_olmoe_model
from .route_patch import (
    MoERouteAdapter,
    MockRouteAblationContext,
    RoutingState,
    RouteAblationContext,
    RoutePatchError,
    UnsupportedOLMoEAdapter,
)
from .output_archive import archive_previous_outputs
from .router_trace import MockRouterTraceProvider, collect_router_trace
from .schemas import (
    AblationResult,
    CorrectnessCheckResult,
    EnvironmentInfo,
    MoELayerInfo,
    RouterTrace,
    RouteSnapshot,
    RunConfig,
)
from .serialization import load_json, save_json
from .utils import ensure_directory, stable_hash

__all__ = [
    "AblationResult",
    "CorrectnessCheckResult",
    "archive_previous_outputs",
    "EnvironmentInfo",
    "MoELayerInfo",
    "ModelLoadError",
    "MoERouteAdapter",
    "MockRouterTraceProvider",
    "MockRouteAblationContext",
    "MoERouteAdapter",
    "RoutingState",
    "RouteAblationContext",
    "RoutePatchError",
    "UnsupportedOLMoEAdapter",
    "get_adapter_for_layer",
    "UnsupportedOLMoEAdapter",
    "RouteSnapshot",
    "RouterTrace",
    "RunConfig",
    "build_config",
    "collect_router_trace",
    "ensure_directory",
    "ensure_output_dir",
    "inspect_model",
    "load_config",
    "load_json",
    "load_olmoe_model",
    "next_token_nll",
    "parse_args",
    "save_json",
    "select_middle_moe_layer",
    "stable_hash",
]
