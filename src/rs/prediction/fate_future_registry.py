"""Deprecated compatibility shim. Use :mod:`rs.prediction.asset_registry`."""
from .asset_registry import merge_predictor_specs as merge_fate_predictor_specs
from .asset_registry import resolves_predictor as resolves_fate_predictor
from .fate_future import FATE_PREDICTOR_SPEC

def create_registered_fate(config=None):
    from .asset_registry import create_predictor
    return create_predictor(FATE_PREDICTOR_SPEC.predictor_id, config)

__all__ = [
    "FATE_PREDICTOR_SPEC", "create_registered_fate",
    "merge_fate_predictor_specs", "resolves_fate_predictor",
]
