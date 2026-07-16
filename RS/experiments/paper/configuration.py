from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "claim_scope",
    "models",
    "inputs",
    "layers",
    "virtual_ep_sizes",
    "physical_world_size",
    "policies",
    "predictors",
    "cost_model",
    "seeds",
    "splits",
    "measurement",
    "eligibility",
    "output",
)


def validate_paper_config(config: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in config]
    if missing:
        raise ValueError(f"missing paper config fields: {missing!r}")
    if not isinstance(config["inputs"], dict):
        raise ValueError("inputs must be a mapping")
    if not isinstance(config["models"], dict):
        raise ValueError("models must be a mapping")
    if not isinstance(config["seeds"], dict):
        raise ValueError("seeds must be a mapping")
    if not isinstance(config["measurement"], dict):
        raise ValueError("measurement must be a mapping")
    if not isinstance(config["eligibility"], dict):
        raise ValueError("eligibility must be a mapping")
    if not isinstance(config["output"], dict):
        raise ValueError("output must be a mapping")
    if not isinstance(config["policies"], list):
        raise ValueError("policies must be a list")
    if not isinstance(config["predictors"], list):
        raise ValueError("predictors must be a list")
    if not isinstance(config["virtual_ep_sizes"], list):
        raise ValueError("virtual_ep_sizes must be a list")


def consumed_config_payload(config: dict[str, Any], *, output_dir: Path, input_path: str | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": config["schema_version"],
        "claim_scope": config["claim_scope"],
        "models": dict(config["models"]),
        "inputs": dict(config["inputs"]),
        "layers": config["layers"],
        "virtual_ep_sizes": list(config["virtual_ep_sizes"]),
        "physical_world_size": int(config["physical_world_size"]),
        "policies": list(config["policies"]),
        "predictors": list(config["predictors"]),
        "cost_model": config["cost_model"],
        "seeds": dict(config["seeds"]),
        "splits": dict(config["splits"]),
        "measurement": dict(config["measurement"]),
        "eligibility": dict(config["eligibility"]),
        "output": {"dir": str(output_dir)},
    }
    if input_path is not None:
        payload["inputs"]["resolved_input"] = str(input_path)
    return payload
